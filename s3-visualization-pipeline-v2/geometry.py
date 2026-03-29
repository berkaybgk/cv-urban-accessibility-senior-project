"""
Geometry engine — perspective-space sidewalk analysis.

Instead of warping to a bird's-eye view (which fails for lateral
sidewalks in GSV forward/backward images), this module works directly
in the perspective image:

  1. Per-row metric width via ray-casting: project left/right mask
     edges to the Z=0 ground plane, measure the 3D distance.
  2. Per-row rectification (v1-style): normalize each row's sidewalk
     span to a fixed canvas width, with vertical scaling based on the
     vanishing point.  This produces a recognisable "top-down strip"
     for visualization and obstacle footprint analysis.

Camera model
------------
  World: X = right, Y = forward, Z = up.
  Camera at (0, 0, h) looking along +Y.
  Pinhole: x_cam = right, y_cam = down, z_cam = forward.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ── Camera intrinsics ────────────────────────────────────────────────────────

def build_intrinsic_matrix(
    image_w: int, image_h: int, hfov_deg: float,
) -> np.ndarray:
    """3x3 intrinsic matrix from image size and horizontal FOV."""
    hfov_rad = math.radians(hfov_deg)
    f_x = (image_w / 2.0) / math.tan(hfov_rad / 2.0)
    c_x = image_w / 2.0
    c_y = image_h / 2.0
    return np.array([
        [f_x, 0.0, c_x],
        [0.0, f_x, c_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _build_rotation(pitch_deg: float) -> np.ndarray:
    """World-to-camera rotation matrix."""
    pitch_rad = math.radians(pitch_deg)
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)

    R_base = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0],
    ], dtype=np.float64)

    R_pitch = np.array([
        [1.0,   0.0,    0.0],
        [0.0,  cos_p, -sin_p],
        [0.0,  sin_p,  cos_p],
    ], dtype=np.float64)

    return R_pitch @ R_base


# ── Per-pixel ray-cast width ─────────────────────────────────────────────────

def compute_metric_widths(
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    valid: np.ndarray,
    K: np.ndarray,
    pitch_deg: float,
    camera_height_m: float,
    min_horizon_dist: int = 10,
) -> np.ndarray:
    """
    Compute metric sidewalk width (cm) per image row by ray-casting
    the left and right edge pixels to the ground plane.

    For each valid row, the left and right edge columns are projected
    through the camera model to the Z=0 plane.  The Euclidean distance
    between the two ground-plane hit points gives the width in metres.

    Rows too close to the horizon (< ``min_horizon_dist`` pixels below
    the computed horizon) are set to NaN because the projection blows up.

    Returns an array of width in centimetres, NaN for invalid rows.
    """
    R = _build_rotation(pitch_deg)
    R_inv = R.T
    K_inv = np.linalg.inv(K)
    H = len(left_edges)
    widths_cm = np.full(H, np.nan)

    # Horizon row: where a horizontal forward ray (world Y-axis) maps to
    cy = K[1, 2]
    fx = K[0, 0]
    horizon_row = cy - fx * math.tan(math.radians(pitch_deg))

    for r in range(H):
        if not valid[r]:
            continue
        if np.isnan(left_edges[r]) or np.isnan(right_edges[r]):
            continue
        if r < horizon_row + min_horizon_dist:
            continue

        left_col = left_edges[r]
        right_col = right_edges[r]

        pt_L = _ray_to_ground(left_col, r, K_inv, R_inv, camera_height_m)
        pt_R = _ray_to_ground(right_col, r, K_inv, R_inv, camera_height_m)

        if pt_L is None or pt_R is None:
            continue

        dist = math.sqrt((pt_R[0] - pt_L[0]) ** 2 + (pt_R[1] - pt_L[1]) ** 2)
        widths_cm[r] = dist * 100.0

    return widths_cm


def _ray_to_ground(
    col: float, row: float,
    K_inv: np.ndarray, R_inv: np.ndarray,
    camera_height_m: float,
) -> tuple[float, float] | None:
    """Project a pixel (col, row) to the Z=0 ground plane."""
    ray_cam = K_inv @ np.array([col, row, 1.0], dtype=np.float64)
    ray_world = R_inv @ ray_cam

    if ray_world[2] >= -1e-8:
        return None
    t = -camera_height_m / ray_world[2]
    if t < 0:
        return None
    return (t * ray_world[0], t * ray_world[1])


# ── Ray-cast obstacle footprint ──────────────────────────────────────────────

def compute_obstacle_ground_footprint(
    obs_mask: np.ndarray,
    K: np.ndarray,
    pitch_deg: float,
    camera_height_m: float,
    base_scan_ratio: float = 0.2,
) -> dict | None:
    """
    Compute the metric ground-plane footprint of an obstacle by
    ray-casting its bottom pixels.

    Scans the lowest ``base_scan_ratio`` fraction of the obstacle's
    bounding box, projects the leftmost and rightmost pixels of each
    scanned row to the ground plane, and takes the median width.

    Returns a dict with metric dimensions, or None if projection fails.
    """
    from skimage.measure import label, regionprops

    labeled, _ = label(obs_mask.astype(np.uint8), return_num=True)
    if labeled.max() == 0:
        return None

    R = _build_rotation(pitch_deg)
    R_inv = R.T
    K_inv = np.linalg.inv(K)

    results = []
    for reg in regionprops(labeled):
        min_row, min_col, max_row, max_col = reg.bbox
        height = max_row - min_row
        if height < 3:
            continue

        scan_rows = max(3, int(height * base_scan_ratio))
        scan_start = max_row - scan_rows

        widths_m = []
        centers_x = []
        for r in range(scan_start, max_row):
            cols = np.where(labeled[r] == reg.label)[0]
            if len(cols) < 2:
                continue
            pt_L = _ray_to_ground(float(cols[0]), float(r),
                                  K_inv, R_inv, camera_height_m)
            pt_R = _ray_to_ground(float(cols[-1]), float(r),
                                  K_inv, R_inv, camera_height_m)
            if pt_L is None or pt_R is None:
                continue
            w = math.sqrt((pt_R[0] - pt_L[0])**2 + (pt_R[1] - pt_L[1])**2)
            cx = (pt_L[0] + pt_R[0]) / 2.0
            widths_m.append(w)
            centers_x.append(cx)

        if not widths_m:
            continue

        results.append({
            "width_m": float(np.median(widths_m)),
            "width_cm": float(np.median(widths_m) * 100),
            "center_x_m": float(np.median(centers_x)),
            "bbox_rows": (int(min_row), int(max_row)),
            "bbox_cols": (int(min_col), int(max_col)),
        })

    return results if results else None


# ── Rectification (v1-style, improved) ───────────────────────────────────────

MAX_STRETCH = 10.0


def rectify_sidewalk(
    image_or_mask: np.ndarray,
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    valid: np.ndarray,
    target_width: int | None = None,
    is_mask: bool = False,
    padding_ratio: float = 0.3,
) -> tuple[np.ndarray, int, int] | None:
    """
    Per-row warp that normalises the sidewalk strip to a fixed width.

    Each row's [left_edge, right_edge] span is linearly mapped to
    a fixed-width canvas.  Vertical scaling uses vanishing-point-based
    perspective correction (rows closer to the vanishing point are
    stretched more to account for perspective foreshortening).

    Returns (rectified_image, target_width, padding) or None if there
    are fewer than 2 valid rows.
    """
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return None

    H_img = image_or_mask.shape[0] if image_or_mask.ndim < 3 else image_or_mask.shape[0]
    W_img = image_or_mask.shape[1] if image_or_mask.ndim < 3 else image_or_mask.shape[1]

    first_valid = valid_idx[0]
    last_valid = valid_idx[-1]

    # Fit edge lines for vanishing point and smooth interpolation
    a_L, b_L = np.polyfit(valid_idx.astype(float), left_edges[valid_idx], 1)
    a_R, b_R = np.polyfit(valid_idx.astype(float), right_edges[valid_idx], 1)

    # Vanishing row: where the two edge lines meet
    if abs(a_L - a_R) > 1e-6:
        vy = (b_R - b_L) / (a_L - a_R)
    else:
        vy = -1e8

    # Target width = median actual span
    if target_width is None:
        spans = right_edges[valid_idx] - left_edges[valid_idx]
        target_width = int(np.median(spans))
    target_width = max(target_width, 4)

    padding = int(target_width * padding_ratio)
    canvas_w = target_width + 2 * padding

    # Vertical scaling based on distance from vanishing point
    ref_dist = abs(last_valid - vy)
    if ref_dist < 1:
        ref_dist = 1.0

    row_scale = np.zeros(H_img, dtype=np.float64)
    for r in range(first_valid, last_valid + 1):
        dist = abs(r - vy)
        s = min((ref_dist / dist) ** 2, MAX_STRETCH) if dist > 0 else MAX_STRETCH
        row_scale[r] = s

    cum = np.cumsum(row_scale)
    total_out_height = int(cum[last_valid]) + 1
    if total_out_height < 2:
        return None

    # Build remap arrays
    map_x = np.zeros((total_out_height, canvas_w), dtype=np.float32)
    map_y = np.zeros((total_out_height, canvas_w), dtype=np.float32)
    out_cols = np.arange(canvas_w, dtype=np.float32)

    out_r = 0
    for src_r in range(first_valid, last_valid + 1):
        n_out = max(1, int(round(row_scale[src_r])))
        L = a_L * src_r + b_L
        R = a_R * src_r + b_R
        src_width = R - L
        if src_width < 1:
            src_width = 1.0

        scale = src_width / target_width
        for _ in range(n_out):
            if out_r >= total_out_height:
                break
            map_x[out_r, :] = L + (out_cols - padding) * scale
            map_y[out_r, :] = float(src_r)
            out_r += 1

    # Warp
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    inp = image_or_mask
    if is_mask:
        inp = image_or_mask.astype(np.uint8) * 255

    warped = cv2.remap(inp, map_x, map_y, interpolation=interp,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if is_mask:
        warped = warped > 127

    return warped, target_width, padding


# ── Rectified strip infill ───────────────────────────────────────────────────

def infill_rectified(
    rectified: np.ndarray,
    is_mask: bool = False,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Fill warp holes in a rectified strip.

    For masks: morphological closing fills small gaps.
    For images: uses OpenCV inpainting on zero-valued pixels within the
    bounding box of non-zero content.
    """
    if is_mask:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size),
        )
        filled = cv2.morphologyEx(
            rectified.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel,
        )
        return filled > 127

    if rectified.ndim == 3:
        gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    else:
        gray = rectified.copy()

    # Inpaint only zero pixels that are surrounded by content
    hole_mask = (gray == 0).astype(np.uint8)

    # Only fill within the bounding box of actual content
    content_rows = np.where(np.any(gray > 0, axis=1))[0]
    content_cols = np.where(np.any(gray > 0, axis=0))[0]
    if len(content_rows) == 0 or len(content_cols) == 0:
        return rectified

    r0, r1 = content_rows[0], content_rows[-1] + 1
    c0, c1 = content_cols[0], content_cols[-1] + 1

    # Zero out the hole mask outside the content bounding box
    outside = np.ones_like(hole_mask)
    outside[r0:r1, c0:c1] = 0
    hole_mask[outside.astype(bool)] = 0

    if hole_mask.sum() == 0:
        return rectified

    return cv2.inpaint(rectified, hole_mask, inpaintRadius=3,
                       flags=cv2.INPAINT_TELEA)
