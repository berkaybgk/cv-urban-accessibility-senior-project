"""
Obstacle analysis in rectified (perspective-normalised) space.

After rectify_sidewalk normalises the sidewalk strip to a fixed canvas
width, obstacles warped through the same mapping have correct relative
positions.  Footprint estimation scans the bottom portion of each
connected component to approximate the ground-level width.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops


def assign_obstacles_to_sidewalks(
    sidewalk_masks: dict[str, np.ndarray],
    obstacle_masks: dict[str, np.ndarray],
) -> dict[str, list[str]]:
    """
    Assign each obstacle to the nearest sidewalk by horizontal proximity
    in the original perspective image.

    Uses column-center distance between the obstacle and sidewalk spans.
    """
    if not sidewalk_masks or not obstacle_masks:
        return {k: [] for k in sidewalk_masks}

    sw_spans: dict[str, tuple[float, float]] = {}
    for sw_key, sw_mask in sidewalk_masks.items():
        cols = np.where(np.any(sw_mask, axis=0))[0]
        if len(cols) == 0:
            sw_spans[sw_key] = (0, 0)
        else:
            sw_spans[sw_key] = (cols[0], cols[-1])

    assignment: dict[str, list[str]] = {k: [] for k in sidewalk_masks}

    for obs_key, obs_mask in obstacle_masks.items():
        cols = np.where(np.any(obs_mask, axis=0))[0]
        if len(cols) == 0:
            continue
        obs_center = (cols[0] + cols[-1]) / 2.0

        best_sw = None
        best_dist = float("inf")
        for sw_key, (sw_left, sw_right) in sw_spans.items():
            if sw_left <= obs_center <= sw_right:
                dist = 0.0
            else:
                dist = min(abs(obs_center - sw_left), abs(obs_center - sw_right))
            if dist < best_dist:
                best_dist = dist
                best_sw = sw_key

        if best_sw is not None:
            assignment[best_sw].append(obs_key)

    return assignment


def estimate_width_footprint(
    rect_mask: np.ndarray,
    is_tree: bool = False,
    base_scan_ratio: float = 0.15,
    trunk_scan_ratio: float = 0.40,
    aspect_ratio: float = 1.0,
    max_height: int = 25,
) -> np.ndarray:
    """
    Estimate ground-level footprint from an obstacle mask in rectified space.

    For each connected component, scans the bottom portion of the blob
    (or trunk region for trees) to estimate the ground-level width.
    Returns a boolean footprint mask of the same shape.
    """
    labeled, _ = label(rect_mask.astype(np.uint8), return_num=True)
    footprint = np.zeros_like(rect_mask)

    for reg in regionprops(labeled):
        min_row, min_col, max_row, max_col = reg.bbox
        height = max_row - min_row
        if height == 0:
            continue

        scan_ratio = trunk_scan_ratio if is_tree else base_scan_ratio
        scan_rows = max(1 if is_tree else 3, int(height * scan_ratio))
        scan_start = max_row - scan_rows

        row_widths = []
        row_centers = []
        for r in range(scan_start, max_row):
            cols = np.where(labeled[r] == reg.label)[0]
            if len(cols) == 0:
                continue
            row_widths.append(cols[-1] - cols[0] + 1)
            row_centers.append((cols[0] + cols[-1]) / 2.0)

        if not row_widths:
            continue

        fp_width = max(int(np.median(row_widths)), 1)
        median_center = np.median(row_centers)
        fp_height = min(max(1, int(fp_width * aspect_ratio)), height, max_height)

        fp_left = max(0, int(median_center - fp_width / 2))
        fp_right = min(rect_mask.shape[1], fp_left + fp_width)
        fp_top = max(min_row, max_row - fp_height)

        footprint[fp_top:max_row, fp_left:fp_right] = True

    return footprint


def compute_obstacle_footprint_metadata(
    footprint: np.ndarray,
    obs_key: str,
    target_width: int,
    padding: int,
    raycast_footprints: list[dict] | None = None,
) -> list[dict]:
    """
    Build metadata dicts for each connected component in a footprint mask.

    Positions are reported relative to the sidewalk strip (subtracting
    padding) so that lateral position 0 = left edge, target_width = right edge.

    If ``raycast_footprints`` is provided (from geometry.compute_obstacle_
    ground_footprint), the metric width_cm and center_x_m are included.
    """
    labeled, _ = label(footprint.astype(np.uint8), return_num=True)
    results = []

    for reg in regionprops(labeled):
        min_row, min_col, max_row, max_col = reg.bbox
        entry = {
            "obs_key": obs_key,
            "label": int(reg.label),
            "bbox_rows": (int(min_row), int(max_row)),
            "bbox_cols": (int(min_col), int(max_col)),
            "width_px": int(max_col - min_col),
            "height_px": int(max_row - min_row),
            "center_col_rel": float((min_col + max_col) / 2.0 - padding),
            "area_px": int(reg.area),
            "fraction_of_sidewalk": float(
                (max_col - min_col) / target_width
            ) if target_width > 0 else 0.0,
        }

        # Attach ray-cast metric data if available
        if raycast_footprints:
            best_match = None
            best_overlap = 0
            for rc in raycast_footprints:
                rc_r0, rc_r1 = rc["bbox_rows"]
                overlap_r0 = max(min_row, rc_r0)
                overlap_r1 = min(max_row, rc_r1)
                overlap = max(0, overlap_r1 - overlap_r0)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = rc
            if best_match:
                entry["metric_width_cm"] = best_match["width_cm"]
                entry["metric_center_x_m"] = best_match["center_x_m"]

        results.append(entry)

    return results


def compute_usable_width_cm(
    widths_cm: np.ndarray,
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    valid: np.ndarray,
    obstacle_masks_rectified: dict[str, np.ndarray],
    target_width: int,
    padding: int,
) -> np.ndarray:
    """
    Compute effective usable sidewalk width, subtracting obstacle
    intrusions per row.

    Works in the rectified space: for each valid row, checks how much
    of the sidewalk strip [padding, padding+target_width] is blocked
    by obstacles, and subtracts it proportionally from the metric width.
    """
    H = len(widths_cm)
    usable = widths_cm.copy()

    if not obstacle_masks_rectified:
        return usable

    # Combine all obstacle masks
    sample_mask = next(iter(obstacle_masks_rectified.values()))
    rect_h, rect_w = sample_mask.shape
    combined = np.zeros((rect_h, rect_w), dtype=bool)
    for m in obstacle_masks_rectified.values():
        if m.shape == (rect_h, rect_w):
            combined |= m

    sw_left = padding
    sw_right = padding + target_width

    for out_r in range(min(rect_h, H)):
        if not valid[out_r] or np.isnan(widths_cm[out_r]):
            continue
        sw_cols = set(range(sw_left, sw_right))
        obs_cols = set(np.where(combined[out_r])[0].tolist())
        blocked = sw_cols & obs_cols
        if not blocked:
            continue
        blocked_frac = len(blocked) / max(1, target_width)
        usable[out_r] = widths_cm[out_r] * (1.0 - blocked_frac)

    return usable
