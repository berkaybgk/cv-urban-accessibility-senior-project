"""
Obstacle assignment and footprint estimation in rectified space (v1-style).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import label, regionprops


def assign_obstacles_to_sidewalks(
    sidewalk_masks: dict[str, np.ndarray],
    obstacle_masks: dict[str, np.ndarray],
) -> dict[str, list[str]]:
    """Assign each obstacle to the nearest sidewalk by horizontal proximity."""
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
    """Estimate ground-level footprint from an obstacle mask in rectified space."""
    labeled, _ = label(rect_mask.astype(np.uint8), return_num=True)
    footprint = np.zeros_like(rect_mask, dtype=bool)

    for reg in regionprops(labeled):
        min_row, min_col, max_row, max_col = reg.bbox
        height = max_row - min_row
        if height == 0:
            continue

        scan_ratio = trunk_scan_ratio if is_tree else base_scan_ratio
        scan_rows = max(1 if is_tree else 3, int(height * scan_ratio))
        scan_start = max_row - scan_rows

        row_widths: list[int] = []
        row_centers: list[float] = []
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


def build_footprint_metadata(
    sw_rect: np.ndarray,
    obstacle_masks_full_rect: dict[str, np.ndarray],
    obstacle_masks_rect: dict[str, np.ndarray],
    obstacle_is_tree: set[str],
    target_w: int,
    pad: int,
    sw_key: str,
) -> dict[str, Any]:
    """Per-obstacle footprint dimensions + image-level stats (v1-compatible)."""
    obstacles_meta: list[dict[str, Any]] = []
    for ot, fp in obstacle_masks_rect.items():
        full = obstacle_masks_full_rect.get(ot, fp)
        is_tree = any(t in ot for t in obstacle_is_tree)
        labeled_fp, _ = label(fp.astype(np.uint8), return_num=True)
        for reg in regionprops(labeled_fp):
            min_r, min_c, max_r, max_c = reg.bbox
            fp_w, fp_h = max_c - min_c, max_r - min_r
            obstacles_meta.append({
                "type": ot,
                "method": "trunk" if is_tree else "base",
                "bbox_xyxy": [int(min_c), int(min_r), int(max_c), int(max_r)],
                "footprint_width_px": int(fp_w),
                "footprint_height_px": int(fp_h),
                "footprint_area_px": int(reg.area),
                "full_silhouette_area_px": int(full.sum()),
                "reduction_pct": round(
                    (1 - fp.sum() / max(full.sum(), 1)) * 100, 1),
            })

    return {
        "segment": sw_key,
        "rectified_height": int(sw_rect.shape[0]),
        "rectified_width": int(sw_rect.shape[1]),
        "sidewalk_target_width_px": int(target_w),
        "padding_px": int(pad),
        "sidewalk_area_px": int(sw_rect.sum()),
        "obstacles": obstacles_meta,
    }


def build_width_metadata(
    sw_key: str,
    med: float,
    mean: float,
    std: float,
    mn: float,
    mx: float,
    rows_used: int,
    n_before_horizon: int,
    n_dropped_horizon: int,
    n_dropped_extrap: int,
    n_outliers: int,
    camera_cfg: dict[str, Any],
    width_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Width summary JSON (v1-compatible)."""
    return {
        "segment": sw_key,
        "width_cm": {
            "median": round(med, 1),
            "mean": round(mean, 1),
            "std": round(std, 1),
            "min": round(mn, 1),
            "max": round(mx, 1),
        },
        "rows_used": rows_used,
        "bias_reduction": {
            "rows_below_horizon": n_before_horizon,
            "dropped_near_horizon": n_dropped_horizon,
            "dropped_extrapolated": n_dropped_extrap,
            "dropped_iqr_outliers": n_outliers,
        },
        "camera": camera_cfg,
        "width_estimation_params": width_cfg,
    }
