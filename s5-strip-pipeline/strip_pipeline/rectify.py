"""Rectification utilities. Ported from cell 6 of the notebook.

Added: an optional ``edge_override`` on :func:`rectify_side_datadriven_fan` so a
caller can supply hand-edited left/right boundary lines (in the rotated frame
that :func:`find_row_edges` operates on). The geometry path is overridden in
:mod:`tiles` by constructing an ``edge_model`` dict and passing it to
:func:`rectify_sidewalk`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from .config import (
    BORDER_MARGIN,
    DEPTH_RATIO,
    HFOV_DEG,
    RANSAC_MIN_SAMPLES,
    RANSAC_RESIDUAL_THRESHOLD,
    RECTIFY_INTERPOLATION,
    ROAD_CONFIRMED_WEIGHT,
    ROAD_TOUCH_MARGIN,
    SHIFT_PERCENTILE,
)


def find_row_edges(mask: np.ndarray, road_mask: np.ndarray | None = None,
                   border_margin: int = BORDER_MARGIN,
                   ransac_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
                   ransac_min_samples: float = RANSAC_MIN_SAMPLES,
                   road_touch_margin: int = ROAD_TOUCH_MARGIN):
    sw_cols = np.where(mask.any(axis=0))[0]
    if len(sw_cols) == 0:
        H = mask.shape[0]
        return np.full(H, np.nan), np.full(H, np.nan), np.zeros(H, dtype=bool), np.zeros(H, dtype=bool), None

    sw_center = np.mean(sw_cols)
    if road_mask is not None and np.any(road_mask):
        road_center = np.mean(np.where(road_mask.any(axis=0))[0])
        is_left_sidewalk = sw_center < road_center
    else:
        is_left_sidewalk = None

    H, W = mask.shape
    left_edges = np.full(H, np.nan)
    right_edges = np.full(H, np.nan)
    valid = np.zeros(H, dtype=bool)
    extrapolated = np.zeros(H, dtype=bool)

    left_valid = np.zeros(H, dtype=bool)
    right_valid = np.zeros(H, dtype=bool)
    left_clipped = np.zeros(H, dtype=bool)
    right_clipped = np.zeros(H, dtype=bool)
    has_sidewalk = np.zeros(H, dtype=bool)
    road_confirmed_L = np.zeros(H, dtype=bool)
    road_confirmed_R = np.zeros(H, dtype=bool)

    for r in range(H):
        cols = np.where(mask[r])[0]
        if len(cols) == 0:
            continue
        has_sidewalk[r] = True
        left_col, right_col = cols[0], cols[-1]
        left_edges[r] = left_col
        right_edges[r] = right_col

        left_at_border = left_col < border_margin
        right_at_border = right_col >= W - border_margin
        left_valid[r] = not left_at_border
        right_valid[r] = not right_at_border
        left_clipped[r] = left_at_border
        right_clipped[r] = right_at_border

        if road_mask is not None and is_left_sidewalk is not None:
            road_cols = np.where(road_mask[r])[0]
            if len(road_cols) > 0:
                if is_left_sidewalk:
                    dist_right = np.min(np.abs(road_cols - right_col))
                    if dist_right <= road_touch_margin:
                        road_confirmed_R[r] = True
                        right_valid[r] = True
                        right_clipped[r] = False
                    else:
                        right_valid[r] = False
                else:
                    dist_left = np.min(np.abs(road_cols - left_col))
                    if dist_left <= road_touch_margin:
                        road_confirmed_L[r] = True
                        left_valid[r] = True
                        left_clipped[r] = False
                    else:
                        left_valid[r] = False

    both_valid = left_valid & right_valid
    left_valid_idx = np.where(left_valid)[0]
    right_valid_idx = np.where(right_valid)[0]
    edge_model = None

    if len(left_valid_idx) >= 2 and len(right_valid_idx) >= 2:
        a_L, b_L, inlier_mask_L = _fit_edge(left_valid_idx, left_edges, road_confirmed_L, side="left",
                                            ransac_threshold=ransac_threshold,
                                            ransac_min_samples=ransac_min_samples)
        a_R, b_R, inlier_mask_R = _fit_edge(right_valid_idx, right_edges, road_confirmed_R, side="right",
                                            ransac_threshold=ransac_threshold,
                                            ransac_min_samples=ransac_min_samples)
        edge_model = {
            "a_L": a_L, "b_L": b_L,
            "a_R": a_R, "b_R": b_R,
            "inlier_mask_L": inlier_mask_L,
            "inlier_mask_R": inlier_mask_R,
            "left_valid_idx": left_valid_idx,
            "right_valid_idx": right_valid_idx,
            "valid_idx": np.where(both_valid)[0],
            "road_confirmed_L": road_confirmed_L,
            "road_confirmed_R": road_confirmed_R,
        }
        for r in np.where(has_sidewalk)[0]:
            if left_clipped[r]:
                left_edges[r] = a_L * r + b_L
            if right_clipped[r]:
                right_edges[r] = a_R * r + b_R
            if left_clipped[r] or right_clipped[r]:
                extrapolated[r] = True
        valid = has_sidewalk.copy()
    else:
        valid = both_valid.copy()

    return left_edges, right_edges, valid, extrapolated, edge_model


def _fit_edge(valid_idx, edges, road_confirmed, side="left",
              ransac_threshold=RANSAC_RESIDUAL_THRESHOLD,
              ransac_min_samples=RANSAC_MIN_SAMPLES):
    rc_mask = road_confirmed[valid_idx]
    n_rc = rc_mask.sum()
    rc_rows = valid_idx[rc_mask]
    is_left = side == "left"

    if n_rc >= 4:
        a, b = np.polyfit(rc_rows.astype(float), edges[rc_rows], 1)
        fitted_rc = a * rc_rows.astype(float) + b
        residuals_rc = edges[rc_rows] - fitted_rc
        b += np.percentile(residuals_rc, SHIFT_PERCENTILE if is_left else 100 - SHIFT_PERCENTILE)
        fitted_all = a * valid_idx.astype(float) + b
        residuals_all = edges[valid_idx] - fitted_all
        inlier_mask = (residuals_all < ransac_threshold) if is_left else (residuals_all > -ransac_threshold)
        inlier_mask[rc_mask] = True
        if inlier_mask.sum() >= 2:
            clean_idx = valid_idx[inlier_mask]
            weights = np.ones(len(clean_idx), dtype=float)
            weights[[road_confirmed[r] for r in clean_idx]] = ROAD_CONFIRMED_WEIGHT
            a, b = np.polyfit(clean_idx.astype(float), edges[clean_idx], 1, w=weights)
        return a, b, inlier_mask

    if len(valid_idx) >= 4:
        X = valid_idx.reshape(-1, 1).astype(float)
        sample_weights = np.ones(len(valid_idx), dtype=float)
        if n_rc > 0:
            sample_weights[rc_mask] = ROAD_CONFIRMED_WEIGHT
        ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=ransac_threshold,
                                 min_samples=ransac_min_samples, random_state=42)
        ransac.fit(X, edges[valid_idx], sample_weight=sample_weights)
        a = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        fitted = a * valid_idx.astype(float) + b
        residuals = edges[valid_idx] - fitted
        inlier_mask = (residuals < ransac_threshold) if is_left else (residuals > -ransac_threshold)
        if n_rc > 0:
            inlier_mask[rc_mask] = True
        if inlier_mask.sum() >= 2:
            clean_idx = valid_idx[inlier_mask]
            weights = np.ones(len(clean_idx), dtype=float)
            weights[[road_confirmed[r] for r in clean_idx]] = ROAD_CONFIRMED_WEIGHT
            a, b = np.polyfit(clean_idx.astype(float), edges[clean_idx], 1, w=weights)
            fitted2 = a * clean_idx.astype(float) + b
            residuals_clean = edges[clean_idx] - fitted2
            b += np.percentile(residuals_clean, SHIFT_PERCENTILE if is_left else 100 - SHIFT_PERCENTILE)
            fitted3 = a * valid_idx.astype(float) + b
            residuals3 = edges[valid_idx] - fitted3
            inlier_mask = (residuals3 < ransac_threshold) if is_left else (residuals3 > -ransac_threshold)
            if n_rc > 0:
                inlier_mask[rc_mask] = True
    else:
        a, b = np.polyfit(valid_idx.astype(float), edges[valid_idx], 1)
        inlier_mask = np.ones(len(valid_idx), dtype=bool)
    return a, b, inlier_mask


def _compute_rectify_params(img: np.ndarray, model: dict[str, Any] | None, original_width: int | None = None):
    if original_width is None:
        original_width = img.shape[1]
    f_px = original_width / (2.0 * np.tan(np.radians(HFOV_DEG / 2.0)))
    if model is not None:
        denom = model["a_L"] - model["a_R"]
        vp_y = (model["b_R"] - model["b_L"]) / denom if abs(denom) > 1e-6 else 0
        vp_x = model["a_L"] * vp_y + model["b_L"]
        cos_corr = np.cos(max(0.0, np.arctan((vp_x - img.shape[1] / 2.0) / f_px)))
    else:
        cos_corr = 1.0
    return f_px, cos_corr


def rectify_sidewalk(image_or_mask, left_edges, right_edges, valid_rows,
                     edge_model=None, target_width=None, is_mask=False,
                     cos_correction=1.0, f_px=None):
    H, W = image_or_mask.shape[:2]
    valid_widths = right_edges[valid_rows] - left_edges[valid_rows]
    if len(valid_widths) == 0:
        return image_or_mask, target_width or 100, 0
    if target_width is None:
        target_width = int(np.median(valid_widths) * cos_correction)

    valid_idx = np.where(valid_rows)[0]
    all_rows = np.arange(H)
    if len(valid_idx) < 2:
        left_interp = np.full(H, left_edges[valid_idx[0]] if len(valid_idx) else 0)
        right_interp = np.full(H, right_edges[valid_idx[0]] if len(valid_idx) else W - 1)
        vy = -1e8
        first_valid = 0
        last_valid = H - 1
        a_L, b_L = 0.0, left_interp[0]
        a_R, b_R = 0.0, right_interp[0]
    else:
        if edge_model is not None:
            a_L, b_L = edge_model["a_L"], edge_model["b_L"]
            a_R, b_R = edge_model["a_R"], edge_model["b_R"]
        else:
            valid_r = valid_idx.astype(float)
            a_L, b_L = np.polyfit(valid_r, left_edges[valid_idx], 1)
            a_R, b_R = np.polyfit(valid_r, right_edges[valid_idx], 1)
        left_interp = a_L * all_rows + b_L
        right_interp = a_R * all_rows + b_R
        vy = (b_R - b_L) / (a_L - a_R) if abs(a_L - a_R) > 1e-6 else -1e8
        first_valid = valid_idx[0]
        last_valid = valid_idx[-1]

    use_perp = False
    vp_perp_x = 0.0
    cy = H / 2.0
    if f_px is not None and abs(a_L - a_R) > 1e-6:
        vp_x = a_L * vy + b_L
        vp_x_offset = vp_x - W / 2.0
        if abs(vp_x_offset) > 1e-6:
            a_avg = (a_L + a_R) / 2.0
            sign = 1.0 if a_avg > 0 else -1.0
            vp_perp_x = W / 2.0 + sign * f_px * f_px / abs(vp_x_offset)
            use_perp = True

    row_scale = np.ones(H, dtype=np.float64)
    ref_dist = abs(last_valid - vy)
    max_stretch = 50.0
    for r in range(first_valid, last_valid + 1):
        dist = abs(r - vy)
        row_scale[r] = min((ref_dist / dist) ** 2 if dist > 0 else max_stretch, max_stretch)

    cum_real = np.cumsum(row_scale)
    cum_real = cum_real - cum_real[0]
    out_height = int(np.ceil(cum_real[-1])) + 1
    out_rows = np.arange(out_height, dtype=np.float32)
    src_row_for_out = np.interp(out_rows, cum_real, np.arange(H, dtype=np.float32))

    padding = 0
    out_width = target_width
    map_x = np.zeros((out_height, out_width), dtype=np.float32)
    map_y = np.zeros((out_height, out_width), dtype=np.float32)
    out_cols = np.arange(out_width, dtype=np.float32)

    for out_r in range(out_height):
        src_r = src_row_for_out[out_r]
        src_r_int = int(src_r)
        L = left_interp[min(src_r_int, H - 1)]
        R = right_interp[min(src_r_int, H - 1)]
        src_r_frac = src_r - src_r_int
        if src_r_frac > 0 and src_r_int + 1 < H:
            L = L * (1 - src_r_frac) + left_interp[src_r_int + 1] * src_r_frac
            R = R * (1 - src_r_frac) + right_interp[src_r_int + 1] * src_r_frac

        src_width = R - L
        if src_width <= 0:
            map_x[out_r, :] = -1
            map_y[out_r, :] = src_r
            continue

        cx = (L + R) / 2.0
        if use_perp and src_r > cy + 1:
            dx_perp = vp_perp_x - cx
            if abs(dx_perp) > 1e-6:
                perp_slope = (cy - src_r) / dx_perp
                denom_L = 1.0 - a_L * perp_slope
                denom_R = 1.0 - a_R * perp_slope
                if abs(denom_L) > 1e-9 and abs(denom_R) > 1e-9:
                    t_L = (L - cx) / denom_L
                    t_R = (R - cx) / denom_R
                    perp_span = t_R - t_L
                    if perp_span > 0:
                        t = t_L + (out_cols - padding) / target_width * perp_span
                        map_x[out_r, :] = cx + t
                        map_y[out_r, :] = src_r + t * perp_slope
                        continue

        scale = src_width / target_width
        map_x[out_r, :] = L + (out_cols - padding) * scale
        map_y[out_r, :] = src_r

    flags = cv2.INTER_NEAREST if is_mask else globals().get("RECTIFY_INTERPOLATION", cv2.INTER_CUBIC)
    inp = image_or_mask.astype(np.uint8) if is_mask else image_or_mask
    warped = cv2.remap(inp, map_x, map_y, interpolation=flags, borderMode=cv2.BORDER_CONSTANT)
    if is_mask:
        warped = warped.astype(bool)
    return warped, target_width, padding


def rotate_code_for_side(direction: str) -> int:
    """The cv2 rotation applied to a side-view image before edge fitting."""
    return cv2.ROTATE_90_COUNTERCLOCKWISE if direction == "left" else cv2.ROTATE_90_CLOCKWISE


def rectify_side_datadriven_fan(image, single_mask, direction: str, target_width=None, is_mask=False,
                                depth_ratio=DEPTH_RATIO, edge_override: dict[str, float] | None = None):
    if direction not in {"left", "right"}:
        raise ValueError(f"direction must be left or right, got {direction!r}")

    safe_image = image.astype(np.uint8) if image.dtype == bool else image
    rotate_code = rotate_code_for_side(direction)
    img_rot = cv2.rotate(safe_image, rotate_code)
    mask_rot = cv2.rotate(single_mask.astype(np.uint8), rotate_code).astype(bool)
    H_rot, W_rot = img_rot.shape[:2]

    left_edges, right_edges, valid, extrap, edge_model = find_row_edges(mask_rot)
    valid_widths = right_edges[valid] - left_edges[valid]
    if len(valid_widths) == 0 and edge_override is None:
        return image, target_width or 100, 0
    if target_width is None:
        if len(valid_widths) > 0:
            target_width = int(np.median(valid_widths))
        else:
            target_width = int(abs((edge_override["a_R"] - edge_override["a_L"]) * (H_rot / 2.0)
                                   + edge_override["b_R"] - edge_override["b_L"])) or 100

    out_width = target_width
    out_height = H_rot
    padding = 0
    if edge_override is not None:
        a_L, b_L = edge_override["a_L"], edge_override["b_L"]
        a_R, b_R = edge_override["a_R"], edge_override["b_R"]
    elif edge_model is not None:
        a_L, b_L = edge_model["a_L"], edge_model["b_L"]
        a_R, b_R = edge_model["a_R"], edge_model["b_R"]
    else:
        valid_idx = np.where(valid)[0]
        a_L, b_L = np.polyfit(valid_idx, left_edges[valid_idx], 1)
        a_R, b_R = np.polyfit(valid_idx, right_edges[valid_idx], 1)

    out_x_grid, out_y_grid = np.meshgrid(np.arange(out_width), np.arange(out_height))
    x_norm = out_x_grid / max(out_width - 1, 1)

    depth_norm = x_norm if direction == "left" else (1 - x_norm)
    stretch_factor = 1.0 + (depth_ratio - 1.0) * depth_norm

    cy_rot = H_rot / 2.0
    src_y = cy_rot + (out_y_grid - cy_rot) * stretch_factor
    L = a_L * src_y + b_L
    R = a_R * src_y + b_R
    src_x = L + x_norm * (R - L)

    flags = cv2.INTER_NEAREST if is_mask else globals().get("RECTIFY_INTERPOLATION", cv2.INTER_CUBIC)
    inp = img_rot.astype(np.uint8) if is_mask else img_rot
    warped = cv2.remap(inp, src_x.astype(np.float32), src_y.astype(np.float32),
                       interpolation=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if is_mask:
        warped = warped.astype(bool)
    return warped, target_width, padding
