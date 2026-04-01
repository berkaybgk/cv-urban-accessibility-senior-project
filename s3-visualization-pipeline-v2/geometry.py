"""
Sidewalk strip rectification (v1-style).

Per-row horizontal normalization to a fixed width with vanishing-point-based
vertical scaling.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_STRETCH = 6.0


def rectify_sidewalk(
    image_or_mask: np.ndarray,
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    valid_rows: np.ndarray,
    target_width: int | None = None,
    is_mask: bool = False,
) -> tuple[np.ndarray, int, int] | None:
    """Per-row warp with horizontal + vertical perspective correction."""
    H, W = image_or_mask.shape[:2]

    valid_widths = right_edges[valid_rows] - left_edges[valid_rows]
    if target_width is None:
        target_width = int(np.median(valid_widths))

    valid_idx = np.where(valid_rows)[0]
    all_rows = np.arange(H)

    left_interp = np.full(H, np.nan)
    right_interp = np.full(H, np.nan)

    if len(valid_idx) < 2:
        left_interp[:] = left_edges[valid_idx[0]] if len(valid_idx) else 0
        right_interp[:] = right_edges[valid_idx[0]] if len(valid_idx) else W - 1
        vy = -1e8
        first_valid, last_valid = 0, H - 1
    else:
        valid_r = valid_idx.astype(float)
        a_L, b_L = np.polyfit(valid_r, left_edges[valid_idx], 1)
        a_R, b_R = np.polyfit(valid_r, right_edges[valid_idx], 1)
        left_interp = a_L * all_rows + b_L
        right_interp = a_R * all_rows + b_R
        vy = (b_R - b_L) / (a_L - a_R) if abs(a_L - a_R) > 1e-6 else -1e8
        first_valid, last_valid = valid_idx[0], valid_idx[-1]

    row_scale = np.ones(H, dtype=np.float64)
    ref_dist = abs(last_valid - vy)
    for r in range(first_valid, last_valid + 1):
        dist = abs(r - vy)
        s = min((ref_dist / dist) ** 2, MAX_STRETCH) if dist > 0 else MAX_STRETCH
        row_scale[r] = s

    cum_real = np.cumsum(row_scale)
    cum_real -= cum_real[0]
    out_height = int(np.ceil(cum_real[-1])) + 1
    out_rows = np.arange(out_height, dtype=np.float32)
    src_row_for_out = np.interp(out_rows, cum_real, np.arange(H, dtype=np.float32))

    padding = int(target_width * 0.3)
    out_width = target_width + 2 * padding

    map_x = np.zeros((out_height, out_width), dtype=np.float32)
    map_y = np.zeros((out_height, out_width), dtype=np.float32)

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

        out_cols = np.arange(out_width, dtype=np.float32)
        scale = src_width / target_width
        map_x[out_r, :] = L + (out_cols - padding) * scale
        map_y[out_r, :] = src_r

    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    inp = image_or_mask.astype(np.uint8) if is_mask else image_or_mask
    warped = cv2.remap(inp, map_x, map_y, interpolation=flags,
                       borderMode=cv2.BORDER_CONSTANT)
    if is_mask:
        warped = warped.astype(bool)

    return warped, target_width, padding
