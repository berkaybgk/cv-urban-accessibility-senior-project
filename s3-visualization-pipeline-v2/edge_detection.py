"""
Per-row sidewalk edge detection (v1-style).

Border-clipped rows are filled by linear extrapolation from fully valid rows.
"""

from __future__ import annotations

import numpy as np

BORDER_MARGIN = 3


def find_row_edges(
    mask: np.ndarray,
    border_margin: int = BORDER_MARGIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row left/right sidewalk edges with border extrapolation (polyfit)."""
    H, W = mask.shape
    left_edges = np.full(H, np.nan)
    right_edges = np.full(H, np.nan)
    valid = np.zeros(H, dtype=bool)
    extrapolated = np.zeros(H, dtype=bool)

    border_rows = np.zeros(H, dtype=bool)
    border_left_clipped = np.zeros(H, dtype=bool)
    border_right_clipped = np.zeros(H, dtype=bool)

    for r in range(H):
        cols = np.where(mask[r])[0]
        if len(cols) == 0:
            continue
        left_col, right_col = cols[0], cols[-1]
        left_at_border = left_col < border_margin
        right_at_border = right_col >= W - border_margin

        if not left_at_border and not right_at_border:
            left_edges[r] = left_col
            right_edges[r] = right_col
            valid[r] = True
        else:
            border_rows[r] = True
            border_left_clipped[r] = left_at_border
            border_right_clipped[r] = right_at_border
            left_edges[r] = left_col
            right_edges[r] = right_col

    valid_idx = np.where(valid)[0]
    if len(valid_idx) >= 2:
        a_L, b_L = np.polyfit(valid_idx.astype(float), left_edges[valid_idx], 1)
        a_R, b_R = np.polyfit(valid_idx.astype(float), right_edges[valid_idx], 1)

        for r in np.where(border_rows)[0]:
            if border_left_clipped[r]:
                left_edges[r] = a_L * r + b_L
            if border_right_clipped[r]:
                right_edges[r] = a_R * r + b_R
            valid[r] = True
            extrapolated[r] = True

    return left_edges, right_edges, valid, extrapolated
