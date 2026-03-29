"""
Perspective-space sidewalk edge detection and mask processing.

Works directly in the original image coordinates:

1. Per-row left/right edge extraction from the binary mask.
2. Border-aware extrapolation using RANSAC-robust linear fitting:
   resistant to outlier rows from occlusions, noisy segmentation,
   and partial masks.  Extrapolation is bounded to a configurable
   number of rows beyond the last valid data to prevent runaway.
3. Morphological gap filling for segmentation noise.
"""

from __future__ import annotations

import cv2
import numpy as np


BORDER_MARGIN = 3
MAX_EXTRAPOLATION_ROWS = 60


def fill_small_gaps(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Fill small holes in a binary mask via morphological closing."""
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size),
    )
    closed = cv2.morphologyEx(
        mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel,
    )
    return closed > 127


def _ransac_line_fit(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = 100,
    inlier_thresh: float = 5.0,
) -> tuple[float, float]:
    """
    Fit y = a*x + b using RANSAC.

    Picks random 2-point subsets, fits a line, counts inliers within
    ``inlier_thresh`` pixels.  Returns (a, b) for the best model.
    Falls back to least-squares if fewer than 3 points.
    """
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n == 1 else 0.0
    if n < 4:
        a, b = np.polyfit(x.astype(float), y.astype(float), 1)
        return float(a), float(b)

    rng = np.random.RandomState(42)
    best_inliers = 0
    best_a, best_b = np.polyfit(x.astype(float), y.astype(float), 1)

    for _ in range(n_iter):
        idx = rng.choice(n, 2, replace=False)
        x1, x2 = float(x[idx[0]]), float(x[idx[1]])
        y1, y2 = float(y[idx[0]]), float(y[idx[1]])
        if abs(x2 - x1) < 1e-9:
            continue
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        residuals = np.abs(y - (a * x + b))
        inlier_count = int(np.sum(residuals < inlier_thresh))

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            inlier_mask = residuals < inlier_thresh
            best_a, best_b = np.polyfit(
                x[inlier_mask].astype(float),
                y[inlier_mask].astype(float), 1,
            )

    return float(best_a), float(best_b)


def find_row_edges(
    mask: np.ndarray,
    border_margin: int = BORDER_MARGIN,
    max_extrap_rows: int = MAX_EXTRAPOLATION_ROWS,
    ransac_inlier_thresh: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-row left/right sidewalk edges with robust border extrapolation.

    For each row in ``mask``, finds the leftmost and rightmost True
    column.  If either edge falls within ``border_margin`` of the
    image border, that row is flagged as border-clipped and later
    filled in by RANSAC-robust linear extrapolation from valid rows.

    Extrapolation is bounded: border rows more than
    ``max_extrap_rows`` rows away from the nearest valid row are left
    as NaN (not extrapolated).  This prevents the fit from making
    wild guesses far from the data.

    Returns
    -------
    left_edges : float array, left edge column per row (NaN if no mask).
    right_edges : float array, right edge column per row.
    valid : bool array, True if this row has usable edge data.
    extrapolated : bool array, True if this row's edge was filled
        by extrapolation.
    """
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

        left_col = cols[0]
        right_col = cols[-1]

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
            if not left_at_border:
                left_edges[r] = left_col
            if not right_at_border:
                right_edges[r] = right_col

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return left_edges, right_edges, valid, extrapolated

    # RANSAC-robust linear fit for each edge
    a_L, b_L = _ransac_line_fit(
        valid_idx, left_edges[valid_idx],
        inlier_thresh=ransac_inlier_thresh,
    )
    a_R, b_R = _ransac_line_fit(
        valid_idx, right_edges[valid_idx],
        inlier_thresh=ransac_inlier_thresh,
    )

    first_valid = valid_idx[0]
    last_valid = valid_idx[-1]

    for r in np.where(border_rows)[0]:
        # Bound extrapolation distance
        dist_to_valid = min(
            abs(r - first_valid) if r < first_valid else 0,
            abs(r - last_valid) if r > last_valid else 0,
        )
        if first_valid <= r <= last_valid:
            dist_to_valid = 0
        if dist_to_valid > max_extrap_rows:
            continue

        if border_left_clipped[r]:
            left_edges[r] = a_L * r + b_L
        if border_right_clipped[r]:
            right_edges[r] = a_R * r + b_R

        # Sanity: left must be < right, both within image bounds
        if (not np.isnan(left_edges[r]) and not np.isnan(right_edges[r])
                and left_edges[r] < right_edges[r]
                and left_edges[r] >= -W * 0.5
                and right_edges[r] <= W * 1.5):
            valid[r] = True
            extrapolated[r] = True
        else:
            left_edges[r] = np.nan
            right_edges[r] = np.nan

    return left_edges, right_edges, valid, extrapolated
