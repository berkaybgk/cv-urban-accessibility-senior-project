"""RANSAC-based boundary fitting and physical width estimation.

Ported and adapted from ransac_width_estimation.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Camera geometry constants
FOV_DEG = 90.0       # vertical field-of-view in degrees
PITCH_DEG = -38.0     # camera downward pitch in degrees (looking down)
CAM_HEIGHT_M = 2.5   # camera height above ground in metres

# RANSAC boundary fitting parameters
RANSAC_RESIDUAL_THRESHOLD = 2.0   # pixels; inlier tolerance for boundary lines
RANSAC_MIN_SAMPLES = 0.25         # fraction of valid rows used as min_samples
BORDER_MARGIN = 3                 # pixels; columns/rows within this of the edge
SHIFT_PERCENTILE_UPPER = 10       # shift RANSAC line toward the tightest upper bound
SHIFT_PERCENTILE_LOWER = 90       # shift RANSAC line toward the tightest lower bound
MIN_MASK_AREA = 500               # pixels; skip masks smaller than this
MIN_ROW_COVERAGE = 0.05           # fraction of rows that must contain sidewalk


@dataclass
class BoundaryResult:
    """Horizontal RANSAC boundary lines for a sidewalk-facing image."""
    # Line parameters:  y_boundary(col) = a * col + b
    a_upper: float = float("nan")
    b_upper: float = float("nan")
    a_lower: float = float("nan")
    b_lower: float = float("nan")

    # Per-column observations
    valid_cols: np.ndarray = field(default_factory=lambda: np.array([]))
    upper_edge_rows: np.ndarray = field(default_factory=lambda: np.array([]))
    lower_edge_rows: np.ndarray = field(default_factory=lambda: np.array([]))
    inlier_upper: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    inlier_lower: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Derived representative y values (median of inlier boundary rows,
    # evaluated at the horizontal image centre)
    y_upper_mid: float = float("nan")  # wall side  → y_wall in calculate_width
    y_lower_mid: float = float("nan")  # road side  → y_curb in calculate_width

    success: bool = False
    reason: str = ""


def _fit_boundary_line(
    valid_cols: np.ndarray,
    edge_rows: np.ndarray,
    side: str,  # "upper" or "lower"
    ransac_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
    ransac_min_samples: float = RANSAC_MIN_SAMPLES,
) -> tuple[float, float, np.ndarray]:
    """Fit a line row = a * col + b to (col, row) boundary samples using RANSAC."""
    is_upper = side == "upper"
    shift_pct = SHIFT_PERCENTILE_UPPER if is_upper else SHIFT_PERCENTILE_LOWER

    n = len(valid_cols)
    if n < 2:
        # Degenerate: return a horizontal line at the mean edge row.
        a = 0.0
        b = float(np.mean(edge_rows))
        return a, b, np.ones(n, dtype=bool)

    if n >= 4:
        X = valid_cols.reshape(-1, 1).astype(float)
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=ransac_threshold,
            min_samples=max(2, int(ransac_min_samples * n)),
            random_state=42,
        )
        ransac.fit(X, edge_rows.astype(float))
        a = float(ransac.estimator_.coef_[0])
        b = float(ransac.estimator_.intercept_)

        fitted = a * valid_cols.astype(float) + b
        residuals = edge_rows.astype(float) - fitted
        # Inlier selection: for upper, keep rows that are ≥ fitted (below);
        # for lower, keep rows that are ≤ fitted (above).
        if is_upper:
            inlier_mask = residuals > -ransac_threshold
        else:
            inlier_mask = residuals < ransac_threshold

        # Refit on clean inliers with a percentile shift to tighten the boundary.
        if inlier_mask.sum() >= 2:
            clean_cols = valid_cols[inlier_mask]
            clean_rows = edge_rows[inlier_mask]
            a, b = np.polyfit(clean_cols.astype(float), clean_rows.astype(float), 1)
            fitted_clean = a * clean_cols.astype(float) + b
            residuals_clean = clean_rows.astype(float) - fitted_clean
            b += np.percentile(residuals_clean, shift_pct)

            # Final inlier mask relative to the shifted line.
            fitted_all = a * valid_cols.astype(float) + b
            residuals_all = edge_rows.astype(float) - fitted_all
            if is_upper:
                inlier_mask = residuals_all > -ransac_threshold
            else:
                inlier_mask = residuals_all < ransac_threshold
    else:
        a, b = np.polyfit(valid_cols.astype(float), edge_rows.astype(float), 1)
        inlier_mask = np.ones(n, dtype=bool)

    return a, b, inlier_mask


def find_horizontal_boundaries(
    mask: np.ndarray,
    border_margin: int = BORDER_MARGIN,
    ransac_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
    ransac_min_samples: float = RANSAC_MIN_SAMPLES,
    min_mask_area: int = MIN_MASK_AREA,
    min_row_coverage: float = MIN_ROW_COVERAGE,
) -> BoundaryResult:
    """Detect upper (wall side) and lower (road side) pixel boundaries of a sidewalk mask."""
    H, W = mask.shape
    result = BoundaryResult()

    # Quality checks
    mask_area = int(mask.sum())
    if mask_area < min_mask_area:
        result.reason = f"mask_area_too_small ({mask_area} < {min_mask_area})"
        return result

    row_coverage = float(np.mean(mask.any(axis=1)))
    if row_coverage < min_row_coverage:
        result.reason = f"row_coverage_too_low ({row_coverage:.3f} < {min_row_coverage})"
        return result

    # Per-column edge extraction
    valid_cols = []
    upper_edge_rows = []   # smallest row index (top of mask)
    lower_edge_rows = []   # largest row index (bottom of mask)
    upper_clipped = []     # True if upper edge touches image top border
    lower_clipped = []     # True if lower edge touches image bottom border

    for col in range(W):
        col_mask = mask[:, col]
        rows_with_mask = np.where(col_mask)[0]
        if len(rows_with_mask) == 0:
            continue
        r_top = int(rows_with_mask[0])
        r_bot = int(rows_with_mask[-1])
        valid_cols.append(col)
        upper_edge_rows.append(r_top)
        lower_edge_rows.append(r_bot)
        upper_clipped.append(r_top < border_margin)
        lower_clipped.append(r_bot >= H - border_margin)

    if len(valid_cols) < 4:
        result.reason = f"too_few_valid_columns ({len(valid_cols)})"
        return result

    valid_cols = np.array(valid_cols, dtype=np.float64)
    upper_edge_rows = np.array(upper_edge_rows, dtype=np.float64)
    lower_edge_rows = np.array(lower_edge_rows, dtype=np.float64)
    upper_clipped = np.array(upper_clipped, dtype=bool)
    lower_clipped = np.array(lower_clipped, dtype=bool)

    # Filter: only use non-clipped columns for initial fitting
    upper_fit_cols = valid_cols[~upper_clipped]
    upper_fit_rows = upper_edge_rows[~upper_clipped]
    lower_fit_cols = valid_cols[~lower_clipped]
    lower_fit_rows = lower_edge_rows[~lower_clipped]

    # Fall back to all columns if too many are clipped.
    if len(upper_fit_cols) < 4:
        upper_fit_cols = valid_cols
        upper_fit_rows = upper_edge_rows
    if len(lower_fit_cols) < 4:
        lower_fit_cols = valid_cols
        lower_fit_rows = lower_edge_rows

    # RANSAC fitting
    a_U, b_U, inlier_U = _fit_boundary_line(
        upper_fit_cols, upper_fit_rows, "upper", ransac_threshold, ransac_min_samples
    )
    a_L, b_L, inlier_L = _fit_boundary_line(
        lower_fit_cols, lower_fit_rows, "lower", ransac_threshold, ransac_min_samples
    )

    # Extrapolate clipped edges using the fitted lines
    upper_edge_rows_final = upper_edge_rows.copy()
    lower_edge_rows_final = lower_edge_rows.copy()
    for i, col in enumerate(valid_cols):
        if upper_clipped[i]:
            upper_edge_rows_final[i] = a_U * col + b_U
        if lower_clipped[i]:
            lower_edge_rows_final[i] = a_L * col + b_L

    # Representative y values at horizontal image centre
    col_mid = float(W / 2.0)
    y_upper_mid = a_U * col_mid + b_U
    y_lower_mid = a_L * col_mid + b_L

    # Map inlier masks back to the full valid_cols array for storage.
    inlier_upper_full = np.zeros(len(valid_cols), dtype=bool)
    inlier_lower_full = np.zeros(len(valid_cols), dtype=bool)
    upper_fit_indices = np.where(~upper_clipped)[0] if (~upper_clipped).any() else np.arange(len(valid_cols))
    if len(upper_fit_indices) == len(inlier_U):
        inlier_upper_full[upper_fit_indices] = inlier_U
    else:
        inlier_upper_full[:] = True  # fallback
    lower_fit_indices = np.where(~lower_clipped)[0] if (~lower_clipped).any() else np.arange(len(valid_cols))
    if len(lower_fit_indices) == len(inlier_L):
        inlier_lower_full[lower_fit_indices] = inlier_L
    else:
        inlier_lower_full[:] = True  # fallback

    result.a_upper = a_U
    result.b_upper = b_U
    result.a_lower = a_L
    result.b_lower = b_L
    result.valid_cols = valid_cols
    result.upper_edge_rows = upper_edge_rows_final
    result.lower_edge_rows = lower_edge_rows_final
    result.inlier_upper = inlier_upper_full
    result.inlier_lower = inlier_lower_full
    result.y_upper_mid = y_upper_mid
    result.y_lower_mid = y_lower_mid
    result.success = True
    result.reason = "ok"
    return result


def calculate_width(
    y_wall: float,
    y_curb: float,
    image_height: int,
    fov_deg: float = FOV_DEG,
    pitch_deg: float = PITCH_DEG,
    cam_height: float = CAM_HEIGHT_M,
) -> tuple[float, float, float]:
    """Calculate physical sidewalk width from pixel boundary positions."""
    fov_rad = math.radians(fov_deg)
    pitch_rad = math.radians(pitch_deg)

    f_y = image_height / (2 * math.tan(fov_rad / 2))
    c_y = image_height / 2.0

    def get_gamma(y: float) -> float:
        return math.atan((y - c_y) / f_y)

    def get_z(y: float) -> float:
        gamma_y = get_gamma(y)
        downward_pitch = abs(pitch_rad)
        angle = downward_pitch + gamma_y
        if angle <= 0:
            return float("inf")
        return cam_height / math.tan(angle)

    z_wall = get_z(y_wall)
    z_curb = get_z(y_curb)
    width = z_wall - z_curb
    return width, z_wall, z_curb
