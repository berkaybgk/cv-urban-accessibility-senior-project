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


def generate_width_debug_plot(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    boundary: BoundaryResult,
    width: float,
    z_wall: float,
    z_curb: float,
    fov_deg: float = FOV_DEG,
    pitch_deg: float = PITCH_DEG,
    cam_height: float = CAM_HEIGHT_M,
    point_id: str = "?",
    side: str = "?",
) -> bytes:
    """Generate a 3-panel PNG debug image illustrating the sidewalk width estimation steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [1.3, 1.0, 1.1]})

    # Panel 1: Image & Boundaries Overlay
    ax_img = axes[0]
    ax_img.imshow(image_rgb)
    H, W = image_rgb.shape[:2]

    # Overlay sidewalk mask
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[mask] = [0, 128, 255] # Sky blue tint
    ax_img.imshow(mask_colored, alpha=0.35)

    if boundary.success:
        cols_line = np.arange(W)
        upper_line = boundary.a_upper * cols_line + boundary.b_upper
        lower_line = boundary.a_lower * cols_line + boundary.b_lower

        ax_img.plot(cols_line, upper_line, color="#00FF00", linewidth=2.5, label="Upper Boundary (Wall)")
        ax_img.plot(cols_line, lower_line, color="#FF3333", linewidth=2.5, label="Lower Boundary (Curb)")
        
        # Center column marker
        col_mid = W / 2.0
        ax_img.axvline(col_mid, color="yellow", linestyle="--", alpha=0.75, label="Center Column")
        
        # Intersection points
        ax_img.scatter(col_mid, boundary.y_upper_mid, color="#00FF00", edgecolors="white", s=80, zorder=5)
        ax_img.scatter(col_mid, boundary.y_lower_mid, color="#FF3333", edgecolors="white", s=80, zorder=5)

    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.set_title(f"Sidewalk Detection & Boundaries\nPoint: {point_id} | Side: {side}", fontsize=12, fontweight="bold")
    ax_img.legend(loc="upper right")
    ax_img.axis("off")

    # Calculate angles for Panel 3 / Panel 2
    fov_rad = math.radians(fov_deg)
    pitch_rad = math.radians(pitch_deg)
    f_y = H / (2 * math.tan(fov_rad / 2))
    c_y = H / 2.0

    gamma_wall_rad = math.atan((boundary.y_upper_mid - c_y) / f_y)
    gamma_wall_deg = math.degrees(gamma_wall_rad)
    theta_wall_rad = abs(pitch_rad) + gamma_wall_rad
    theta_wall_deg = math.degrees(theta_wall_rad)

    gamma_curb_rad = math.atan((boundary.y_lower_mid - c_y) / f_y)
    gamma_curb_deg = math.degrees(gamma_curb_rad)
    theta_curb_rad = abs(pitch_rad) + gamma_curb_rad
    theta_curb_deg = math.degrees(theta_curb_rad)

    # Panel 2: 2D Pinhole Geometry Schematic
    ax_geom = axes[1]
    
    # Establish max distance for plotting
    max_z = max(z_wall if not math.isinf(z_wall) else 10.0, z_curb if not math.isinf(z_curb) else 5.0)
    limit_z = max_z + 1.5
    
    # Ground plane
    ax_geom.axhline(0, color="black", linewidth=1.5, zorder=1)
    
    # Camera pole
    ax_geom.plot([0, 0], [0, cam_height], color="#555555", linewidth=3, zorder=2)
    ax_geom.scatter(0, cam_height, color="#111111", marker="o", s=120, label="Camera Lens", zorder=6)
    
    # Horizontal reference at camera level
    ax_geom.plot([0, limit_z], [cam_height, cam_height], color="grey", linestyle=":", alpha=0.7, zorder=1)
    
    # Ray paths
    if not math.isinf(z_wall):
        ax_geom.plot([0, z_wall], [cam_height, 0], color="#22C55E", linewidth=2, linestyle="-", label="Wall Ray (Far)", zorder=3)
        ax_geom.scatter(z_wall, 0, color="#22C55E", edgecolors="black", s=70, zorder=5)
    if not math.isinf(z_curb):
        ax_geom.plot([0, z_curb], [cam_height, 0], color="#EF4444", linewidth=2, linestyle="-", label="Curb Ray (Near)", zorder=3)
        ax_geom.scatter(z_curb, 0, color="#EF4444", edgecolors="black", s=70, zorder=5)

    # Optical axis (center row c_y ray)
    z_center = cam_height / math.tan(abs(pitch_rad))
    if 0 < z_center < limit_z:
        ax_geom.plot([0, z_center], [cam_height, 0], color="grey", linewidth=1.5, linestyle="--", alpha=0.5, label="Optical Axis (c_y)", zorder=2)

    # Width Dimension Annotation
    if not math.isinf(z_wall) and not math.isinf(z_curb) and width > 0:
        y_arrow = -0.15
        ax_geom.annotate(
            "",
            xy=(z_curb, y_arrow),
            xytext=(z_wall, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="purple", lw=2, shrinkA=0, shrinkB=0),
            zorder=4
        )
        ax_geom.text(
            (z_curb + z_wall) / 2.0,
            y_arrow - 0.2,
            f"Width: {width:.3f} m",
            color="purple",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="top"
        )
        
    ax_geom.set_xlim(-0.5, limit_z)
    ax_geom.set_ylim(-0.6, cam_height + 0.6)
    ax_geom.set_xlabel("Ground Distance Z (meters)", fontsize=10)
    ax_geom.set_ylabel("Height Y (meters)", fontsize=10)
    ax_geom.set_title("2D Ground Projection Geometry", fontsize=12, fontweight="bold")
    ax_geom.grid(True, linestyle=":", alpha=0.5)
    ax_geom.legend(loc="upper right", fontsize=8)

    # Panel 3: Equations and Math Walkthrough
    ax_math = axes[2]
    ax_math.axis("off")
    
    math_text = (
        f"PINHOLE PROJECTION MATH:\n"
        f"========================\n\n"
        f"1. CAMERA CONFIGURATION:\n"
        f"  • Height (h)       = {cam_height:.2f} m\n"
        f"  • Down Pitch       = {abs(pitch_deg):.2f}° ({abs(pitch_rad):.4f} rad)\n"
        f"  • Vertical FOV     = {fov_deg:.2f}° ({fov_rad:.4f} rad)\n\n"
        f"2. CALIBRATION CONSTANTS:\n"
        f"  • Image Height (H) = {H} px\n"
        f"  • Center Row (c_y) = {c_y:.1f} px\n"
        f"  • Focal Length (f) = H / (2*tan(FOV/2)) = {f_y:.2f} px\n\n"
        f"3. UPPER BOUNDARY (WALL SIDE):\n"
        f"  • Pixel Row (y_w)  = {boundary.y_upper_mid:.2f} px\n"
        f"  • Offset (γ_w)     = atan((y_w - c_y) / f)\n"
        f"                     = {gamma_wall_deg:+.2f}° ({gamma_wall_rad:+.4f} rad)\n"
        f"  • Ray Angle (θ_w)  = |pitch| + γ_w\n"
        f"                     = {theta_wall_deg:.2f}° ({theta_wall_rad:.4f} rad)\n"
        f"  • Ground Z (Z_w)   = h / tan(θ_w) = {z_wall:.3f} m\n\n"
        f"4. LOWER BOUNDARY (CURB SIDE):\n"
        f"  • Pixel Row (y_c)  = {boundary.y_lower_mid:.2f} px\n"
        f"  • Offset (γ_c)     = atan((y_c - c_y) / f)\n"
        f"                     = {gamma_curb_deg:+.2f}° ({gamma_curb_rad:+.4f} rad)\n"
        f"  • Ray Angle (θ_c)  = |pitch| + γ_c\n"
        f"                     = {theta_curb_deg:.2f}° ({theta_curb_rad:.4f} rad)\n"
        f"  • Ground Z (Z_c)   = h / tan(θ_c) = {z_curb:.3f} m\n\n"
        f"5. ESTIMATED SIDEWALK WIDTH:\n"
        f"  • Width = Z_w - Z_c\n"
        f"          = {z_wall:.3f} - {z_curb:.3f} = {width:.3f} m"
    )
    
    ax_math.text(
        0.05, 0.95,
        math_text,
        fontsize=9,
        fontfamily="monospace",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#E2E8F0")
    )
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return buf.getvalue()
