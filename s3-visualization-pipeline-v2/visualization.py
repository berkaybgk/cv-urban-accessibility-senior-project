"""
Visualization outputs aligned with s3-visualization-pipeline (v1).
"""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


OUTPUT_DPI = 150


def _figure_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=OUTPUT_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_obstacle_silhouettes(
    original_image: np.ndarray,
    sidewalk_mask: np.ndarray,
    obstacle_masks: dict[str, np.ndarray],
    obstacle_colors: dict[str, tuple],
    sw_key: str,
) -> bytes:
    """Obstacle silhouette overlay on the original image (v1 layout)."""
    obstacle_union = np.zeros_like(sidewalk_mask, dtype=bool)
    for m in obstacle_masks.values():
        obstacle_union |= m
    effective = sidewalk_mask & ~obstacle_union

    overlay = np.zeros((*sidewalk_mask.shape, 3), dtype=np.float32)
    overlay[effective] = [0.2, 0.6, 1.0]
    for ot, m in obstacle_masks.items():
        overlay[m] = obstacle_colors[ot]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[1].imshow(sidewalk_mask, cmap="Blues")
    axes[1].set_title(f"Sidewalk Mask ({sw_key})")
    axes[2].imshow(overlay)
    axes[2].set_title("Obstacle Silhouettes")

    handles = [Patch(facecolor=(0.2, 0.6, 1.0), label="Usable sidewalk")]
    for ot in obstacle_masks:
        handles.append(Patch(facecolor=obstacle_colors[ot], label=ot))
    axes[2].legend(handles=handles, loc="upper right", fontsize=8)

    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Segment: {sw_key}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _figure_to_bytes(fig)


def render_rectified_footprint(
    sw_rect: np.ndarray,
    obstacle_masks_full_rect: dict[str, np.ndarray],
    obstacle_masks_rect: dict[str, np.ndarray],
    obstacle_colors: dict[str, tuple],
    obstacle_is_tree: set[str],
    target_w: int,
    pad: int,
    sw_key: str,
) -> bytes:
    """Footprint-only rectified top-view image (v1 layout)."""
    tv_fp = np.ones((*sw_rect.shape, 3), dtype=np.float32) * 0.15
    tv_fp[sw_rect] = [0.75, 0.9, 1.0]

    for ot, full in obstacle_masks_full_rect.items():
        fp = obstacle_masks_rect.get(ot, np.zeros_like(full))
        above = full & ~fp
        tv_fp[above] = [0.75, 0.9, 1.0]
    for ot, fp in obstacle_masks_rect.items():
        tv_fp[fp] = obstacle_colors.get(ot, (1.0, 0.2, 0.2))

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(tv_fp)
    ax.axvline(x=pad, color="white", linewidth=1.5, alpha=0.6)
    ax.axvline(x=pad + target_w, color="white", linewidth=1.5, alpha=0.6)

    lh = [Patch(facecolor=(0.75, 0.9, 1.0), edgecolor="gray", label="Sidewalk")]
    for ot in obstacle_masks_rect:
        is_tree = any(t in ot for t in obstacle_is_tree)
        method = "trunk" if is_tree else "base"
        lh.append(Patch(facecolor=obstacle_colors[ot], label=f"{ot} ({method})"))
    ax.legend(handles=lh, loc="upper right", fontsize=9, framealpha=0.9,
              facecolor="white")
    ax.set_title(f"Rectified Footprint — {sw_key}", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return _figure_to_bytes(fig)


def render_width_overlay(
    original_image: np.ndarray,
    left_e: np.ndarray,
    right_e: np.ndarray,
    valid_idx_clean: np.ndarray,
    width_cm_clean: np.ndarray,
    med: float,
    std: float,
    cy: float,
    min_horizon_dist: int,
    sw_key: str,
) -> bytes:
    """Width-colored overlay on the original image (v1 layout)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(original_image, alpha=0.7)
    ax.axhline(y=cy, color="cyan", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Horizon")
    ax.axhline(y=cy + min_horizon_dist, color="orange", linewidth=0.8,
               linestyle=":", alpha=0.5, label=f"Min dist ({min_horizon_dist}px)")

    norm = plt.Normalize(vmin=max(0, med - 3 * std), vmax=med + 3 * std)
    cmap = plt.cm.RdYlGn
    for i, v in enumerate(valid_idx_clean):
        ax.plot([left_e[v], right_e[v]], [v, v],
                color=cmap(norm(width_cm_clean[i])), lw=0.6, alpha=0.6)

    for frac in (0.25, 0.5, 0.75):
        j = int(len(valid_idx_clean) * frac)
        if len(valid_idx_clean) == 0:
            break
        j = min(j, len(valid_idx_clean) - 1)
        v = valid_idx_clean[j]
        w = width_cm_clean[j]
        ax.annotate(f"{w:.0f} cm",
                    xy=((left_e[v] + right_e[v]) / 2, v),
                    fontsize=9, color="white", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))

    ax.set_title(f"Width overlay — {sw_key}")
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")
    plt.suptitle(f"Sidewalk Width Estimation — {sw_key}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _figure_to_bytes(fig)


def render_width_profile(
    valid_idx: np.ndarray,
    width_cm: np.ndarray,
    valid_idx_clean: np.ndarray,
    width_cm_clean: np.ndarray,
    inlier: np.ndarray,
    med: float,
    mean: float,
    std: float,
    lo_fence: float,
    hi_fence: float,
    sw_key: str,
) -> bytes:
    """Width profile plot: row vs width(cm) (v1 layout)."""
    n_outliers = int((~inlier).sum())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(valid_idx, width_cm, color="gray", lw=0.5, alpha=0.3, label="All rows")
    ax.plot(valid_idx_clean, width_cm_clean, "b-", lw=0.8, alpha=0.5, label="Inliers")

    if n_outliers > 0:
        outlier_idx = valid_idx[~inlier]
        outlier_vals = width_cm[~inlier]
        ax.scatter(outlier_idx, outlier_vals, c="red", s=8, zorder=5,
                   label=f"Outliers ({n_outliers})")

    ax.axhline(med, color="r", ls="--", label=f"Median {med:.1f} cm")
    ax.fill_between(valid_idx_clean, mean - std, mean + std,
                    color="blue", alpha=0.08, label="\u00b11 \u03c3")
    ax.axhspan(lo_fence, hi_fence, color="green", alpha=0.04, label="IQR fence")
    ax.set_xlabel("Image row")
    ax.set_ylabel("Sidewalk width (cm)")
    ax.set_title(f"Width profile — {sw_key}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _figure_to_bytes(fig)
