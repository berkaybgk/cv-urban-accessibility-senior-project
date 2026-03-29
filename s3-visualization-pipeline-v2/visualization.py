"""
Visualization functions for the sidewalk analysis pipeline v2.

All rendering follows these rules:
  - Legends are placed in a strip BELOW the main image, never overlapping.
  - Consistent color conventions across all outputs.
  - Every figure uses a fixed DPI for reproducible sizing.

Outputs:
  - obstacle_overlay: original image + segmentation masks side by side
  - rectified_footprint: rectified sidewalk strip with obstacle + footprint views
  - width_overlay: original image with coloured width bands on the sidewalk
  - width_profile: width (cm) vs image row chart
"""

from __future__ import annotations

import io

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import matplotlib.gridspec as gridspec
import numpy as np


OUTPUT_DPI = 150

SIDEWALK_COLOR = np.array([0.6, 0.85, 1.0])
SIDEWALK_EDGE_COLOR = "lime"
OBSTACLE_CMAP = plt.cm.tab10


def _figure_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=OUTPUT_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _make_legend_strip(ax_legend, patches: list[Patch], ncol: int = 4):
    """Place a legend in a dedicated axes strip below the main image."""
    ax_legend.axis("off")
    ax_legend.legend(
        handles=patches,
        loc="center",
        ncol=ncol,
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#cccccc",
    )


def render_obstacle_overlay(
    original_image: np.ndarray,
    sw_mask: np.ndarray,
    obs_masks: dict[str, np.ndarray],
) -> bytes:
    """Input visualization: original image with sidewalk and obstacle overlays."""
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.06], hspace=0.08, wspace=0.05)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])

    ax_orig.imshow(original_image)
    ax_orig.set_title("Original", fontsize=10)
    ax_orig.axis("off")

    overlay = original_image.astype(np.float32) / 255.0
    sw_rgb = np.zeros_like(overlay)
    sw_rgb[sw_mask] = SIDEWALK_COLOR
    overlay = np.where(sw_mask[..., None], overlay * 0.4 + sw_rgb * 0.6, overlay)

    patches = [Patch(facecolor=SIDEWALK_COLOR, label="Sidewalk")]

    for i, (key, mask) in enumerate(obs_masks.items()):
        color = OBSTACLE_CMAP(i % 10)[:3]
        obs_rgb = np.zeros_like(overlay)
        obs_rgb[mask] = color
        overlay = np.where(mask[..., None], overlay * 0.3 + obs_rgb * 0.7, overlay)
        patches.append(Patch(facecolor=color, label=key))

    ax_overlay.imshow(np.clip(overlay, 0, 1))
    ax_overlay.set_title("Segmentation overlay", fontsize=10)
    ax_overlay.axis("off")

    _make_legend_strip(ax_legend, patches)
    return _figure_to_bytes(fig)


def render_rectified_footprint(
    sw_rect: np.ndarray,
    obs_full_rect: dict[str, np.ndarray],
    obs_fp_rect: dict[str, np.ndarray],
    target_width: int,
    padding: int,
    img_rect: np.ndarray | None = None,
) -> bytes:
    """
    Rectified sidewalk strip with obstacle full masks and footprints.

    Layout: 3 columns — rectified image, full obstacles, footprints.
    Vertical lines mark the sidewalk edges (at padding and padding+target_width).
    """
    ncols = 3 if img_rect is not None else 2
    fig = plt.figure(figsize=(3 * ncols, 10))
    gs = gridspec.GridSpec(2, ncols, height_ratios=[1, 0.05], hspace=0.05, wspace=0.08)

    col_idx = 0

    # Rectified image (if available)
    if img_rect is not None:
        ax_img = fig.add_subplot(gs[0, col_idx])
        ax_img.imshow(img_rect)
        ax_img.axvline(padding, color="lime", linewidth=1, linestyle="--")
        ax_img.axvline(padding + target_width, color="lime", linewidth=1, linestyle="--")
        ax_img.set_title("Rectified image", fontsize=9)
        ax_img.axis("off")
        col_idx += 1

    # Sidewalk + obstacle full masks
    ax_full = fig.add_subplot(gs[0, col_idx])
    H_r, W_r = sw_rect.shape
    canvas_full = np.zeros((H_r, W_r, 3), dtype=np.float32)
    canvas_full[sw_rect] = SIDEWALK_COLOR

    patches = [Patch(facecolor=SIDEWALK_COLOR, label="Sidewalk")]

    for i, (key, mask) in enumerate(obs_full_rect.items()):
        color = np.array(OBSTACLE_CMAP(i % 10)[:3])
        if mask.shape == (H_r, W_r):
            canvas_full[mask] = color
        patches.append(Patch(facecolor=color, label=key))

    ax_full.imshow(canvas_full)
    ax_full.axvline(padding, color="white", linewidth=1, linestyle="--", alpha=0.6)
    ax_full.axvline(padding + target_width, color="white", linewidth=1,
                    linestyle="--", alpha=0.6)
    ax_full.set_title("Full masks", fontsize=9)
    ax_full.axis("off")
    col_idx += 1

    # Sidewalk + obstacle footprints
    ax_fp = fig.add_subplot(gs[0, col_idx])
    canvas_fp = np.zeros((H_r, W_r, 3), dtype=np.float32)
    canvas_fp[sw_rect] = SIDEWALK_COLOR * 0.5

    for i, (key, fp_mask) in enumerate(obs_fp_rect.items()):
        color = np.array(OBSTACLE_CMAP(i % 10)[:3])
        if fp_mask.shape == (H_r, W_r):
            canvas_fp[fp_mask] = color

    ax_fp.imshow(canvas_fp)
    ax_fp.axvline(padding, color="white", linewidth=1, linestyle="--", alpha=0.6)
    ax_fp.axvline(padding + target_width, color="white", linewidth=1,
                  linestyle="--", alpha=0.6)
    ax_fp.set_title("Footprints", fontsize=9)
    ax_fp.axis("off")

    ax_legend = fig.add_subplot(gs[1, :])
    patches.append(Patch(facecolor="none", edgecolor="white",
                         linestyle="--", label="Sidewalk edges"))
    _make_legend_strip(ax_legend, patches)
    return _figure_to_bytes(fig)


def render_width_overlay(
    original_image: np.ndarray,
    sw_mask: np.ndarray,
    widths_cm: np.ndarray,
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    valid: np.ndarray,
) -> bytes:
    """
    Original image with coloured width bands on the sidewalk.

    Width bands are coloured by a diverging colourmap centred on the
    median width.  Annotations at 25/50/75% positions show the width.
    """
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05], hspace=0.05)

    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    overlay = original_image.astype(np.float32) / 255.0 * 0.5
    ax.imshow(overlay)

    valid_rows = np.where(valid)[0]
    valid_with_width = [r for r in valid_rows if not np.isnan(widths_cm[r])]

    if len(valid_with_width) == 0:
        ax.set_title("Width overlay (no valid measurements)", fontsize=10)
        ax.axis("off")
        ax_legend.axis("off")
        return _figure_to_bytes(fig)

    valid_widths = widths_cm[valid_with_width]
    median_w = np.nanmedian(valid_widths)
    std_w = np.nanstd(valid_widths)
    vmin = max(0, median_w - 3 * std_w)
    vmax = median_w + 3 * std_w
    cmap = plt.cm.RdYlGn

    for r in valid_with_width:
        if np.isnan(left_edges[r]) or np.isnan(right_edges[r]):
            continue
        L = int(left_edges[r])
        R = int(right_edges[r])
        w = widths_cm[r]
        t = np.clip((w - vmin) / (vmax - vmin + 1e-9), 0, 1)
        color = cmap(t)[:3]
        ax.plot([L, R], [r, r], color=color, linewidth=1.0, alpha=0.8)

    # Annotate at 25/50/75% positions
    for pct in [0.25, 0.5, 0.75]:
        idx = int(pct * (len(valid_with_width) - 1))
        r = valid_with_width[idx]
        w = widths_cm[r]
        if np.isnan(w):
            continue
        cx = (left_edges[r] + right_edges[r]) / 2
        ax.annotate(f"{w:.0f} cm", xy=(cx, r), fontsize=8, color="white",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

    ax.set_title(f"Sidewalk width (median {median_w:.0f} cm)", fontsize=10)
    ax.axis("off")

    patches = [
        Patch(facecolor=cmap(0.0)[:3], label=f"Narrow ({vmin:.0f} cm)"),
        Patch(facecolor=cmap(0.5)[:3], label=f"Median ({median_w:.0f} cm)"),
        Patch(facecolor=cmap(1.0)[:3], label=f"Wide ({vmax:.0f} cm)"),
    ]
    _make_legend_strip(ax_legend, patches, ncol=3)
    return _figure_to_bytes(fig)


def render_width_profile(
    widths_cm: np.ndarray,
    iqr_factor: float = 1.5,
) -> bytes:
    """
    Width profile plot: width (cm) vs image row.

    Unlike the BEV version, the Y-axis is the image row (bottom =
    near, top = far).  IQR fences and outliers are shown.
    """
    valid_idx = np.where(~np.isnan(widths_cm))[0]
    if len(valid_idx) == 0:
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.text(0.5, 0.5, "No valid width data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return _figure_to_bytes(fig)

    valid_widths = widths_cm[valid_idx]

    q1, q3 = np.nanpercentile(valid_widths, [25, 75])
    iqr = q3 - q1
    lo_fence = q1 - iqr_factor * iqr
    hi_fence = q3 + iqr_factor * iqr
    inlier = (valid_widths >= lo_fence) & (valid_widths <= hi_fence)

    median_w = np.nanmedian(valid_widths[inlier]) if inlier.any() else np.nanmedian(valid_widths)
    std_w = np.nanstd(valid_widths[inlier]) if inlier.any() else np.nanstd(valid_widths)

    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    # Plot with image row on Y-axis (inverted: top = far, bottom = near)
    ax.plot(valid_widths, valid_idx, color="#aaaaaa", linewidth=0.8, alpha=0.6)
    ax.plot(valid_widths[inlier], valid_idx[inlier], color="#3377cc",
            linewidth=1.0, label="Inlier")
    outlier_mask = ~inlier
    if outlier_mask.any():
        ax.scatter(valid_widths[outlier_mask], valid_idx[outlier_mask],
                   color="red", s=8, zorder=5, label="Outlier")

    ax.axvline(median_w, color="#cc3333", linestyle="--", linewidth=1.0)
    ax.axvspan(median_w - std_w, median_w + std_w, alpha=0.15, color="#3377cc")
    ax.axvspan(lo_fence, hi_fence, alpha=0.08, color="#33cc33")

    ax.set_xlabel("Width (cm)", fontsize=9)
    ax.set_ylabel("Image row (top = far, bottom = near)", fontsize=9)
    ax.set_title("Sidewalk width profile", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    patches = [
        Patch(facecolor="#3377cc", alpha=0.3, label=f"Median {median_w:.0f} +/- {std_w:.0f} cm"),
        Patch(facecolor="#33cc33", alpha=0.2, label=f"IQR fence [{lo_fence:.0f}, {hi_fence:.0f}]"),
        Patch(facecolor="red", label="Outliers"),
    ]
    _make_legend_strip(ax_legend, patches, ncol=3)
    return _figure_to_bytes(fig)
