"""
Sidewalk Visualization Pipeline
────────────────────────────────
Fetches pre-computed segmentation masks from GCS, runs geometric analysis
(edge detection, perspective rectification, footprint estimation, width
measurement via pinhole camera model), generates visualization images,
and uploads results back to GCS.  All processing is in-memory.

Usage:
    python main.py --config config.yaml
    python main.py                          # uses default config.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"

BORDER_MARGIN = 3


# ═════════════════════════════════════════════════════════════════════════════
#  Logging helpers
# ═════════════════════════════════════════════════════════════════════════════

def _banner(msg: str) -> None:
    width = max(len(msg) + 4, 50)
    print("\n" + "─" * width)
    print(f"  {msg}")
    print("─" * width)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ═════════════════════════════════════════════════════════════════════════════
#  YAML Configuration
# ═════════════════════════════════════════════════════════════════════════════

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    defaults: dict[str, Any] = {
        "gcs": {
            "bucket_name": "",
            "project_id": "",
            "image_prefix": "",
            "masks_prefix": "segmentation-results/",
            "output_prefix": "visualization-results/",
        },
        "batch": {
            "enabled": False,
            "prefix_min": 0,
            "prefix_max": 9999,
        },
        "camera": {
            "height_m": 1.8,
            "hfov_deg": 90,
            "pitch_deg": 0,
        },
        "footprint": {
            "obstacle_is_tree": ["tree"],
            "base_scan_ratio": 0.15,
            "trunk_scan_ratio": 0.40,
            "aspect_ratio": 1.0,
            "max_height_px": 25,
        },
        "width_estimation": {
            "min_horizon_dist": 20,
            "exclude_extrapolated": True,
            "iqr_factor": 1.5,
        },
        "local_output_dir": None,
    }
    for section, section_defaults in defaults.items():
        if isinstance(section_defaults, dict):
            cfg.setdefault(section, {})
            for k, v in section_defaults.items():
                cfg[section].setdefault(k, v)
        else:
            cfg.setdefault(section, section_defaults)
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
#  GCS Client  (in-memory only — no temp files)
# ═════════════════════════════════════════════════════════════════════════════

class GCSClient:
    def __init__(self, project_id: str, bucket_name: str):
        from google.cloud import storage
        self._client = storage.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def download_as_bytes(self, blob_name: str) -> bytes:
        return self._bucket.blob(blob_name).download_as_bytes()

    def upload_bytes(self, data: bytes, blob_name: str,
                     content_type: str = "image/png") -> str:
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{blob_name}"

    def list_blobs(self, prefix: str) -> list[str]:
        return [b.name for b in self._bucket.list_blobs(prefix=prefix)]


# ═════════════════════════════════════════════════════════════════════════════
#  In-memory image / mask helpers
# ═════════════════════════════════════════════════════════════════════════════

def bytes_to_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def bytes_to_mask(data: bytes) -> np.ndarray:
    mask = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return (mask > 127).astype(bool)


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def image_array_to_png_bytes(arr: np.ndarray) -> bytes:
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ═════════════════════════════════════════════════════════════════════════════
#  Filename parsing  (mirrors inference-pipeline/main.py)
# ═════════════════════════════════════════════════════════════════════════════

_FILENAME_RE = re.compile(
    r"^(\d+)_(forward|backward|left|right)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)


def _extract_numeric_prefix(filename: str) -> int | None:
    m = re.match(r"(\d+)", Path(filename).name)
    return int(m.group(1)) if m else None


def _parse_image_filename(filename: str) -> dict[str, str] | None:
    m = _FILENAME_RE.match(Path(filename).name)
    if not m:
        return None
    return {
        "index": m.group(1),
        "direction": m.group(2),
        "lat": m.group(3),
        "lon": m.group(4),
        "heading": m.group(5),
        "coordinate_folder": f"{m.group(1)}_{m.group(3)}_{m.group(4)}",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Analysis functions  (from sidewalk_measurement.ipynb)
# ═════════════════════════════════════════════════════════════════════════════

def find_row_edges(mask: np.ndarray, border_margin: int = BORDER_MARGIN):
    """Per-row left/right sidewalk edges with border extrapolation."""
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


def rectify_sidewalk(image_or_mask, left_edges, right_edges, valid_rows,
                     target_width=None, is_mask=False):
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

    MAX_STRETCH = 6.0
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


def estimate_width_footprint(mask: np.ndarray, is_tree: bool = False,
                             base_scan_ratio: float = 0.15,
                             trunk_scan_ratio: float = 0.40,
                             aspect_ratio: float = 1.0,
                             max_height: int = 25) -> np.ndarray:
    """Estimate ground-level footprint from an obstacle mask."""
    labeled, _ = label(mask, return_num=True)
    footprint = np.zeros_like(mask)

    for reg in regionprops(labeled):
        min_row, min_col, max_row, max_col = reg.bbox
        height = max_row - min_row
        component = labeled == reg.label
        if height == 0:
            continue

        scan_rows = max(1 if is_tree else 3,
                        int(height * (trunk_scan_ratio if is_tree else base_scan_ratio)))
        scan_rows = min(scan_rows, height)
        scan_start = max_row - scan_rows

        row_widths, row_centers = [], []
        for r in range(scan_start, max_row):
            cols = np.where(component[r])[0]
            if len(cols) == 0:
                continue
            row_widths.append(cols[-1] - cols[0] + 1)
            row_centers.append((cols[0] + cols[-1]) / 2.0)

        if not row_widths:
            continue

        fp_width = max(int(np.median(row_widths)), 1)
        median_center = np.median(row_centers)
        fp_height = min(max(1, int(fp_width * aspect_ratio)), height, max_height)
        fp_top = max(0, max_row - fp_height)
        fp_left = max(0, int(median_center - fp_width / 2.0))
        fp_right = min(mask.shape[1], fp_left + fp_width)

        footprint[fp_top:max_row, fp_left:fp_right] = True

    return footprint


# ═════════════════════════════════════════════════════════════════════════════
#  Data loading  (all in-memory from GCS)
# ═════════════════════════════════════════════════════════════════════════════

def load_image_and_masks(
    gcs: GCSClient,
    image_blob: str,
    masks_prefix: str,
) -> tuple[np.ndarray, dict, dict]:
    """Download original image + all masks under masks_prefix from GCS.

    Returns (original_image, sidewalk_masks, all_obstacle_masks).
    """
    original_image = bytes_to_image(gcs.download_as_bytes(image_blob))

    all_blobs = gcs.list_blobs(masks_prefix + "/")
    mask_blobs = sorted(b for b in all_blobs if b.endswith(".png"))

    sidewalk_masks: dict[str, np.ndarray] = {}
    all_obstacle_masks: dict[str, np.ndarray] = {}

    for blob_name in mask_blobs:
        parts = blob_name.split("/")
        class_name = parts[-2]
        idx = parts[-1].replace("mask_", "").replace(".png", "")
        key = f"{class_name}_{idx}"

        mask = bytes_to_mask(gcs.download_as_bytes(blob_name))
        if class_name == "sidewalk":
            sidewalk_masks[key] = mask
        else:
            all_obstacle_masks[key] = mask

    return original_image, sidewalk_masks, all_obstacle_masks


def assign_obstacles_to_sidewalks(
    sidewalk_masks: dict[str, np.ndarray],
    all_obstacle_masks: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Group obstacles to nearest sidewalk segment by horizontal proximity."""
    segments: dict[str, dict] = {}
    sw_col_ranges: dict[str, tuple[int, int]] = {}

    for sw_key, sw_mask in sidewalk_masks.items():
        segments[sw_key] = {"sidewalk_mask": sw_mask, "obstacle_masks": {}}
        sw_cols = np.where(sw_mask.any(axis=0))[0]
        if len(sw_cols) > 0:
            sw_col_ranges[sw_key] = (int(sw_cols[0]), int(sw_cols[-1]))

    for obs_key, obs_mask in all_obstacle_masks.items():
        obs_cols = np.where(obs_mask.any(axis=0))[0]
        if len(obs_cols) == 0:
            continue
        obs_center = (obs_cols[0] + obs_cols[-1]) / 2.0
        best_sw, best_dist = None, float("inf")
        for sw_key, (c_lo, c_hi) in sw_col_ranges.items():
            dist = 0.0 if c_lo <= obs_center <= c_hi else min(
                abs(obs_center - c_lo), abs(obs_center - c_hi))
            if dist < best_dist:
                best_dist, best_sw = dist, sw_key
        if best_sw is not None:
            segments[best_sw]["obstacle_masks"][obs_key] = obs_mask

    return segments


# ═════════════════════════════════════════════════════════════════════════════
#  Visualization generators  (return PNG bytes)
# ═════════════════════════════════════════════════════════════════════════════

def _make_color_palette(n: int) -> dict[int, tuple]:
    cmap = plt.cm.get_cmap("tab10", max(n, 1))
    return {i: cmap(i)[:3] for i in range(n)}


def render_obstacle_silhouettes(
    original_image: np.ndarray,
    sidewalk_mask: np.ndarray,
    obstacle_masks: dict[str, np.ndarray],
    obstacle_colors: dict[str, tuple],
    sw_key: str,
) -> bytes:
    """Obstacle silhouette overlay on the original image."""
    obstacle_union = np.zeros_like(sidewalk_mask, dtype=bool)
    for m in obstacle_masks.values():
        obstacle_union |= m
    effective = sidewalk_mask & ~obstacle_union

    overlay = np.zeros((*sidewalk_mask.shape, 3), dtype=np.float32)
    overlay[effective] = [0.2, 0.6, 1.0]
    for ot, m in obstacle_masks.items():
        overlay[m] = obstacle_colors[ot]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_image); axes[0].set_title("Original Image")
    axes[1].imshow(sidewalk_mask, cmap="Blues"); axes[1].set_title(f"Sidewalk Mask ({sw_key})")
    axes[2].imshow(overlay); axes[2].set_title("Obstacle Silhouettes")

    handles = [Patch(facecolor=(0.2, 0.6, 1.0), label="Usable sidewalk")]
    for ot in obstacle_masks:
        handles.append(Patch(facecolor=obstacle_colors[ot], label=ot))
    axes[2].legend(handles=handles, loc="upper right", fontsize=8)

    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Segment: {sw_key}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return figure_to_png_bytes(fig)


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
    """Footprint-only rectified top-view image."""
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
    return figure_to_png_bytes(fig)


def build_footprint_metadata(
    sw_rect: np.ndarray,
    obstacle_masks_full_rect: dict[str, np.ndarray],
    obstacle_masks_rect: dict[str, np.ndarray],
    obstacle_is_tree: set[str],
    target_w: int,
    pad: int,
    sw_key: str,
) -> dict:
    """Per-obstacle footprint dimensions + image-level stats."""
    obstacles_meta = []
    for ot, fp in obstacle_masks_rect.items():
        full = obstacle_masks_full_rect.get(ot, fp)
        is_tree = any(t in ot for t in obstacle_is_tree)
        labeled_fp, _ = label(fp, return_num=True)
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
    """Width-colored overlay on the original image."""
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
    return figure_to_png_bytes(fig)


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
    """Width profile plot: row vs width(cm)."""
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
    return figure_to_png_bytes(fig)


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
    camera_cfg: dict,
    width_cfg: dict,
) -> dict:
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


# ═════════════════════════════════════════════════════════════════════════════
#  Per-image analysis pipeline
# ═════════════════════════════════════════════════════════════════════════════

def process_single_image(
    gcs: GCSClient,
    image_blob: str,
    masks_prefix: str,
    output_folder: str,
    cfg: dict[str, Any],
    local_output_dir: Path | None = None,
) -> dict | None:
    """Full analysis for one streetview image. Returns summary dict or None."""
    fp_cfg = cfg["footprint"]
    cam_cfg = cfg["camera"]
    we_cfg = cfg["width_estimation"]
    obstacle_is_tree = set(fp_cfg["obstacle_is_tree"])

    parsed = _parse_image_filename(image_blob)
    label_str = parsed["coordinate_folder"] if parsed else Path(image_blob).stem

    _info(f"Loading image + masks for {label_str} ...")
    original_image, sidewalk_masks, all_obstacle_masks = load_image_and_masks(
        gcs, image_blob, masks_prefix)

    if not sidewalk_masks:
        _info(f"No sidewalk masks found under {masks_prefix} — skipping")
        return None

    segments = assign_obstacles_to_sidewalks(sidewalk_masks, all_obstacle_masks)
    H_img, W_img = original_image.shape[:2]
    cy = H_img / 2.0

    image_summary: dict[str, Any] = {
        "source_image": image_blob,
        "masks_prefix": masks_prefix,
        "image_size": [W_img, H_img],
        "segments": [],
    }

    for seg_idx, (sw_key, seg) in enumerate(segments.items()):
        sw_mask = seg["sidewalk_mask"]
        obs_masks = seg["obstacle_masks"]
        n_obs = len(obs_masks)

        _info(f"  Segment {seg_idx + 1}/{len(segments)}: {sw_key}  ({n_obs} obstacles)")

        palette = _make_color_palette(n_obs)
        obs_colors = {ot: palette[i] for i, ot in enumerate(obs_masks)}

        seg_folder = f"{output_folder}/{sw_key}"
        uploads: list[tuple[bytes, str, str]] = []

        # ── 1. Obstacle silhouettes ──────────────────────────────────
        sil_bytes = render_obstacle_silhouettes(
            original_image, sw_mask, obs_masks, obs_colors, sw_key)
        uploads.append((sil_bytes, f"{seg_folder}/obstacle_silhouettes.png", "image/png"))

        # ── 2. Rectification + footprint ─────────────────────────────
        left_e, right_e, valid_r, extrap_r = find_row_edges(sw_mask)
        if valid_r.sum() == 0:
            _info(f"    No valid edges for {sw_key} — skipping rectification & width")
            for data, blob, ct in uploads:
                gcs.upload_bytes(data, blob, ct)
            continue

        sw_rect, target_w, pad = rectify_sidewalk(
            sw_mask, left_e, right_e, valid_r, is_mask=True)

        obs_full_rect: dict[str, np.ndarray] = {}
        obs_fp_rect: dict[str, np.ndarray] = {}
        for ot, m in obs_masks.items():
            full_r, _, _ = rectify_sidewalk(
                m, left_e, right_e, valid_r, target_width=target_w, is_mask=True)
            obs_full_rect[ot] = full_r
            is_tree = any(t in ot for t in obstacle_is_tree)
            obs_fp_rect[ot] = estimate_width_footprint(
                full_r, is_tree=is_tree,
                base_scan_ratio=fp_cfg["base_scan_ratio"],
                trunk_scan_ratio=fp_cfg["trunk_scan_ratio"],
                aspect_ratio=fp_cfg["aspect_ratio"],
                max_height=fp_cfg["max_height_px"])

        fp_img_bytes = render_rectified_footprint(
            sw_rect, obs_full_rect, obs_fp_rect, obs_colors,
            obstacle_is_tree, target_w, pad, sw_key)
        uploads.append((fp_img_bytes, f"{seg_folder}/rectified_footprint.png", "image/png"))

        fp_meta = build_footprint_metadata(
            sw_rect, obs_full_rect, obs_fp_rect,
            obstacle_is_tree, target_w, pad, sw_key)
        uploads.append((json.dumps(fp_meta, indent=2).encode(),
                        f"{seg_folder}/footprint_metadata.json", "application/json"))

        # ── 3. Width estimation ──────────────────────────────────────
        valid_idx = np.where(valid_r)[0]
        valid_idx = valid_idx[valid_idx > cy]
        n_before_horizon = len(valid_idx)
        valid_idx = valid_idx[valid_idx > cy + we_cfg["min_horizon_dist"]]
        n_dropped_horizon = n_before_horizon - len(valid_idx)

        n_before_extrap = len(valid_idx)
        n_dropped_extrap = 0
        if we_cfg["exclude_extrapolated"]:
            keep = ~extrap_r[valid_idx]
            valid_idx = valid_idx[keep]
            n_dropped_extrap = n_before_extrap - len(valid_idx)

        width_meta = None
        if len(valid_idx) > 0:
            delta_v = valid_idx - cy
            width_px = right_e[valid_idx] - left_e[valid_idx]
            width_m = cam_cfg["height_m"] * width_px / delta_v
            width_cm = width_m * 100

            q1, q3 = np.percentile(width_cm, [25, 75])
            iqr = q3 - q1
            lo_fence = q1 - we_cfg["iqr_factor"] * iqr
            hi_fence = q3 + we_cfg["iqr_factor"] * iqr
            inlier = (width_cm >= lo_fence) & (width_cm <= hi_fence)
            n_outliers = int((~inlier).sum())

            width_cm_clean = width_cm[inlier]
            valid_idx_clean = valid_idx[inlier]

            if len(width_cm_clean) > 0:
                med = float(np.median(width_cm_clean))
                mean_val = float(np.mean(width_cm_clean))
                std_val = float(np.std(width_cm_clean))
                mn_val, mx_val = float(width_cm_clean.min()), float(width_cm_clean.max())

                _info(f"    Width: median={med:.1f} cm, "
                      f"mean={mean_val:.1f} cm, range={mn_val:.1f}–{mx_val:.1f} cm")

                overlay_bytes = render_width_overlay(
                    original_image, left_e, right_e,
                    valid_idx_clean, width_cm_clean,
                    med, std_val, cy, we_cfg["min_horizon_dist"], sw_key)
                uploads.append((overlay_bytes,
                                f"{seg_folder}/width_overlay.png", "image/png"))

                profile_bytes = render_width_profile(
                    valid_idx, width_cm, valid_idx_clean, width_cm_clean,
                    inlier, med, mean_val, std_val, lo_fence, hi_fence, sw_key)
                uploads.append((profile_bytes,
                                f"{seg_folder}/width_profile.png", "image/png"))

                width_meta = build_width_metadata(
                    sw_key, med, mean_val, std_val, mn_val, mx_val,
                    len(width_cm_clean), n_before_horizon,
                    n_dropped_horizon, n_dropped_extrap, n_outliers,
                    cam_cfg, we_cfg)
                uploads.append((json.dumps(width_meta, indent=2).encode(),
                                f"{seg_folder}/width_metadata.json",
                                "application/json"))
            else:
                _info(f"    No inlier rows after IQR filter for {sw_key}")
        else:
            _info(f"    No valid rows after filtering for {sw_key}")

        # ── Upload all artefacts ─────────────────────────────────────
        for data, blob_name, content_type in uploads:
            uri = gcs.upload_bytes(data, blob_name, content_type)
            _ok(f"  {Path(blob_name).name} -> {uri}")

        # ── Optional local save ──────────────────────────────────────
        if local_output_dir:
            local_seg = local_output_dir / sw_key
            local_seg.mkdir(parents=True, exist_ok=True)
            for data, blob_name, _ in uploads:
                local_path = local_seg / Path(blob_name).name
                local_path.write_bytes(data)
            _ok(f"  Local copy -> {local_seg}")

        image_summary["segments"].append({
            "segment": sw_key,
            "n_obstacles": n_obs,
            "obstacle_types": list(obs_masks.keys()),
            "footprint": fp_meta,
            "width": width_meta,
        })

    return image_summary


# ═════════════════════════════════════════════════════════════════════════════
#  Batch orchestration
# ═════════════════════════════════════════════════════════════════════════════

def run_batch(cfg: dict[str, Any]) -> None:
    gcs_cfg = cfg["gcs"]
    batch_cfg = cfg["batch"]

    gcs = GCSClient(project_id=gcs_cfg["project_id"],
                    bucket_name=gcs_cfg["bucket_name"])

    image_prefix = gcs_cfg["image_prefix"].rstrip("/") + "/"
    masks_prefix = gcs_cfg["masks_prefix"].rstrip("/")
    output_prefix = gcs_cfg["output_prefix"].rstrip("/")
    pmin, pmax = batch_cfg["prefix_min"], batch_cfg["prefix_max"]

    local_out = Path(cfg["local_output_dir"]) if cfg.get("local_output_dir") else None

    _banner("Sidewalk Visualization Pipeline — Batch Mode")
    _info(f"Image prefix  : {image_prefix}")
    _info(f"Masks prefix  : {masks_prefix}")
    _info(f"Output prefix : {output_prefix}")
    _info(f"Prefix range  : {pmin} – {pmax}")

    all_image_blobs = gcs.list_blobs(image_prefix)
    image_blobs = [b for b in all_image_blobs
                   if b.lower().endswith((".jpg", ".jpeg", ".png"))]

    filtered: list[tuple[int, str]] = []
    for blob in image_blobs:
        num = _extract_numeric_prefix(blob)
        if num is not None and pmin <= num <= pmax:
            filtered.append((num, blob))
    filtered.sort(key=lambda t: t[0])

    if not filtered:
        _info("No images matched the prefix range.")
        return

    _info(f"Matched {len(filtered)} / {len(image_blobs)} images")

    all_summaries: list[dict] = []
    t0 = time.time()

    for idx, (num, blob) in enumerate(filtered, 1):
        parsed = _parse_image_filename(blob)
        if parsed is None:
            _info(f"[{idx}/{len(filtered)}] Skipping (unrecognised name): {Path(blob).name}")
            continue

        coord_folder = parsed["coordinate_folder"]
        direction = parsed["direction"]
        img_masks = f"{masks_prefix}/{coord_folder}/{direction}"
        img_output = f"{output_prefix}/{coord_folder}/{direction}"

        _banner(f"[{idx}/{len(filtered)}] {Path(blob).name}")
        summary = process_single_image(
            gcs, blob, img_masks, img_output, cfg,
            local_output_dir=local_out / coord_folder / direction if local_out else None)

        if summary:
            all_summaries.append(summary)

    elapsed = time.time() - t0

    # ── Batch summary ────────────────────────────────────────────────
    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_prefix": image_prefix,
        "prefix_range": [pmin, pmax],
        "images_processed": len(all_summaries),
        "elapsed_s": round(elapsed, 1),
        "per_image": all_summaries,
    }
    summary_blob = f"{output_prefix}/_batch_summary.json"
    gcs.upload_bytes(json.dumps(batch_summary, indent=2).encode(),
                     summary_blob, "application/json")

    _banner("Batch Complete")
    _info(f"Images processed : {len(all_summaries)}")
    _info(f"Elapsed          : {elapsed:.1f}s")
    _ok(f"Summary -> gs://{gcs.bucket_name}/{summary_blob}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sidewalk visualization pipeline — analysis & uploads to GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (default: visualization-pipeline/config.yaml)")
    p.add_argument("--image", type=str, default=None,
                   help="Process a single image blob path (overrides batch mode)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Also save results locally to this directory")
    return p


def main():
    args = build_parser().parse_args()

    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)

    cfg = load_config(args.config)

    if args.output_dir:
        cfg["local_output_dir"] = args.output_dir

    if args.image:
        gcs_cfg = cfg["gcs"]
        gcs = GCSClient(project_id=gcs_cfg["project_id"],
                        bucket_name=gcs_cfg["bucket_name"])

        parsed = _parse_image_filename(args.image)
        if parsed is None:
            _fail(f"Cannot parse image filename: {args.image}")
            sys.exit(1)

        masks_prefix = gcs_cfg["masks_prefix"].rstrip("/")
        output_prefix = gcs_cfg["output_prefix"].rstrip("/")
        coord = parsed["coordinate_folder"]
        direction = parsed["direction"]

        local_out = Path(cfg["local_output_dir"]) if cfg.get("local_output_dir") else None

        _banner("Sidewalk Visualization Pipeline — Single Image")
        process_single_image(
            gcs, args.image,
            f"{masks_prefix}/{coord}/{direction}",
            f"{output_prefix}/{coord}/{direction}",
            cfg,
            local_output_dir=local_out / coord / direction if local_out else None)
        _banner("Done")
    elif cfg["batch"].get("enabled"):
        run_batch(cfg)
    else:
        _info("Nothing to do. Set batch.enabled=true or use --image.")


if __name__ == "__main__":
    main()
