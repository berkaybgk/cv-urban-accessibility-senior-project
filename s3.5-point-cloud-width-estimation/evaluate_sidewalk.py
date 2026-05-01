#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sidewalk width measurement & evaluation using VGGT + SegFormer.

Improved algorithm: column-wise scanning of segmentation mask replaces
morphological boundary extraction, avoiding inner/outer confusion,
red-yellow stickiness, and multiple-line artifacts.

Usage:
    python evaluate_sidewalk.py --img_dir streetview_out --gt_csv submission_result.csv
    python evaluate_sidewalk.py --image side_image.jpg --visualize --plot3d --out_dir eval_one
    python evaluate_sidewalk.py --img_dir streetview_out --cam_height 2.5 --visualize

Apple Silicon (M2): uses MPS when available. For a manually downloaded VGGT checkpoint:
    export VGGT_WEIGHTS="$PWD/weights/model.pt"
    python evaluate_sidewalk.py --img_dir streetview_out --gt_csv submission_result.csv
    # or: --vggt_weights weights/model.pt
"""

import os
import glob
import csv
import argparse
import contextlib
from collections import Counter
from typing import Tuple, List, Dict, Optional

import numpy as np
import cv2
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


# ============================================================
# Device & VGGT loading (local weights / MPS)
# ============================================================

def resolve_device(preference: str) -> str:
    """Return 'cuda', 'mps', or 'cpu' from user preference."""
    pref = preference.lower().strip()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return "cuda"
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available.")
        return "mps"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_context(device: str):
    """Mixed-precision context appropriate for the device."""
    if device == "cuda":
        dtype = (torch.bfloat16
                 if torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
        return torch.amp.autocast("cuda", dtype=dtype)
    if device == "mps":
        try:
            return torch.amp.autocast("mps", dtype=torch.float16)
        except Exception:
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def maybe_empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def load_vggt(weights_path: Optional[str], device: str) -> VGGT:
    """Load VGGT from a local model.pt or from Hugging Face (facebook/VGGT-1B)."""
    if weights_path:
        path = os.path.expanduser(weights_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"VGGT weights not found: {path}")
        print(f"[INFO] Loading VGGT weights from file: {path}")
        model = VGGT()
        load_kw = {"map_location": "cpu"}
        try:
            ckpt = torch.load(path, **load_kw, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, **load_kw)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unexpected checkpoint format in {path}")
        key0 = next(iter(ckpt.keys()))
        if isinstance(key0, str) and key0.startswith("module."):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=True)
        return model.to(device)

    print("[INFO] Loading VGGT from Hugging Face: facebook/VGGT-1B")
    return VGGT.from_pretrained("facebook/VGGT-1B").to(device)


# ============================================================
# Geometry helpers
# ============================================================

def find_contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True runs in a 1D boolean array.
    Returns list of (start_inclusive, end_exclusive) tuples."""
    if len(mask) == 0 or not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def batch_bilinear_sample(array: np.ndarray,
                          xs: np.ndarray,
                          ys: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of array[y, x] for multiple float coords.
    array: [H, W, C], xs/ys: 1D float arrays. Returns [N, C]."""
    H, W, C = array.shape
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)

    wx = (xs - x0)[:, None]
    wy = (ys - y0)[:, None]
    val = (array[y0, x0] * (1 - wx) * (1 - wy) +
           array[y0, x1] * wx * (1 - wy) +
           array[y1, x0] * (1 - wx) * wy +
           array[y1, x1] * wx * wy)
    return val


def fit_ground_plane(points: np.ndarray,
                     iters: int = 1000,
                     min_inlier_ratio: float = 0.3
                     ) -> Optional[Tuple[np.ndarray, float]]:
    """RANSAC plane fitting with MAD-adaptive threshold.
    Returns (unit_normal, d) where n·x + d = 0, or None."""
    N = len(points)
    if N < 50:
        return None

    # Coarse SVD fit for threshold estimation
    mean = points.mean(0, keepdims=True)
    P = points - mean
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    n_c = vh[-1]
    n_c /= np.linalg.norm(n_c) + 1e-12
    d_c = -float(np.dot(n_c, mean.ravel()))
    dists = np.abs(points @ n_c + d_c)
    mad = np.median(np.abs(dists - np.median(dists))) + 1e-12
    thresh = float(np.clip(2.5 * 1.4826 * mad, 0.005, 0.05))

    rng = np.random.default_rng(42)
    best_count = 0
    best_n, best_d = n_c, d_c

    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -float(np.dot(n, p0))
        count = int((np.abs(points @ n + d) < thresh).sum())
        if count > best_count:
            best_count = count
            best_n, best_d = n, d

    if best_count < max(int(N * min_inlier_ratio), 30):
        return (n_c, d_c)

    # Refine with inliers
    inlier_mask = np.abs(points @ best_n + best_d) < thresh
    inlier_pts = points[inlier_mask]
    P = inlier_pts - inlier_pts.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    n = vh[-1]
    n /= np.linalg.norm(n) + 1e-12
    d = -float(np.dot(n, inlier_pts.mean(0)))
    return (n, d)


def project_to_plane(points: np.ndarray,
                     n: np.ndarray,
                     d: float) -> np.ndarray:
    """Project 3D points onto plane n·x + d = 0.
    points: (N,3). Returns (N,3)."""
    signed = (points @ n + d)[:, None]
    return points - signed * n[None, :]


def gather_ground_points(world_points: np.ndarray,
                         seg_map: np.ndarray,
                         id_sidewalk: int,
                         id_road: int,
                         max_samples: int = 40000) -> Optional[np.ndarray]:
    """Collect 3D points from road + sidewalk regions for plane fitting."""
    H, W, _ = world_points.shape
    seg = seg_map
    if seg.shape[0] != H or seg.shape[1] != W:
        seg = cv2.resize(seg.astype(np.int32), (W, H),
                         interpolation=cv2.INTER_NEAREST)

    mask = (seg == id_sidewalk) | (seg == id_road)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    idx = np.arange(len(xs))
    if len(idx) > max_samples:
        idx = np.random.default_rng(0).choice(idx, size=max_samples, replace=False)

    pts = world_points[ys[idx], xs[idx]]
    pts = pts[np.isfinite(pts).all(1)]
    return pts if len(pts) >= 50 else None


# ============================================================
# Quality check
# ============================================================

def check_quality(seg_map: np.ndarray,
                  id_sidewalk: int,
                  id_road: int,
                  min_sw_frac: float = 0.02,
                  band_frac: float = 0.20,
                  min_col_coverage: float = 0.30
                  ) -> Tuple[bool, str]:
    """Check if an image is suitable for sidewalk width measurement."""
    H, W = seg_map.shape
    total = H * W

    sw_area = int((seg_map == id_sidewalk).sum())
    if sw_area < min_sw_frac * total:
        return False, "Q:sw_area_small"

    road_area = int((seg_map == id_road).sum())
    if road_area < 0.01 * total:
        return False, "Q:road_area_small"

    # Check sidewalk presence in midline band
    x_mid = W // 2
    half_band = max(10, int(W * band_frac / 2))
    x_start = max(0, x_mid - half_band)
    x_end = min(W, x_mid + half_band)
    band_width = x_end - x_start

    cols_with_sw = 0
    for x in range(x_start, x_end):
        if (seg_map[:, x] == id_sidewalk).any():
            cols_with_sw += 1

    if cols_with_sw < min_col_coverage * band_width:
        return False, "Q:midline_coverage_low"

    return True, "ok"


# ============================================================
# Core measurement: column-wise scanning
# ============================================================

def measure_width_columnwise(
        world_points: np.ndarray,    # [H_d, W_d, 3]
        seg_map: np.ndarray,         # [H_img, W_img] int
        pose_enc: np.ndarray,        # [9]: cx,cy,cz, qw,qx,qy,qz, fovh,fovw
        cam_height: float,
        id_sidewalk: int,
        id_road: int,
        img_gray: Optional[np.ndarray] = None,
        band_frac: float = 0.20,
        min_run_px: int = 15,
        min_valid_cols: int = 10,
        width_range: Tuple[float, float] = (0.50, 4.00),
        max_phys_width: float = 5.0,
        max_width_cv: float = 0.40,
        collect_geometry: bool = False,
) -> Tuple[Optional[float], int, dict, str]:
    """
    Measure sidewalk width via column-wise scanning of segmentation mask.

    Returns (width_m, n_valid_cols, info_dict, fail_reason).
    On success: (width, n, info, "").
    On failure: (None, 0, {}, "M:reason").
    """
    H_img, W_img = seg_map.shape
    H_d, W_d = world_points.shape[:2]
    sx = (W_d - 1) / max(W_img - 1, 1)
    sy = (H_d - 1) / max(H_img - 1, 1)

    # -- Clean sidewalk mask: remove small components, fill small holes --
    sw_mask = (seg_map == id_sidewalk).astype(np.uint8)
    min_area = max(50, int(0.001 * H_img * W_img))
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(sw_mask, connectivity=8)
    for k in range(1, num_cc):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            sw_mask[labels == k] = 0
    sw_mask = cv2.morphologyEx(sw_mask, cv2.MORPH_CLOSE,
                               np.ones((3, 3), np.uint8))

    # -- Define midline band --
    x_mid = W_img // 2
    half_band = max(10, int(W_img * band_frac / 2))
    x_start = max(0, x_mid - half_band)
    x_end = min(W_img, x_mid + half_band)

    # -- Determine expected y-center of sidewalk in the midline band --
    y_centers = []
    for x in range(x_start, x_end):
        col = sw_mask[:, x]
        if col.any():
            ys = np.where(col > 0)[0]
            y_centers.append(float(np.median(ys)))
    if not y_centers:
        return None, 0, {}, "M:no_sw_in_band"
    y_center_global = np.median(y_centers)

    # -- Column-wise edge detection --
    edge_xs = []
    edge_y_tops = []
    edge_y_bots = []
    n_border_skip = 0

    for x in range(x_start, x_end):
        col = sw_mask[:, x].astype(bool)
        runs = find_contiguous_runs(col)
        if not runs:
            continue

        # Select run: prefer long runs close to the expected y-center
        best_run = None
        best_score = -1.0
        for rs, re in runs:
            length = re - rs
            if length < min_run_px:
                continue
            mid_y = (rs + re) / 2.0
            dist = abs(mid_y - y_center_global)
            score = length / (1.0 + dist / 50.0)
            if score > best_score:
                best_score = score
                best_run = (rs, re)

        if best_run is None:
            continue

        y_top = best_run[0]       # inner edge (building side)
        y_bot = best_run[1] - 1   # outer edge (road side)

        # Skip if touching image borders (can't see true edge)
        if y_top <= 1 or y_bot >= H_img - 2:
            n_border_skip += 1
            continue

        edge_xs.append(x)
        edge_y_tops.append(y_top)
        edge_y_bots.append(y_bot)

    if len(edge_xs) < min_valid_cols:
        return None, 0, {}, f"M:too_few_cols({len(edge_xs)},border={n_border_skip})"

    edge_xs = np.array(edge_xs, dtype=np.float64)
    edge_y_tops = np.array(edge_y_tops, dtype=np.float64)
    edge_y_bots = np.array(edge_y_bots, dtype=np.float64)

    # -- Geometric gap constraint (detect & correct over-segmentation) --
    # At distance d, a sidewalk of width w spans:
    #   gap_px = f_y * cam_height * w / (d * (d + w))
    # If actual pixel gap >> expected max, SegFormer over-segmented.
    overseg_corrected = False
    fov_h = float(pose_enc[7])  # vertical FOV in radians
    if fov_h > 0.05:
        f_y_img = (H_img / 2.0) / (np.tan(fov_h / 2.0) + 1e-9)
        y_horizon = H_img / 2.0

        med_y_bot = np.median(edge_y_bots)
        dy_bot = med_y_bot - y_horizon

        if dy_bot > 5:
            d_out = f_y_img * cam_height / dy_bot
            max_gap = (f_y_img * cam_height * max_phys_width
                       / (d_out * (d_out + max_phys_width)))

            med_gap = np.median(edge_y_bots - edge_y_tops)

            if med_gap > max_gap * 1.3:
                # Over-segmentation detected – try gradient refinement
                refined_y_top = None

                if img_gray is not None:
                    # Search for strongest horizontal edge within
                    # the geometrically plausible region
                    y_st = max(0, int(med_y_bot - max_gap * 1.5))
                    y_sb = int(med_y_bot)
                    x_lo = int(edge_xs.min())
                    x_hi = int(edge_xs.max()) + 1

                    if y_sb - y_st > 3 and x_hi - x_lo > 3:
                        region = img_gray[y_st:y_sb,
                                          x_lo:x_hi].astype(np.float32)
                        if region.shape[0] > 3:
                            grad_y = np.abs(np.diff(region, axis=0))
                            profile = grad_y.mean(axis=1)

                            # Smooth the 1-D gradient profile
                            k = min(5, max(3, len(profile) // 5))
                            k = k if k % 2 == 1 else k + 1
                            kernel = np.ones(k) / k
                            if len(profile) >= k:
                                profile = np.convolve(
                                    profile, kernel, mode='same')

                            peak = int(np.argmax(profile))
                            if profile[peak] > 3.0:
                                refined_y_top = float(y_st + peak)

                if refined_y_top is not None:
                    edge_y_tops[:] = refined_y_top
                    overseg_corrected = True
                else:
                    return None, 0, {}, (
                        f"M:over_seg(gap={med_gap:.0f},"
                        f"max={max_gap:.0f},d={d_out:.1f}m)")

    # -- Remove columns with inconsistent edges (MAD filter) --
    med_top = np.median(edge_y_tops)
    mad_top = np.median(np.abs(edge_y_tops - med_top)) * 1.4826 + 1e-6
    med_bot = np.median(edge_y_bots)
    mad_bot = np.median(np.abs(edge_y_bots - med_bot)) * 1.4826 + 1e-6

    consistent = ((np.abs(edge_y_tops - med_top) < 3 * mad_top) &
                  (np.abs(edge_y_bots - med_bot) < 3 * mad_bot))

    edge_xs = edge_xs[consistent]
    edge_y_tops = edge_y_tops[consistent]
    edge_y_bots = edge_y_bots[consistent]

    if len(edge_xs) < min_valid_cols:
        return None, 0, {}, f"M:inconsistent_edges({int(consistent.sum())})"

    # -- Map to VGGT resolution and sample 3D points --
    xs_d = edge_xs * sx
    ys_d_top = edge_y_tops * sy
    ys_d_bot = edge_y_bots * sy

    pts_top = batch_bilinear_sample(world_points, xs_d, ys_d_top)
    pts_bot = batch_bilinear_sample(world_points, xs_d, ys_d_bot)

    valid_3d = (np.isfinite(pts_top).all(1) & np.isfinite(pts_bot).all(1) &
                (np.linalg.norm(pts_top, axis=1) < 1e5) &
                (np.linalg.norm(pts_bot, axis=1) < 1e5))

    pts_top = pts_top[valid_3d]
    pts_bot = pts_bot[valid_3d]
    e_xs = edge_xs[valid_3d]
    e_y_tops = edge_y_tops[valid_3d]
    e_y_bots = edge_y_bots[valid_3d]

    if len(pts_top) < min_valid_cols:
        return None, 0, {}, f"M:invalid_3d_pts({int(valid_3d.sum())})"

    # -- Fit ground plane --
    ground_pts = gather_ground_points(world_points, seg_map,
                                      id_sidewalk, id_road)
    if ground_pts is None:
        return None, 0, {}, "M:no_ground_pts"

    plane_result = fit_ground_plane(ground_pts)
    if plane_result is None:
        return None, 0, {}, "M:plane_fit_fail"
    plane_n, plane_d = plane_result

    # -- Scale calibration --
    cam_center = pose_enc[:3].astype(np.float64)
    h_pred = abs(float(np.dot(plane_n, cam_center) + plane_d))
    if h_pred < 1e-6:
        return None, 0, {}, "M:cam_on_plane"
    scale = cam_height / h_pred

    if scale < 0.2 or scale > 50:
        return None, 0, {}, f"M:bad_scale({scale:.1f})"

    # -- Project edge points to ground plane, compute widths --
    pts_top_proj = project_to_plane(pts_top.astype(np.float64),
                                    plane_n, plane_d) * scale
    pts_bot_proj = project_to_plane(pts_bot.astype(np.float64),
                                    plane_n, plane_d) * scale

    # Use PCA to find the across-sidewalk direction for accurate width
    all_proj = np.vstack([pts_top_proj, pts_bot_proj])
    mean_p = all_proj.mean(0)
    centered = all_proj - mean_p
    # Remove residual normal component
    centered = centered - (centered @ plane_n[:, None]) * plane_n[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    along_dir = vh[0]
    along_dir /= np.linalg.norm(along_dir) + 1e-12
    across_dir = np.cross(plane_n, along_dir)
    across_dir /= np.linalg.norm(across_dir) + 1e-12

    diffs = pts_top_proj - pts_bot_proj
    widths_pca = np.abs(diffs @ across_dir)
    widths_raw = np.linalg.norm(diffs, axis=1)

    # Use PCA width if it's consistent with raw width; otherwise fallback
    ratio = np.median(widths_pca) / (np.median(widths_raw) + 1e-9)
    widths = widths_pca if 0.5 < ratio < 1.05 else widths_raw

    # Filter unreasonable per-column widths
    reasonable = (widths > 0.05) & (widths < 15.0) & np.isfinite(widths)
    widths = widths[reasonable]
    if len(widths) < 3:
        return None, 0, {}, f"M:unreasonable_widths({int(reasonable.sum())})"

    # -- Robust aggregation: MAD filtering --
    med_w = np.median(widths)
    mad_w = np.median(np.abs(widths - med_w)) * 1.4826 + 1e-6
    inliers = np.abs(widths - med_w) < 2.5 * mad_w
    filtered = widths[inliers]

    if len(filtered) < 3:
        return None, 0, {}, f"M:too_few_inliers({int(inliers.sum())})"

    width = float(np.median(filtered))
    width_std = float(np.std(filtered))

    # Physical range check
    if not (width_range[0] <= width <= width_range[1]):
        return None, 0, {}, f"M:out_of_range({width:.2f}m)"

    # Width consistency check (coefficient of variation)
    width_cv = width_std / (width + 1e-9)
    if width_cv > max_width_cv:
        return None, 0, {}, f"M:width_inconsistent(cv={width_cv:.2f})"

    # Geometric cross-validation: compare 3D width with pixel-geometry estimate
    fov_h = float(pose_enc[7])
    if fov_h > 0.05:
        f_y_img = (H_img / 2.0) / (np.tan(fov_h / 2.0) + 1e-9)
        y_horizon = H_img / 2.0
        med_y_top = float(np.median(e_y_tops))
        med_y_bot = float(np.median(e_y_bots))
        dy_top = med_y_top - y_horizon
        dy_bot = med_y_bot - y_horizon

        if dy_top > 5 and dy_bot > dy_top:
            d_in = f_y_img * cam_height / dy_top
            d_out = f_y_img * cam_height / dy_bot
            w_geom = abs(d_in - d_out)
            if w_geom > 0.01:
                geom_ratio = width / w_geom
                if geom_ratio > 3.0 or geom_ratio < 0.33:
                    return None, 0, {}, (
                        f"M:geom_mismatch(3d={width:.2f},"
                        f"geom={w_geom:.2f},ratio={geom_ratio:.2f})")

    info = {
        'y_top_med': float(np.median(e_y_tops)),
        'y_bot_med': float(np.median(e_y_bots)),
        'x_start': int(x_start),
        'x_end': int(x_end),
        'scale': float(scale),
        'n_cols': int(len(filtered)),
        'width_std': width_std,
        'width_cv': width_cv,
        'overseg_corrected': overseg_corrected,
    }

    if collect_geometry:
        pts_top_r = pts_top_proj[reasonable]
        pts_bot_r = pts_bot_proj[reasonable]
        pts_top_f = pts_top_r[inliers]
        pts_bot_f = pts_bot_r[inliers]
        rng = np.random.default_rng(0)
        gs = (ground_pts * scale).astype(np.float64)
        if len(gs) > 8000:
            gs = gs[rng.choice(len(gs), 8000, replace=False)]
        info.update({
            'geom_ground_pts': gs,
            'geom_pts_top': pts_top_f.astype(np.float64),
            'geom_pts_bot': pts_bot_f.astype(np.float64),
            'geom_plane_n': plane_n.astype(np.float64),
            'geom_plane_d_scaled': float(plane_d * scale),
            'geom_across_dir': across_dir.astype(np.float64),
            'geom_along_dir': along_dir.astype(np.float64),
            'geom_cam_center': (cam_center * scale).astype(np.float64),
            'geom_width_m': float(width),
        })

    return width, len(filtered), info, ""


# ============================================================
# Visualization (optional)
# ============================================================

def visualize_measurement(img_bgr: np.ndarray,
                          info: dict,
                          width_m: Optional[float],
                          out_path: str) -> None:
    """Draw midline band and inner/outer edge lines on the image."""
    overlay = img_bgr.copy()
    H, W = overlay.shape[:2]
    x_start = info.get('x_start', 0)
    x_end = info.get('x_end', W)

    # Draw midline band
    band = overlay.copy()
    cv2.rectangle(band, (x_start, 0), (x_end, H - 1),
                  (255, 255, 0), thickness=-1)
    overlay = cv2.addWeighted(overlay, 0.7, band, 0.3, 0.0)

    # Draw inner edge (yellow) and outer edge (red)
    y_top = int(info.get('y_top_med', 0))
    y_bot = int(info.get('y_bot_med', 0))
    if y_top > 0:
        cv2.line(overlay, (x_start, y_top), (x_end, y_top),
                 (0, 255, 255), 2)
    if y_bot > 0:
        cv2.line(overlay, (x_start, y_bot), (x_end, y_bot),
                 (0, 0, 255), 2)

    # Text overlay
    if width_m is not None:
        text = f"Width: {width_m:.3f} m  (cols={info.get('n_cols', 0)})"
        cv2.putText(overlay, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, overlay)


def _axes_3d_equal(ax, centers: np.ndarray, span: float) -> None:
    """Roughly equal aspect for 3D axes around centers (N,3) or (3,)."""
    c = np.asarray(centers).reshape(-1, 3).mean(0)
    r = float(max(span, 0.5))
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _plane_grid(n: np.ndarray, d_scaled: float, center: np.ndarray,
                extent: float = 5.0, ngrid: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample grid on plane n·x + d_scaled = 0 for wireframe plot."""
    n = np.asarray(n, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, n))) > 0.85:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2) + 1e-12
    dist = float(np.dot(n, center) + d_scaled)
    x0 = center - dist * n
    uu = np.linspace(-extent, extent, ngrid)
    vv = np.linspace(-extent, extent, ngrid)
    U, V = np.meshgrid(uu, vv, indexing='xy')
    X = x0[0] + U * t1[0] + V * t2[0]
    Y = x0[1] + U * t1[1] + V * t2[1]
    Z = x0[2] + U * t1[2] + V * t2[2]
    return X, Y, Z


def save_geometry_plots(geom: dict, out_prefix: str) -> None:
    """
    Write two PNGs explaining the 3D reasoning:
      1) Scene: subsampled road/sidewalk points, fitted ground plane, camera,
         inner/outer edge samples (metric space after height calibration).
      2) Width: 3D chords from inner to outer edge per column; PCA across-dir.
    """
    need = ('geom_pts_top', 'geom_pts_bot', 'geom_plane_n', 'geom_plane_d_scaled',
            'geom_across_dir', 'geom_cam_center', 'geom_ground_pts', 'geom_width_m')
    if not all(k in geom for k in need):
        return

    top = geom['geom_pts_top']
    bot = geom['geom_pts_bot']
    n_pl = geom['geom_plane_n']
    d_pl = geom['geom_plane_d_scaled']
    across = geom['geom_across_dir']
    cam = geom['geom_cam_center']
    ground = geom['geom_ground_pts']
    w_m = geom['geom_width_m']

    all_pts = np.vstack([ground, top, bot, cam.reshape(1, 3)])
    center = all_pts.mean(0)
    span = float(np.ptp(all_pts, axis=0).max()) * 0.65 + 1.0

    # --- Figure 1: scene context ---
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ground[:, 0], ground[:, 1], ground[:, 2],
               c='#6b6b6b', s=2, alpha=0.35, linewidths=0, label='Ground samples (road+sw)')
    ax.scatter(top[:, 0], top[:, 1], top[:, 2],
               c='#2ca02c', s=14, alpha=0.9, label='Inner edge (3D, inliers)')
    ax.scatter(bot[:, 0], bot[:, 1], bot[:, 2],
               c='#d62728', s=14, alpha=0.9, label='Outer edge (3D, inliers)')
    ax.scatter([cam[0]], [cam[1]], [cam[2]], c='#1f77b4', s=80, marker='^',
               label='Camera (scaled)', depthshade=False)

    Xg, Yg, Zg = _plane_grid(n_pl, d_pl, center, extent=span * 0.9, ngrid=12)
    ax.plot_wireframe(Xg, Yg, Zg, color='#9467bd', linewidth=0.4, alpha=0.55,
                      label='Fitted ground plane')

    ax.set_title(
        f'VGGT metric scene  |  median width = {w_m:.2f} m  |  n cols = {len(top)}',
        fontsize=11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    _axes_3d_equal(ax, all_pts, span)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path1 = f"{out_prefix}_scene3d.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved 3D scene plot: {path1}")

    # --- Figure 2: width chords + across direction ---
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection='3d')
    centroid = 0.5 * (top + bot)
    c0 = centroid.mean(0)

    for i in range(len(top)):
        ax.plot(
            [top[i, 0], bot[i, 0]],
            [top[i, 1], bot[i, 1]],
            [top[i, 2], bot[i, 2]],
            c='#17becf', linewidth=1.0, alpha=0.65)
    ax.scatter(top[:, 0], top[:, 1], top[:, 2], c='#2ca02c', s=18, zorder=5)
    ax.scatter(bot[:, 0], bot[:, 1], bot[:, 2], c='#d62728', s=18, zorder=5)

    ac = across / (np.linalg.norm(across) + 1e-12)
    ax.quiver(
        c0[0], c0[1], c0[2],
        ac[0], ac[1], ac[2],
        length=w_m, normalize=False, color='#ff7f0e', linewidth=2.0,
        arrow_length_ratio=0.12, label='PCA across-sidewalk (width direction)')

    ax.set_title(
        'Per-column 3D chords (SegFormer edges → VGGT depths) '
        f'| median ‖proj‖ ≈ {w_m:.2f} m',
        fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    chord_pts = np.vstack([top, bot])
    span2 = float(np.ptp(chord_pts, axis=0).max()) * 0.85 + 0.5
    _axes_3d_equal(ax, chord_pts, span2)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path2 = f"{out_prefix}_width3d.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved 3D width plot: {path2}")


# ============================================================
# Ground truth & evaluation
# ============================================================

def load_ground_truth(csv_path: str) -> Dict[int, float]:
    """Load ground truth widths from submission_result.csv."""
    gt = {}
    if not csv_path or not os.path.isfile(csv_path):
        return gt
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gt[int(row['id'])] = float(row['central_width'])
            except (ValueError, KeyError):
                continue
    return gt


def print_metrics(results: Dict[int, float],
                  gt: Dict[int, float]) -> None:
    """Compute and print evaluation metrics."""
    errors = []
    abs_errors = []
    matched = []

    for img_id, pred in sorted(results.items()):
        if img_id in gt:
            err = pred - gt[img_id]
            errors.append(err)
            abs_errors.append(abs(err))
            matched.append((img_id, pred, gt[img_id], err))

    if not abs_errors:
        print("[WARN] No images matched ground truth for evaluation.")
        return

    errors = np.array(errors)
    abs_errors = np.array(abs_errors)

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Images measured:        {len(results)}")
    print(f"  Images with GT match:   {len(abs_errors)}")
    print(f"  MAE:                    {np.mean(abs_errors):.4f} m")
    print(f"  RMSE:                   {np.sqrt(np.mean(errors ** 2)):.4f} m")
    print(f"  Median AE:              {np.median(abs_errors):.4f} m")
    print(f"  Mean signed error:      {np.mean(errors):+.4f} m")
    print(f"  Std of error:           {np.std(errors):.4f} m")
    print(f"  % within 0.25m:         {100 * np.mean(abs_errors < 0.25):.1f}%")
    print(f"  % within 0.50m:         {100 * np.mean(abs_errors < 0.50):.1f}%")
    print(f"{'=' * 60}")

    # Show worst cases
    matched.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"\n  Top 10 worst predictions:")
    print(f"  {'ID':>5}  {'Pred':>7}  {'GT':>7}  {'Error':>8}")
    for img_id, pred, gt_w, err in matched[:10]:
        print(f"  {img_id:>5}  {pred:>7.3f}  {gt_w:>7.3f}  {err:>+8.3f}")
    print()


# ============================================================
# Main
# ============================================================

def main(args):
    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")
    amp_ctx = autocast_context(device)
    use_non_blocking = device == "cuda"

    # 1. Find images
    if args.image:
        img_path = os.path.abspath(os.path.expanduser(args.image))
        if not os.path.isfile(img_path):
            print(f"[ERROR] Image not found: {img_path}")
            return
        image_paths = [img_path]
        print(f"[INFO] Single-image mode: {img_path}")
    else:
        image_paths = sorted(
            glob.glob(os.path.join(args.img_dir, "*.jpg")) +
            glob.glob(os.path.join(args.img_dir, "*.jpeg")) +
            glob.glob(os.path.join(args.img_dir, "*.png")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            if os.path.splitext(os.path.basename(p))[0].isdigit() else p
        )
        if not image_paths:
            print("[ERROR] No images found.")
            return
        print(f"[INFO] Found {len(image_paths)} images in {args.img_dir}")

    # 2. Load ground truth
    gt = load_ground_truth(args.gt_csv)
    if gt:
        print(f"[INFO] Loaded {len(gt)} ground truth entries from {args.gt_csv}")
    elif os.path.isfile(args.gt_csv):
        print(f"[WARN] No GT rows parsed from {args.gt_csv}")
    else:
        print(f"[WARN] GT file missing or not set: {args.gt_csv} (evaluation metrics skipped)")

    # 3. Load VGGT
    vggt_weights = (args.vggt_weights or os.environ.get("VGGT_WEIGHTS") or "").strip()
    vggt_weights = vggt_weights or None
    vggt_model = load_vggt(vggt_weights, device)
    vggt_model.eval()

    # 4. Load SegFormer
    print("[INFO] Loading SegFormer...")
    seg_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    processor = AutoImageProcessor.from_pretrained(seg_id)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        seg_id, revision="refs/pr/3", use_safetensors=True
    ).to(device)
    seg_model.eval()

    id2label = {int(k): v for k, v in seg_model.config.id2label.items()}
    label2id = {v.lower(): k for k, v in id2label.items()}
    ID_SIDEWALK = label2id["sidewalk"]
    ID_ROAD = label2id["road"]
    print(f"[INFO] sidewalk_id={ID_SIDEWALK}, road_id={ID_ROAD}")

    # 5. Process each image
    results = {}
    skip_reasons = Counter()   # detailed reason -> count
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # For skip report CSV
    skip_records = []  # (img_id, reason)

    for path in tqdm(image_paths, desc="Processing"):
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.image_id is not None:
            img_id = args.image_id
        elif stem.isdigit():
            img_id = int(stem)
        elif args.image:
            img_id = 0
        else:
            continue

        try:
            # -- VGGT inference --
            img_tensor = load_and_preprocess_images([path]).to(
                device, non_blocking=use_non_blocking)
            with torch.inference_mode(), amp_ctx:
                pred = vggt_model(img_tensor)

            wp = pred["world_points"]
            if wp.dim() == 5:
                wp = wp[0]
            wp = wp[0].detach().cpu().numpy()  # [H_d, W_d, 3]

            pose_enc = None
            if "pose_enc" in pred and pred["pose_enc"] is not None:
                pe = pred["pose_enc"]
                if pe.dim() == 3:
                    pe = pe[0]
                pose_enc = pe[0].detach().cpu().numpy()

            del pred, img_tensor
            maybe_empty_cache(device)

            if pose_enc is None:
                skip_reasons["E:no_pose_enc"] += 1
                skip_records.append((img_id, "E:no_pose_enc"))
                continue

            # -- SegFormer inference --
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                skip_reasons["E:imread_fail"] += 1
                skip_records.append((img_id, "E:imread_fail"))
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            with torch.inference_mode():
                inputs = processor(images=[img_rgb],
                                   return_tensors="pt").to(device)
                with amp_ctx:
                    outputs = seg_model(**inputs)
                seg_list = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[img_rgb.shape[:2]]
                )
                seg_map = seg_list[0].cpu().numpy().astype(np.int32)
                del inputs, outputs
                maybe_empty_cache(device)

            # -- Quality check --
            ok, reason = check_quality(seg_map, ID_SIDEWALK, ID_ROAD,
                                       min_sw_frac=args.min_sw_frac,
                                       band_frac=args.band_frac)
            if not ok:
                skip_reasons[reason] += 1
                skip_records.append((img_id, reason))
                if args.verbose:
                    tqdm.write(f"  [SKIP] {stem}: {reason}")
                continue

            # -- Measure width --
            width, n_cols, info, fail_reason = measure_width_columnwise(
                wp, seg_map, pose_enc, args.cam_height,
                ID_SIDEWALK, ID_ROAD,
                img_gray=img_gray,
                band_frac=args.band_frac,
                min_run_px=args.min_run_px,
                min_valid_cols=args.min_valid_cols,
                max_phys_width=args.max_phys_width,
                max_width_cv=args.max_width_cv,
                collect_geometry=args.plot3d,
            )

            if width is not None:
                results[img_id] = width
                gt_w = gt.get(img_id)
                gt_str = (f"  GT={gt_w:.3f}m  err={width - gt_w:+.3f}m"
                          if gt_w else "")
                if args.verbose:
                    tqdm.write(
                        f"  {stem}: {width:.3f}m (cols={n_cols}){gt_str}")

                if args.plot3d:
                    geom_keys = [k for k in list(info.keys())
                                 if k.startswith('geom_')]
                    if geom_keys:
                        geom = {k: info.pop(k) for k in geom_keys}
                        prefix = os.path.join(out_dir, f"geom3d_{stem}")
                        save_geometry_plots(geom, prefix)

                if args.visualize:
                    vis_path = os.path.join(out_dir, f"vis_{stem}.png")
                    visualize_measurement(img_bgr, info, width, vis_path)
            else:
                skip_reasons[fail_reason] += 1
                skip_records.append((img_id, fail_reason))
                if args.verbose:
                    tqdm.write(f"  {stem}: {fail_reason}")

        except Exception as e:
            skip_reasons[f"E:exception"] += 1
            skip_records.append((img_id, f"E:exception({e})"))
            tqdm.write(f"  [ERROR] {stem}: {e}")
            continue

    # 6. Save results CSV
    csv_path = os.path.join(out_dir, "predicted_widths.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "predicted_width", "ground_truth", "abs_error"])
        for img_id in sorted(results.keys()):
            pred = results[img_id]
            gt_w = gt.get(img_id)
            ae = abs(pred - gt_w) if gt_w is not None else ""
            writer.writerow([
                img_id,
                f"{pred:.4f}",
                f"{gt_w:.4f}" if gt_w is not None else "",
                f"{ae:.4f}" if isinstance(ae, float) else "",
            ])
    print(f"\n[INFO] Saved predictions to {csv_path}")

    # 7. Save skip report
    skip_csv = os.path.join(out_dir, "skip_report.csv")
    with open(skip_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "reason"])
        for img_id, reason in sorted(skip_records):
            writer.writerow([img_id, reason])
    print(f"[INFO] Saved skip report to {skip_csv}")

    # 8. Summary
    n_quality = sum(v for k, v in skip_reasons.items() if k.startswith("Q:"))
    n_measure = sum(v for k, v in skip_reasons.items() if k.startswith("M:"))
    n_error = sum(v for k, v in skip_reasons.items() if k.startswith("E:"))

    print(f"\n[INFO] Summary:")
    print(f"  Total images:       {len(image_paths)}")
    print(f"  Measured:           {len(results)}")
    print(f"  Skipped (quality):  {n_quality}")
    print(f"  Skipped (measure):  {n_measure}")
    print(f"  Skipped (error):    {n_error}")

    print(f"\n[INFO] Skip reason breakdown:")
    for reason, count in skip_reasons.most_common():
        print(f"  {reason:40s} {count:>5}")

    # 9. Evaluate
    print_metrics(results, gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sidewalk width measurement & evaluation "
                    "using VGGT + SegFormer")
    parser.add_argument(
        "--vggt_weights",
        type=str,
        default=None,
        help="Path to local VGGT model.pt (no Hugging Face download for VGGT). "
             "If unset, uses env VGGT_WEIGHTS; if both unset, loads facebook/VGGT-1B.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Compute device: auto prefers CUDA, then MPS (Apple Silicon), then CPU.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Process one image file (any filename). Overrides --img_dir.",
    )
    parser.add_argument(
        "--image_id",
        type=int,
        default=None,
        help="Numeric id for CSV output when using --image (default: stem if "
             "numeric, else 0).",
    )
    parser.add_argument("--img_dir", type=str, default="streetview_out",
                        help="Directory containing street view images")
    parser.add_argument("--gt_csv", type=str, default="submission_result.csv",
                        help="Ground truth CSV; optional if missing (no metrics).")
    parser.add_argument("--out_dir", type=str, default="eval_out",
                        help="Output directory for results and visualizations")
    parser.add_argument("--cam_height", type=float, default=2.5,
                        help="Camera height in meters (GSV default 2.5m)")
    parser.add_argument("--band_frac", type=float, default=0.20,
                        help="Midline band width as fraction of image width")
    parser.add_argument("--min_run_px", type=int, default=15,
                        help="Min sidewalk run height in pixels per column")
    parser.add_argument("--min_valid_cols", type=int, default=10,
                        help="Min valid columns required for measurement")
    parser.add_argument("--min_sw_frac", type=float, default=0.02,
                        help="Min sidewalk area fraction for quality check")
    parser.add_argument("--max_phys_width", type=float, default=5.0,
                        help="Max physical sidewalk width (m) for geometric "
                             "over-segmentation detection")
    parser.add_argument("--max_width_cv", type=float, default=0.40,
                        help="Max coefficient of variation of per-column "
                             "widths (consistency check)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save 2D overlay (midline band + edges)")
    parser.add_argument(
        "--plot3d",
        action="store_true",
        help="Save two 3D PNGs (scene + plane + camera; per-column width chords).",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-image results")

    args = parser.parse_args()
    main(args)
