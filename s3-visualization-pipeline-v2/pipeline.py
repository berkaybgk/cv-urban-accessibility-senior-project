"""
Per-image processing — v1-style geometry and outputs, v2 GCS layout + filenames.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from gcs_utils import GCSClient, bytes_to_image, bytes_to_mask
from geometry import rectify_sidewalk
from edge_detection import find_row_edges
from obstacles import (
    assign_obstacles_to_sidewalks,
    build_footprint_metadata,
    build_width_metadata,
    estimate_width_footprint,
)
from visualization import (
    render_obstacle_silhouettes,
    render_rectified_footprint,
    render_width_overlay,
    render_width_profile,
)


def _log(msg: str) -> None:
    print(f"  [PIPE] {msg}")


def _make_color_palette(n: int) -> dict[int, tuple]:
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("tab10", max(n, 1))
    return {i: cmap(i)[:3] for i in range(n)}


def _load_masks_from_gcs(
    gcs: GCSClient,
    masks_prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    blobs = gcs.list_blobs(masks_prefix)
    png_blobs = [b for b in blobs if b.endswith(".png")]

    sidewalk_masks: dict[str, np.ndarray] = {}
    obstacle_masks: dict[str, np.ndarray] = {}

    for blob_name in png_blobs:
        parts = blob_name.split("/")
        if len(parts) < 3:
            continue
        class_name = parts[-2]
        detection_idx = parts[-3]
        mask_idx = parts[-1].replace("mask_", "").replace(".png", "")
        key = f"{class_name}_{detection_idx}_{mask_idx}"

        mask_bytes = gcs.download_as_bytes(blob_name)
        mask = bytes_to_mask(mask_bytes)

        if "sidewalk" in class_name.lower():
            sidewalk_masks[key] = mask
        else:
            obstacle_masks[key] = mask

    return sidewalk_masks, obstacle_masks


def process_street_image(
    gcs: GCSClient,
    image_blob: str,
    masks_prefix: str,
    output_prefix: str,
    cfg: dict[str, Any],
    local_output_dir: str | None = None,
) -> dict[str, Any]:
    fp_cfg = cfg["footprint"]
    cam_cfg = cfg["camera"]
    we_cfg = cfg["width_estimation"]
    obstacle_is_tree = set(fp_cfg["obstacle_is_tree"])

    _log(f"Loading image: {image_blob}")
    _log(f"Searching masks under: {masks_prefix}")
    original_image = bytes_to_image(gcs.download_as_bytes(image_blob))
    img_h, img_w = original_image.shape[:2]
    cy = img_h / 2.0

    sidewalk_masks, obstacle_masks = _load_masks_from_gcs(gcs, masks_prefix)

    if not sidewalk_masks:
        _log(f"No sidewalk masks found under '{masks_prefix}', skipping.")
        return {"status": "no_sidewalk_mask", "image": image_blob}

    _log(f"Found {len(sidewalk_masks)} sidewalk mask(s), "
         f"{len(obstacle_masks)} obstacle mask(s).")

    assignment = assign_obstacles_to_sidewalks(sidewalk_masks, obstacle_masks)
    all_results: dict[str, Any] = {}

    for sw_key, sw_mask in sidewalk_masks.items():
        _log(f"Processing sidewalk segment: {sw_key}")

        obs_masks = {
            k: obstacle_masks[k] for k in assignment.get(sw_key, [])
            if k in obstacle_masks
        }
        n_obs = len(obs_masks)
        palette = _make_color_palette(n_obs)
        obs_colors = {ot: palette[i] for i, ot in enumerate(obs_masks)}

        seg_prefix = f"{output_prefix}/{sw_key}"
        uploads: list[tuple[bytes, str, str]] = []

        sil_bytes = render_obstacle_silhouettes(
            original_image, sw_mask, obs_masks, obs_colors, sw_key)
        uploads.append((sil_bytes, f"{seg_prefix}/obstacle_silhouettes.png", "image/png"))

        left_e, right_e, valid_r, extrap_r = find_row_edges(sw_mask)
        if valid_r.sum() == 0:
            _log(f"  {sw_key}: no valid edges — skipping rectification & width")
            for data, blob, ct in uploads:
                gcs.upload_bytes(data, blob, content_type=ct)
            if local_output_dir:
                _write_local(local_output_dir, sw_key, uploads)
            all_results[sw_key] = {"status": "no_valid_edges"}
            continue

        rect = rectify_sidewalk(sw_mask, left_e, right_e, valid_r, is_mask=True)
        if rect is None:
            _log(f"  {sw_key}: rectification failed")
            for data, blob, ct in uploads:
                gcs.upload_bytes(data, blob, content_type=ct)
            if local_output_dir:
                _write_local(local_output_dir, sw_key, uploads)
            all_results[sw_key] = {"status": "rectification_failed"}
            continue

        sw_rect, target_w, pad = rect

        obs_full_rect: dict[str, np.ndarray] = {}
        obs_fp_rect: dict[str, np.ndarray] = {}
        for ot, m in obs_masks.items():
            full_r = rectify_sidewalk(
                m, left_e, right_e, valid_r, target_width=target_w, is_mask=True)
            if full_r is None:
                continue
            obs_full_rect[ot] = full_r[0]
            is_tree = any(t in ot for t in obstacle_is_tree)
            obs_fp_rect[ot] = estimate_width_footprint(
                obs_full_rect[ot], is_tree=is_tree,
                base_scan_ratio=fp_cfg["base_scan_ratio"],
                trunk_scan_ratio=fp_cfg["trunk_scan_ratio"],
                aspect_ratio=fp_cfg["aspect_ratio"],
                max_height=fp_cfg["max_height_px"],
            )

        fp_img_bytes = render_rectified_footprint(
            sw_rect, obs_full_rect, obs_fp_rect, obs_colors,
            obstacle_is_tree, target_w, pad, sw_key)
        uploads.append((fp_img_bytes, f"{seg_prefix}/rectified_footprint.png", "image/png"))

        fp_meta = build_footprint_metadata(
            sw_rect, obs_full_rect, obs_fp_rect,
            obstacle_is_tree, target_w, pad, sw_key)
        uploads.append((
            json.dumps(fp_meta, indent=2).encode(),
            f"{seg_prefix}/footprint_metadata.json",
            "application/json",
        ))

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
        width_stats: dict[str, Any] | None = None

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
                mn_val = float(width_cm_clean.min())
                mx_val = float(width_cm_clean.max())

                _log(f"  {sw_key}: width median={med:.1f} cm")

                overlay_bytes = render_width_overlay(
                    original_image, left_e, right_e,
                    valid_idx_clean, width_cm_clean,
                    med, std_val, cy, we_cfg["min_horizon_dist"], sw_key)
                uploads.append((overlay_bytes, f"{seg_prefix}/width_overlay.png", "image/png"))

                profile_bytes = render_width_profile(
                    valid_idx, width_cm, valid_idx_clean, width_cm_clean,
                    inlier, med, mean_val, std_val, lo_fence, hi_fence, sw_key)
                uploads.append((profile_bytes, f"{seg_prefix}/width_profile.png", "image/png"))

                width_meta = build_width_metadata(
                    sw_key, med, mean_val, std_val, mn_val, mx_val,
                    len(width_cm_clean), n_before_horizon,
                    n_dropped_horizon, n_dropped_extrap, n_outliers,
                    cam_cfg, we_cfg)
                uploads.append((
                    json.dumps(width_meta, indent=2).encode(),
                    f"{seg_prefix}/width_metadata.json",
                    "application/json",
                ))
                width_stats = {
                    "median_cm": med,
                    "mean_cm": mean_val,
                    "std_cm": std_val,
                    "n_valid_rows": len(width_cm_clean),
                }
            else:
                _log(f"  {sw_key}: no inlier rows after IQR")
        else:
            _log(f"  {sw_key}: no valid rows after horizon / extrap filter")

        for data, blob, ct in uploads:
            gcs.upload_bytes(data, blob, content_type=ct)

        if local_output_dir:
            _write_local(local_output_dir, sw_key, uploads)

        all_results[sw_key] = {
            "status": "ok",
            "footprint": fp_meta,
            "width": width_meta,
            "width_stats": width_stats,
            "n_obstacles": n_obs,
        }

    return {
        "status": "ok",
        "image": image_blob,
        "segments": all_results,
    }


def _write_local(
    local_output_dir: str,
    sw_key: str,
    uploads: list[tuple[bytes, str, str]],
) -> None:
    local_seg = os.path.join(local_output_dir, sw_key)
    os.makedirs(local_seg, exist_ok=True)
    for data, blob, _ in uploads:
        name = os.path.basename(blob)
        with open(os.path.join(local_seg, name), "wb") as f:
            f.write(data)


class JunctionAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        image: np.ndarray,
        sidewalk_masks: dict[str, np.ndarray],
        obstacle_masks: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        ...


def process_junction_image(
    gcs: GCSClient,
    image_blob: str,
    masks_prefix: str,
    output_prefix: str,
    metadata: dict[str, Any] | None = None,
    analyzer: JunctionAnalyzer | None = None,
) -> dict[str, Any]:
    _log("Junction point — writing stub info.")

    info: dict[str, Any] = {
        "status": "junction_stub",
        "image_blob": image_blob,
        "masks_prefix": masks_prefix,
        "metadata": metadata or {},
        "analysis": None,
    }

    if analyzer is not None:
        _log("Running junction analyzer...")
        image_bytes = gcs.download_as_bytes(image_blob)
        image = bytes_to_image(image_bytes)
        sw_masks, obs_masks = _load_masks_from_gcs(gcs, masks_prefix)
        info["analysis"] = analyzer.analyze(image, sw_masks, obs_masks, metadata or {})
        info["status"] = "junction_analyzed"

    info_json = json.dumps(info, indent=2, default=str)
    gcs.upload_bytes(
        info_json.encode(),
        f"{output_prefix}/junction_info.json",
        content_type="application/json",
    )

    return info
