"""
Per-image processing orchestration.

``process_street_image`` runs the full analysis pipeline for a
point_type="street" image:

  1. Load image + masks from GCS
  2. Find per-row edges in perspective space (with border extrapolation)
  3. Compute metric widths via ray-casting
  4. Rectify sidewalk strip (vanishing-point-based)
  5. Warp obstacles into rectified space, estimate footprints
  6. Generate visualizations, upload results

``process_junction_image`` writes a stub JSON for future junction
analysis (crosswalk detection, ramp presence, etc.).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from gcs_utils import GCSClient, bytes_to_image, bytes_to_mask
from geometry import (
    build_intrinsic_matrix,
    compute_metric_widths,
    compute_obstacle_ground_footprint,
    infill_rectified,
    rectify_sidewalk,
)
from edge_detection import fill_small_gaps, find_row_edges
from obstacles import (
    assign_obstacles_to_sidewalks,
    compute_obstacle_footprint_metadata,
    compute_usable_width_cm,
    estimate_width_footprint,
)
from visualization import (
    render_obstacle_overlay,
    render_rectified_footprint,
    render_width_overlay,
    render_width_profile,
)


def _log(msg: str) -> None:
    print(f"  [PIPE] {msg}")


def _load_masks_from_gcs(
    gcs: GCSClient,
    masks_prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Discover and download all masks under a GCS prefix.

    Expected GCS layout::

        {masks_prefix}/{detection_index}/{class_name}/mask_NNN.png

    Returns (sidewalk_masks, obstacle_masks).
    """
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


def _width_stats(
    widths_cm: np.ndarray,
    iqr_factor: float,
) -> dict[str, Any]:
    """Compute width statistics with IQR outlier rejection."""
    vals = widths_cm[~np.isnan(widths_cm)]
    if len(vals) == 0:
        return {"median_cm": None, "mean_cm": None, "std_cm": None,
                "n_valid_rows": 0, "n_outliers": 0}

    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo = q1 - iqr_factor * iqr
    hi = q3 + iqr_factor * iqr
    inlier = vals[(vals >= lo) & (vals <= hi)]
    n_outliers = len(vals) - len(inlier)

    return {
        "median_cm": round(float(np.median(inlier)), 1) if len(inlier) else None,
        "mean_cm": round(float(np.mean(inlier)), 1) if len(inlier) else None,
        "std_cm": round(float(np.std(inlier)), 1) if len(inlier) else None,
        "q1_cm": round(float(q1), 1),
        "q3_cm": round(float(q3), 1),
        "n_valid_rows": int(len(vals)),
        "n_inlier_rows": int(len(inlier)),
        "n_outliers": int(n_outliers),
    }


def process_street_image(
    gcs: GCSClient,
    image_blob: str,
    masks_prefix: str,
    output_prefix: str,
    cfg: dict[str, Any],
    local_output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Full analysis pipeline for a street-type sample point.

    Uses perspective-space analysis with ray-cast metric widths and
    v1-style rectification for visualization.
    """
    cam_cfg = cfg["camera"]
    sw_cfg = cfg["sidewalk"]
    fp_cfg = cfg["obstacle_footprint"]
    we_cfg = cfg["width_estimation"]
    obstacle_is_tree = cfg["obstacle"].get("is_tree", ["tree"])

    # 1. Load image and masks
    _log(f"Loading image: {image_blob}")
    _log(f"Searching masks under: {masks_prefix}")
    image_bytes = gcs.download_as_bytes(image_blob)
    original_image = bytes_to_image(image_bytes)
    img_h, img_w = original_image.shape[:2]

    sidewalk_masks, obstacle_masks = _load_masks_from_gcs(gcs, masks_prefix)

    if not sidewalk_masks:
        _log(f"No sidewalk masks found under '{masks_prefix}', skipping.")
        return {"status": "no_sidewalk_mask", "image": image_blob}

    _log(f"Found {len(sidewalk_masks)} sidewalk mask(s), "
         f"{len(obstacle_masks)} obstacle mask(s).")

    # 2. Camera intrinsics (shared across all masks)
    K = build_intrinsic_matrix(img_w, img_h, cam_cfg["hfov_deg"])

    # 3. Assign obstacles to sidewalks in original image space
    assignment = assign_obstacles_to_sidewalks(sidewalk_masks, obstacle_masks)

    # 4. Process each sidewalk segment
    all_results = {}

    for sw_key, sw_mask in sidewalk_masks.items():
        _log(f"Processing sidewalk segment: {sw_key}")

        # 4a. Gap-fill and find edges
        filled = fill_small_gaps(sw_mask, sw_cfg["gap_fill_px"])
        left_e, right_e, valid, extrap = find_row_edges(filled)

        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 2:
            _log(f"  {sw_key}: too few valid rows ({len(valid_idx)}), skipping.")
            all_results[sw_key] = {"status": "too_few_valid_rows"}
            continue

        # 4b. Filter rows: only below horizon + min_horizon_dist
        cy = img_h / 2.0
        below_horizon = valid_idx[valid_idx > cy + we_cfg["min_horizon_dist"]]

        if we_cfg.get("exclude_extrapolated", False):
            below_horizon = below_horizon[~extrap[below_horizon]]

        if len(below_horizon) < 2:
            _log(f"  {sw_key}: too few rows below horizon, skipping.")
            all_results[sw_key] = {"status": "insufficient_below_horizon"}
            continue

        # 4c. Compute metric widths via ray-casting
        widths_cm = compute_metric_widths(
            left_e, right_e, valid,
            K=K,
            pitch_deg=cam_cfg["pitch_deg"],
            camera_height_m=cam_cfg["height_m"],
            min_horizon_dist=we_cfg["min_horizon_dist"],
        )

        # Also mask out extrapolated rows if configured
        if we_cfg.get("exclude_extrapolated", False):
            widths_cm[extrap] = np.nan

        stats = _width_stats(widths_cm, sw_cfg["iqr_factor"])

        if stats["n_valid_rows"] == 0:
            _log(f"  {sw_key}: no valid width measurements.")
            all_results[sw_key] = {"status": "no_valid_widths"}
            continue

        _log(f"  {sw_key}: median width = {stats['median_cm']} cm "
             f"({stats['n_valid_rows']} rows, {stats['n_outliers']} outliers)")

        # 4d. Rectify sidewalk strip
        rect_result = rectify_sidewalk(
            filled, left_e, right_e, valid, is_mask=True,
        )
        if rect_result is None:
            _log(f"  {sw_key}: rectification failed.")
            all_results[sw_key] = {"status": "rectification_failed",
                                   "width_stats": stats}
            continue

        sw_rect, target_w, pad = rect_result
        sw_rect = infill_rectified(sw_rect, is_mask=True)

        # Rectify original image with same parameters
        img_rect_result = rectify_sidewalk(
            original_image, left_e, right_e, valid,
            target_width=target_w, is_mask=False,
        )
        img_rect = None
        if img_rect_result is not None:
            img_rect = infill_rectified(img_rect_result[0], is_mask=False)

        # 4e. Rectify obstacle masks and estimate footprints
        assigned_obs_keys = assignment.get(sw_key, [])
        assigned_obs = {k: obstacle_masks[k] for k in assigned_obs_keys
                        if k in obstacle_masks}

        obs_full_rect: dict[str, np.ndarray] = {}
        obs_fp_rect: dict[str, np.ndarray] = {}
        all_footprint_meta = []

        for obs_key, obs_mask in assigned_obs.items():
            rect_obs_result = rectify_sidewalk(
                obs_mask, left_e, right_e, valid,
                target_width=target_w, is_mask=True,
            )
            if rect_obs_result is None:
                continue
            full_r = rect_obs_result[0]
            obs_full_rect[obs_key] = full_r

            is_tree = any(t in obs_key.lower() for t in obstacle_is_tree)
            fp = estimate_width_footprint(
                full_r, is_tree=is_tree,
                base_scan_ratio=fp_cfg["base_scan_ratio"],
                trunk_scan_ratio=fp_cfg["trunk_scan_ratio"],
                aspect_ratio=fp_cfg["aspect_ratio"],
                max_height=fp_cfg["max_height"],
            )
            obs_fp_rect[obs_key] = fp

            # Ray-cast metric footprint from original mask
            rc_footprints = compute_obstacle_ground_footprint(
                obs_mask, K,
                pitch_deg=cam_cfg["pitch_deg"],
                camera_height_m=cam_cfg["height_m"],
                base_scan_ratio=fp_cfg["base_scan_ratio"],
            )

            meta = compute_obstacle_footprint_metadata(
                fp, obs_key, target_w, pad,
                raycast_footprints=rc_footprints,
            )
            all_footprint_meta.extend(meta)

        # 4f. Compute usable width
        usable_cm = compute_usable_width_cm(
            widths_cm, left_e, right_e, valid,
            obs_full_rect, target_w, pad,
        )
        usable_stats = _width_stats(usable_cm, sw_cfg["iqr_factor"])

        # 5. Generate visualizations
        _log(f"  {sw_key}: generating visualizations...")

        obs_overlay_bytes = render_obstacle_overlay(
            original_image, sw_mask,
            {k: obstacle_masks[k] for k in assigned_obs_keys
             if k in obstacle_masks},
        )

        rect_fp_bytes = render_rectified_footprint(
            sw_rect, obs_full_rect, obs_fp_rect,
            target_w, pad, img_rect,
        )

        width_overlay_bytes = render_width_overlay(
            original_image, sw_mask,
            widths_cm, left_e, right_e, valid,
        )

        width_profile_bytes = render_width_profile(
            widths_cm, sw_cfg["iqr_factor"],
        )

        # 6. Upload
        seg_prefix = f"{output_prefix}/{sw_key}"

        artifacts = {
            "obstacle_overlay.png": obs_overlay_bytes,
            "rectified_footprint.png": rect_fp_bytes,
            "width_overlay.png": width_overlay_bytes,
            "width_profile.png": width_profile_bytes,
        }

        for name, data in artifacts.items():
            blob = f"{seg_prefix}/{name}"
            gcs.upload_bytes(data, blob, content_type="image/png")

        analysis = {
            "sidewalk_key": sw_key,
            "image_blob": image_blob,
            "width_stats": stats,
            "usable_width_stats": usable_stats,
            "obstacle_footprints": all_footprint_meta,
            "n_obstacles": len(all_footprint_meta),
            "image_size": [img_w, img_h],
            "camera": cam_cfg,
        }

        analysis_json = json.dumps(analysis, indent=2, default=str)
        gcs.upload_bytes(
            analysis_json.encode(),
            f"{seg_prefix}/analysis.json",
            content_type="application/json",
        )

        if local_output_dir:
            local_seg = os.path.join(local_output_dir, sw_key)
            os.makedirs(local_seg, exist_ok=True)
            for name, data in artifacts.items():
                with open(os.path.join(local_seg, name), "wb") as f:
                    f.write(data)
            with open(os.path.join(local_seg, "analysis.json"), "w") as f:
                f.write(analysis_json)

        all_results[sw_key] = {
            "status": "ok",
            "width_stats": stats,
            "usable_width_stats": usable_stats,
            "n_obstacles": len(all_footprint_meta),
        }

    return {
        "status": "ok",
        "image": image_blob,
        "segments": all_results,
    }


# ── Junction analysis interface ─────────────────────────────────────────────

class JunctionAnalyzer(ABC):
    """
    Abstract base class for future junction-specific analysis.

    Subclass this to implement crosswalk detection, ramp presence
    detection, or other junction accessibility analysis.
    """

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
    """
    Handle a junction-type sample point.

    Currently writes a stub ``junction_info.json`` with paths and metadata.
    If an ``analyzer`` is provided, it is invoked for full analysis.
    """
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
