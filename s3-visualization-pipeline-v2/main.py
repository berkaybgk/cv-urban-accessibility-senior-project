#!/usr/bin/env python3
"""
Sidewalk Analysis Pipeline v2
──────────────────────────────
Fetches pre-computed segmentation masks from GCS, runs perspective-space
analysis (ray-cast metric widths, per-row rectification, obstacle
footprints), generates visualizations, and uploads results back to GCS.

Key v2 improvements over v1:
  - Ray-cast metric width: pixel edges are projected to the ground plane
    via the camera model for accurate width measurement.
  - Per-row rectification with vanishing-point vertical scaling for
    recognisable "top-down strip" visualization.
  - Manifest-aware: reads the v2 sampler manifest to get point_type,
    segment_id, and skip junction points.
  - Segment-grouped batch processing with per-segment summaries.
  - Clean visualizations with legends below the image.

Usage
─────
  python main.py --config config.yaml
  python main.py                              # uses default config.yaml
  python main.py --image <blob_path>          # single image mode
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import load_config
from gcs_utils import GCSClient
from pipeline import process_junction_image, process_street_image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"


# ── Logging ─────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print("\n" + "-" * width)
    print(f"  {msg}")
    print("-" * width)


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


# ── Filename parsing (supports both v1 and v2 naming) ───────────────────────

_FILENAME_RE_V2 = re.compile(
    r"^(\d+)-(\d+)_(forward|backward)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)
_FILENAME_RE_V1 = re.compile(
    r"^(\d+)_(forward|backward|left|right)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)


def _parse_image_filename(filename: str) -> dict[str, str] | None:
    """Parse a sampler image filename into its components.

    Returns a dict with parsed fields including ``masks_folder`` which
    matches the GCS mask structure produced by the inference pipeline:
    ``{segment_id}_{direction}_{lat}``
    """
    name = Path(filename).name

    m = _FILENAME_RE_V2.match(name)
    if m:
        return {
            "segment_id": m.group(1),
            "point_id": m.group(2),
            "direction": m.group(3),
            "lat": m.group(4),
            "lon": m.group(5),
            "heading": m.group(6),
            "masks_folder": f"{m.group(1)}_{m.group(3)}_{m.group(4)}",
            "coordinate_folder": f"{m.group(2)}_{m.group(4)}_{m.group(5)}",
        }

    m = _FILENAME_RE_V1.match(name)
    if m:
        return {
            "segment_id": "0",
            "point_id": m.group(1),
            "direction": m.group(2),
            "lat": m.group(3),
            "lon": m.group(4),
            "heading": m.group(5),
            "masks_folder": f"{m.group(1)}_{m.group(3)}_{m.group(4)}/{m.group(2)}",
            "coordinate_folder": f"{m.group(1)}_{m.group(3)}_{m.group(4)}",
        }

    return None


def _extract_numeric_prefix(filename: str) -> int | None:
    m = re.match(r"(\d+)", Path(filename).name)
    return int(m.group(1)) if m else None


# ── Manifest loading ────────────────────────────────────────────────────────

def _load_manifest(gcs: GCSClient, manifest_blob: str) -> dict[str, dict]:
    """
    Load the v2 sampler manifest.csv from GCS.

    Returns a dict keyed by ``(segment_id, point_id, direction)`` with
    the full manifest row as value.  Returns empty dict on failure.
    """
    if not manifest_blob:
        return {}

    try:
        data = gcs.download_as_bytes(manifest_blob)
        text = data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        manifest = {}
        for row in reader:
            key = (
                row.get("segment_id", ""),
                row.get("point_id", ""),
                row.get("direction", ""),
            )
            manifest[key] = row
        _info(f"Loaded manifest with {len(manifest)} entries")
        return manifest
    except Exception as e:
        _info(f"Could not load manifest: {e}")
        return {}


def _get_point_type(
    parsed: dict[str, str],
    manifest: dict[str, dict],
) -> str:
    """
    Determine point_type from manifest if available, else default to "street".
    """
    key = (parsed["segment_id"], parsed["point_id"], parsed["direction"])
    row = manifest.get(key, {})
    return row.get("point_type", "street")


# ── Batch orchestration ─────────────────────────────────────────────────────

def run_batch(cfg: dict[str, Any]) -> None:
    gcs_cfg = cfg["gcs"]
    batch_cfg = cfg["batch"]

    gcs = GCSClient(
        project_id=gcs_cfg["project_id"],
        bucket_name=gcs_cfg["bucket_name"],
    )

    image_prefix = gcs_cfg["image_prefix"].rstrip("/") + "/"
    masks_prefix = gcs_cfg["masks_prefix"].rstrip("/")
    output_prefix = gcs_cfg["output_prefix"].rstrip("/")
    pmin, pmax = batch_cfg["prefix_min"], batch_cfg["prefix_max"]
    local_out = cfg.get("local_output_dir")

    _banner("Sidewalk Analysis Pipeline v2 -- Batch Mode")
    _info(f"Image prefix  : {image_prefix}")
    _info(f"Masks prefix  : {masks_prefix}")
    _info(f"Output prefix : {output_prefix}")
    _info(f"Prefix range  : {pmin} - {pmax}")

    manifest = _load_manifest(gcs, gcs_cfg.get("manifest_blob", ""))

    all_blobs = gcs.list_blobs(image_prefix)
    image_blobs = [b for b in all_blobs
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
    segment_results: dict[str, list[dict]] = defaultdict(list)
    n_junctions_skipped = 0
    t0 = time.time()

    for idx, (num, blob) in enumerate(filtered, 1):
        parsed = _parse_image_filename(blob)
        if parsed is None:
            _info(f"[{idx}/{len(filtered)}] Skipping (unrecognised name): {Path(blob).name}")
            continue

        coord_folder = parsed["coordinate_folder"]
        masks_folder = parsed["masks_folder"]
        direction = parsed["direction"]
        segment_id = parsed["segment_id"]
        point_type = _get_point_type(parsed, manifest)

        img_masks = f"{masks_prefix}/{masks_folder}"
        img_output = f"{output_prefix}/{coord_folder}/{direction}"

        _banner(f"[{idx}/{len(filtered)}] {Path(blob).name} "
                f"(seg={segment_id}, type={point_type})")

        local_dir = None
        if local_out:
            local_dir = str(Path(local_out) / coord_folder / direction)

        if point_type == "junction":
            n_junctions_skipped += 1
            meta = {
                "segment_id": segment_id,
                "point_id": parsed["point_id"],
                "direction": direction,
                "lat": parsed["lat"],
                "lon": parsed["lon"],
            }
            summary = process_junction_image(
                gcs, blob, img_masks, img_output, metadata=meta,
            )
            summary["segment_id"] = segment_id
            all_summaries.append(summary)
            continue

        summary = process_street_image(
            gcs, blob, img_masks, img_output, cfg,
            local_output_dir=local_dir,
        )
        summary["segment_id"] = segment_id
        all_summaries.append(summary)
        segment_results[segment_id].append(summary)

    elapsed = time.time() - t0

    # Per-segment summaries
    for seg_id, results in segment_results.items():
        seg_widths = []
        for r in results:
            if r.get("status") != "ok":
                continue
            for seg_data in r.get("segments", {}).values():
                stats = seg_data.get("width_stats", {})
                if stats and stats.get("median_cm") is not None:
                    seg_widths.append(stats["median_cm"])

        seg_summary = {
            "segment_id": seg_id,
            "n_images": len(results),
            "n_successful": sum(1 for r in results if r.get("status") == "ok"),
            "median_width_cm": (
                round(float(__import__("numpy").median(seg_widths)), 1)
                if seg_widths else None
            ),
            "width_samples": seg_widths,
        }

        seg_blob = f"{output_prefix}/segment_{seg_id}_summary.json"
        gcs.upload_bytes(
            json.dumps(seg_summary, indent=2).encode(),
            seg_blob, "application/json",
        )

    # Batch summary
    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "v2",
        "image_prefix": image_prefix,
        "prefix_range": [pmin, pmax],
        "images_processed": len(all_summaries),
        "junctions_skipped": n_junctions_skipped,
        "segments_analyzed": len(segment_results),
        "elapsed_s": round(elapsed, 1),
        "per_image": all_summaries,
    }
    summary_blob = f"{output_prefix}/_batch_summary.json"
    gcs.upload_bytes(
        json.dumps(batch_summary, indent=2, default=str).encode(),
        summary_blob, "application/json",
    )

    _banner("Batch Complete")
    _info(f"Images processed    : {len(all_summaries)}")
    _info(f"Junctions skipped   : {n_junctions_skipped}")
    _info(f"Segments analysed   : {len(segment_results)}")
    _info(f"Elapsed             : {elapsed:.1f}s")
    _ok(f"Summary -> gs://{gcs.bucket_name}/{summary_blob}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sidewalk analysis pipeline v2 — perspective-space analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config (default: config.yaml)")
    p.add_argument("--image", type=str, default=None,
                   help="Process a single image blob path (overrides batch)")
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
        gcs = GCSClient(
            project_id=gcs_cfg["project_id"],
            bucket_name=gcs_cfg["bucket_name"],
        )

        parsed = _parse_image_filename(args.image)
        if parsed is None:
            print(f"  [FAIL] Cannot parse image filename: {args.image}")
            sys.exit(1)

        masks_prefix = gcs_cfg["masks_prefix"].rstrip("/")
        output_prefix = gcs_cfg["output_prefix"].rstrip("/")
        coord = parsed["coordinate_folder"]
        masks_folder = parsed["masks_folder"]
        direction = parsed["direction"]

        local_dir = None
        if cfg.get("local_output_dir"):
            local_dir = str(Path(cfg["local_output_dir"]) / coord / direction)

        _banner("Sidewalk Analysis Pipeline v2 -- Single Image")
        process_street_image(
            gcs, args.image,
            f"{masks_prefix}/{masks_folder}",
            f"{output_prefix}/{coord}/{direction}",
            cfg,
            local_output_dir=local_dir,
        )
        _banner("Done")

    elif cfg["batch"].get("enabled"):
        run_batch(cfg)
    else:
        _info("Nothing to do. Set batch.enabled=true or use --image.")


if __name__ == "__main__":
    main()
