#!/usr/bin/env python3
"""
Sidewalk visualization pipeline v2
──────────────────────────────────
Same core analysis as v1 (pinhole width, polyfit edges, strip rectification,
footprint estimation), with v2-only conveniences: modular layout, v1+v2
filename parsing, optional manifest for junction vs street.

Usage
─────
  python main.py --config config.yaml
  python main.py --image <blob_path>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
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

    Mask paths match the inference layout (same as v1):
    ``{masks_prefix}/{point_id}_{lat}_{lon}/{direction}``
    """
    name = Path(filename).name

    m = _FILENAME_RE_V2.match(name)
    if m:
        coordinate_folder = f"{m.group(2)}_{m.group(4)}_{m.group(5)}"
        direction = m.group(3)
        return {
            "segment_id": m.group(1),
            "point_id": m.group(2),
            "direction": direction,
            "lat": m.group(4),
            "lon": m.group(5),
            "heading": m.group(6),
            "coordinate_folder": coordinate_folder,
        }

    m = _FILENAME_RE_V1.match(name)
    if m:
        coordinate_folder = f"{m.group(1)}_{m.group(3)}_{m.group(4)}"
        direction = m.group(2)
        return {
            "segment_id": "0",
            "point_id": m.group(1),
            "direction": direction,
            "lat": m.group(3),
            "lon": m.group(4),
            "heading": m.group(5),
            "coordinate_folder": coordinate_folder,
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
    n_junction_stubs = 0
    t0 = time.time()

    for idx, (num, blob) in enumerate(filtered, 1):
        parsed = _parse_image_filename(blob)
        if parsed is None:
            _info(f"[{idx}/{len(filtered)}] Skipping (unrecognised name): {Path(blob).name}")
            continue

        coord_folder = parsed["coordinate_folder"]
        direction = parsed["direction"]
        segment_id = parsed["segment_id"]
        point_type = _get_point_type(parsed, manifest)

        img_masks = f"{masks_prefix}/{coord_folder}/{direction}"
        img_output = f"{output_prefix}/{coord_folder}/{direction}"

        _banner(f"[{idx}/{len(filtered)}] {Path(blob).name} "
                f"(seg={segment_id}, type={point_type})")

        local_dir = None
        if local_out:
            local_dir = str(Path(local_out) / coord_folder / direction)

        if point_type == "junction":
            n_junction_stubs += 1
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

    elapsed = time.time() - t0

    segment_ids = {s.get("segment_id") for s in all_summaries if s.get("segment_id")}

    batch_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "v2",
        "image_prefix": image_prefix,
        "prefix_range": [pmin, pmax],
        "images_processed": len(all_summaries),
        "junction_stubs": n_junction_stubs,
        "segment_ids": sorted(segment_ids, key=lambda x: str(x)),
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
    _info(f"Junction stubs      : {n_junction_stubs}")
    _info(f"Elapsed             : {elapsed:.1f}s")
    _ok(f"Summary -> gs://{gcs.bucket_name}/{summary_blob}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sidewalk visualization pipeline v2 (v1-style analysis)",
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
        direction = parsed["direction"]

        local_dir = None
        if cfg.get("local_output_dir"):
            local_dir = str(Path(cfg["local_output_dir"]) / coord / direction)

        _banner("Sidewalk Analysis Pipeline v2 -- Single Image")
        process_street_image(
            gcs, args.image,
            f"{masks_prefix}/{coord}/{direction}",
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
