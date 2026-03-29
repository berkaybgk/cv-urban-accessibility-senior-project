#!/usr/bin/env python3
"""
Street View Sampling Pipeline v2
─────────────────────────────────
Sample points along streets inside a polygon area, fetch Google Street View
images (forward + backward) for each point, and upload everything to GCS.

Key v2 improvements:
  - Polygon-only mode; no place/bbox/point.
  - Forward + backward only (no left/right).
  - Junction-aware segment decomposition with consistent direction.
  - Images fetched by pano_id; both requested AND actual panorama
    coordinates are recorded.
  - Segment-aware file naming for easy batch grouping.

Usage
─────
  python main.py run.yaml

Required environment variables (in .env):
    GOOGLE_MAPS_API_KEY
    GCS_BUCKET_NAME
    GCP_PROJECT_ID
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import yaml
from tqdm import tqdm

from config import (
    GOOGLE_MAPS_API_KEY,
    GCS_BUCKET_NAME,
    CAMERA_DIRECTIONS,
    SV_FOV,
    SV_PITCH,
    SV_SIZE,
    DEFAULT_SAMPLE_INTERVAL_M,
    DEFAULT_NETWORK_TYPE,
)
from gcs_utils import upload_bytes, upload_file
from street_sampler import sample_polygon
from streetview import check_availability, fetch_image


def _slug(text: str) -> str:
    """Turn an arbitrary string into a URL/path-safe slug."""
    return text.split(",")[0].strip().lower().replace(" ", "_")


def run_pipeline(
    *,
    vertices: list[tuple[float, float]],
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
) -> pd.DataFrame:
    """
    End-to-end pipeline:
      1. Sample points along streets inside the polygon
      2. For each point, check Street View availability
      3. Fetch forward + backward images (by pano_id)
      4. Upload images + per-image metadata to GCS
      5. Upload a CSV manifest + run_metadata.json to GCS
      6. Return a results DataFrame
    """
    # ── 1. Sample points ────────────────────────────────────────────────────
    df = sample_polygon(vertices, interval_m, network_type)
    run_label = f"polygon_{len(vertices)}v"

    print(f"\n  Sampled {len(df)} points every {interval_m} m\n")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    place_slug = _slug(run_label)
    gcs_prefix = f"streetview/{place_slug}/{run_ts}"
    results: list[dict] = []

    # ── 2. Fetch images and upload ──────────────────────────────────────────
    uploaded_count = 0
    skipped_count = 0
    total_images = len(df) * len(CAMERA_DIRECTIONS)

    point_bar = tqdm(df.iterrows(), total=len(df), desc="Points",
                     unit="pt", position=0)
    image_bar = tqdm(total=total_images, desc="Images", unit="img",
                     position=1, leave=True)

    for _, row in point_bar:
        lat = row["latitude"]
        lon = row["longitude"]
        bearing = row["bearing"]
        dist = row["distance_along_m"]
        street = row["street_name"]
        seg_id = row["segment_id"]
        point_id = row["point_id"]
        point_type = row["point_type"]

        point_bar.set_postfix_str(
            f"seg{seg_id}/{point_id} ({lat:.4f}, {lon:.4f})  "
            f"ok={uploaded_count} skip={skipped_count}"
        )

        sv_meta = check_availability(lat, lon)
        if sv_meta is None:
            skipped_count += len(CAMERA_DIRECTIONS)
            image_bar.update(len(CAMERA_DIRECTIONS))
            results.append({
                "segment_id": seg_id,
                "point_id": point_id,
                "point_type": point_type,
                "direction": "",
                "status": "no_coverage",
            })
            continue

        pano_id = sv_meta["pano_id"]
        pano_lat = sv_meta["pano_lat"]
        pano_lng = sv_meta["pano_lng"]

        for direction_label, offset_deg in CAMERA_DIRECTIONS:
            heading = (bearing + offset_deg) % 360

            img_bytes = fetch_image(pano_id, heading=heading)
            if img_bytes is None:
                skipped_count += 1
                image_bar.update(1)
                results.append({
                    "segment_id": seg_id,
                    "point_id": point_id,
                    "point_type": point_type,
                    "direction": direction_label,
                    "status": "fetch_failed",
                })
                continue

            blob_name = (
                f"{gcs_prefix}/"
                f"{seg_id:03d}-{point_id}_{direction_label}"
                f"_{lat}_{lon}_{heading:.1f}.jpg"
            )

            custom_metadata = {
                "segment_id":       str(seg_id),
                "point_id":         point_id,
                "point_type":       point_type,
                "direction":        direction_label,
                "latitude":         str(lat),
                "longitude":        str(lon),
                "pano_id":          pano_id,
                "pano_lat":         str(pano_lat),
                "pano_lng":         str(pano_lng),
                "street_bearing":   str(bearing),
                "heading":          str(heading),
                "heading_offset":   str(offset_deg),
                "distance_along_m": str(dist),
                "street_name":      str(street),
                "place":            run_label,
                "sample_interval_m": str(interval_m),
                "fov":              str(SV_FOV),
                "pitch":            str(SV_PITCH),
                "image_size":       SV_SIZE,
                "captured_utc":     run_ts,
                "sv_capture_date":  sv_meta.get("date", ""),
            }

            upload_bytes(img_bytes, blob_name, metadata=custom_metadata)
            uploaded_count += 1
            image_bar.update(1)

            results.append({
                "segment_id":     seg_id,
                "point_id":       point_id,
                "point_type":     point_type,
                "direction":      direction_label,
                "status":         "uploaded",
                "gcs_uri":        f"gs://{GCS_BUCKET_NAME}/{blob_name}",
                "latitude":       lat,
                "longitude":      lon,
                "pano_id":        pano_id,
                "pano_lat":       pano_lat,
                "pano_lng":       pano_lng,
                "street_bearing": bearing,
                "heading":        heading,
            })

            time.sleep(0.15)

    point_bar.close()
    image_bar.close()

    results_df = pd.DataFrame(results)

    # ── 3. Upload manifest ──────────────────────────────────────────────────
    manifest_local = os.path.join(
        os.path.dirname(__file__) or ".", f"manifest_{run_ts}.csv",
    )
    results_df.to_csv(manifest_local, index=False)
    manifest_blob = f"{gcs_prefix}/manifest.csv"
    upload_file(manifest_local, manifest_blob)
    print(f"\n  Manifest uploaded -> gs://{GCS_BUCKET_NAME}/{manifest_blob}")

    # ── 4. Upload run metadata JSON ─────────────────────────────────────────
    run_meta = {
        "label": run_label,
        "network_type": network_type,
        "sample_interval_m": interval_m,
        "total_sample_points": len(df),
        "total_segments": int(df["segment_id"].nunique()),
        "junction_points": int((df["point_type"] == "junction").sum()),
        "images_uploaded": int((results_df["status"] == "uploaded").sum())
            if len(results_df) else 0,
        "run_utc": run_ts,
        "version": "v2",
    }
    meta_blob = f"{gcs_prefix}/run_metadata.json"
    upload_bytes(
        json.dumps(run_meta, indent=2).encode(),
        meta_blob,
        content_type="application/json",
    )
    print(f"  Metadata uploaded  -> gs://{GCS_BUCKET_NAME}/{meta_blob}")

    # ── 5. Summary ──────────────────────────────────────────────────────────
    uploaded = (int((results_df["status"] == "uploaded").sum())
                if len(results_df) else 0)
    skipped_total = len(results_df) - uploaded if len(results_df) else 0
    print("\n-- Summary ---------------------------------------------------")
    print(f"  Area                : {run_label}")
    print(f"  Total sample points : {len(df)}")
    print(f"  Segments            : {int(df['segment_id'].nunique())}")
    print(f"  Junction points     : {int((df['point_type'] == 'junction').sum())}")
    print(f"  Directions / point  : {len(CAMERA_DIRECTIONS)}")
    print(f"  Images uploaded     : {uploaded}")
    print(f"  Skipped / failed    : {skipped_total}")

    return results_df


# ── YAML config loading ─────────────────────────────────────────────────────

def _run_job(job: dict) -> pd.DataFrame:
    """
    Execute a single pipeline job described by a dict (one YAML entry).

    Required keys:
      vertices: list of [lat, lon] pairs (at least 3)

    Optional keys:
      interval      : float  (default 15)
      network_type  : str    (default "all_public")
    """
    raw = job.get("vertices")
    if not raw or not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(
            "Each job requires a 'vertices' list with >= 3 [lat, lon] pairs"
        )
    verts = [(float(v[0]), float(v[1])) for v in raw]

    return run_pipeline(
        vertices=verts,
        network_type=job.get("network_type", DEFAULT_NETWORK_TYPE),
        interval_m=float(job.get("interval", DEFAULT_SAMPLE_INTERVAL_M)),
    )


def load_and_run(yaml_path: str) -> list[pd.DataFrame]:
    """
    Load a YAML file and execute every job defined in it.

    The YAML can be:
      - A single mapping  (one job)
      - A list of mappings (multiple jobs run sequentially)
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        jobs = [data]
    elif isinstance(data, list):
        jobs = data
    else:
        raise ValueError("YAML must be a mapping (single job) or a list of mappings")

    results = []
    for i, job in enumerate(jobs, 1):
        print(f"\n{'=' * 70}")
        print(f"  Job {i}/{len(jobs)}")
        print(f"{'=' * 70}\n")
        results.append(_run_job(job))

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample Street View images inside a polygon and upload to GCS.",
    )
    parser.add_argument(
        "config",
        help="Path to a YAML config file (see run.yaml)",
    )
    args = parser.parse_args(argv)

    if not GOOGLE_MAPS_API_KEY:
        sys.exit("ERROR: GOOGLE_MAPS_API_KEY is not set. Add it to .env")
    if not GCS_BUCKET_NAME:
        sys.exit("ERROR: GCS_BUCKET_NAME is not set. Add it to .env")

    load_and_run(args.config)


if __name__ == "__main__":
    main()
