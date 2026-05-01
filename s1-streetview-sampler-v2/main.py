#!/usr/bin/env python3
"""
Street View Sampling Pipeline v2
─────────────────────────────────
Sample points along streets inside a polygon area, fetch Google Street View
images (forward + backward + left + right) for each point, and upload
everything to GCS.

Key v2 improvements:
  - Polygon-only mode; no place/bbox/point.
  - Junction-aware segment decomposition with consistent direction.
  - Images fetched by pano_id; both requested AND actual panorama
    coordinates are recorded.
  - Segment-aware file naming for easy batch grouping.
  - Greedy nearest-neighbor ordering so consecutive points are as close
    as possible, with pano-id deduplication (no two points share a pano,
    except forward/backward of the same point).
  - Left/right images at a steeper downward pitch (-40°) in addition
    to the standard forward/backward pair.

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
import math
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


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _deduplicate_nearest_order(rows: list[dict]) -> list[dict]:
    """
    Greedy nearest-neighbour ordering with pano-id deduplication.

    Starting from the first point, repeatedly pick the closest unvisited
    point whose pano_id has not been seen yet.  This gives a traversal
    where consecutive points are as close together as possible, with no
    duplicate panoramas.
    """
    if not rows:
        return []

    remaining = list(range(len(rows)))
    current = remaining.pop(0)
    ordered: list[int] = [current]
    seen_panos: set[str] = {rows[current]["pano_id"]}

    while remaining:
        cur_lat = rows[current]["latitude"]
        cur_lon = rows[current]["longitude"]

        best_idx = None
        best_dist = float("inf")
        best_pos = -1

        for pos, idx in enumerate(remaining):
            pid = rows[idx]["pano_id"]
            if pid in seen_panos:
                continue
            d = _haversine_m(cur_lat, cur_lon,
                             rows[idx]["latitude"], rows[idx]["longitude"])
            if d < best_dist:
                best_dist = d
                best_idx = idx
                best_pos = pos

        if best_idx is None:
            break

        remaining.pop(best_pos)
        ordered.append(best_idx)
        seen_panos.add(rows[best_idx]["pano_id"])
        current = best_idx

    return [rows[i] for i in ordered]


def run_pipeline(
    *,
    vertices: list[tuple[float, float]],
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
) -> pd.DataFrame:
    """
    End-to-end pipeline:
      1. Sample points along streets inside the polygon.
      2. Check Street View availability for every point.
      3. Deduplicate by pano_id using greedy nearest-neighbour ordering.
      4. Fetch 4 images per unique point (forward, backward, left, right).
      5. Upload images + per-image metadata to GCS.
      6. Upload a CSV manifest + run_metadata.json to GCS.
      7. Return a results DataFrame.
    """
    # ── 1. Sample points ────────────────────────────────────────────────────
    df = sample_polygon(vertices, interval_m, network_type)
    run_label = f"polygon_{len(vertices)}v"

    print(f"\n  Sampled {len(df)} points every {interval_m} m")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    place_slug = _slug(run_label)
    gcs_prefix = f"streetview/{place_slug}/{run_ts}"

    # ── 2. Resolve pano metadata for every sample point ─────────────────────
    print("\n  Resolving Street View metadata …")
    sv_rows: list[dict] = []
    no_coverage = 0

    meta_bar = tqdm(df.iterrows(), total=len(df), desc="Metadata",
                    unit="pt", position=0)
    for _, row in meta_bar:
        lat = row["latitude"]
        lon = row["longitude"]
        sv_meta = check_availability(lat, lon)
        if sv_meta is None:
            no_coverage += 1
            continue
        sv_rows.append({
            "segment_id":       row["segment_id"],
            "point_id":         row["point_id"],
            "point_type":       row["point_type"],
            "latitude":         lat,
            "longitude":        lon,
            "bearing":          row["bearing"],
            "distance_along_m": row["distance_along_m"],
            "street_name":      row["street_name"],
            "pano_id":          sv_meta["pano_id"],
            "pano_lat":         sv_meta["pano_lat"],
            "pano_lng":         sv_meta["pano_lng"],
            "sv_date":          sv_meta.get("date", ""),
        })
    meta_bar.close()

    print(f"  Coverage: {len(sv_rows)}/{len(df)} points "
          f"({no_coverage} without Street View)")

    # ── 3. Deduplicate by pano-id, nearest-neighbour order ──────────────────
    unique_rows = _deduplicate_nearest_order(sv_rows)
    dup_count = len(sv_rows) - len(unique_rows)
    print(f"  After pano dedup: {len(unique_rows)} unique panoramas "
          f"({dup_count} duplicates removed)")

    if not unique_rows:
        print("  WARNING: No unique panoramas to fetch!")
        return pd.DataFrame()

    # ── 4. Fetch images and upload ──────────────────────────────────────────
    results: list[dict] = []
    uploaded_count = 0
    skipped_count = 0
    total_images = len(unique_rows) * len(CAMERA_DIRECTIONS)

    point_bar = tqdm(enumerate(unique_rows), total=len(unique_rows),
                     desc="Points", unit="pt", position=0)
    image_bar = tqdm(total=total_images, desc="Images", unit="img",
                     position=1, leave=True)

    for new_idx, row in point_bar:
        lat = row["latitude"]
        lon = row["longitude"]
        bearing = row["bearing"]
        dist = row["distance_along_m"]
        street = row["street_name"]
        seg_id = row["segment_id"]
        point_id = f"{new_idx:04d}"
        point_type = row["point_type"]
        pano_id = row["pano_id"]
        pano_lat = row["pano_lat"]
        pano_lng = row["pano_lng"]
        sv_date = row["sv_date"]

        point_bar.set_postfix_str(
            f"seg{seg_id}/{point_id} ({lat:.4f}, {lon:.4f})  "
            f"ok={uploaded_count} skip={skipped_count}"
        )

        for direction_label, offset_deg, pitch in CAMERA_DIRECTIONS:
            heading = (bearing + offset_deg) % 360

            img_bytes = fetch_image(pano_id, heading=heading, pitch=pitch)
            if img_bytes is None:
                skipped_count += 1
                image_bar.update(1)
                results.append({
                    "segment_id": seg_id,
                    "point_id":   point_id,
                    "point_type": point_type,
                    "direction":  direction_label,
                    "status":     "fetch_failed",
                })
                continue

            blob_name = (
                f"{gcs_prefix}/"
                f"{seg_id:03d}-{point_id}_{direction_label}"
                f"_{lat}_{lon}_{heading:.1f}.jpg"
            )

            custom_metadata = {
                "segment_id":        str(seg_id),
                "point_id":          point_id,
                "point_type":        point_type,
                "direction":         direction_label,
                "latitude":          str(lat),
                "longitude":         str(lon),
                "pano_id":           pano_id,
                "pano_lat":          str(pano_lat),
                "pano_lng":          str(pano_lng),
                "street_bearing":    str(bearing),
                "heading":           str(heading),
                "heading_offset":    str(offset_deg),
                "pitch":             str(pitch),
                "distance_along_m":  str(dist),
                "street_name":       str(street),
                "place":             run_label,
                "sample_interval_m": str(interval_m),
                "fov":               str(SV_FOV),
                "image_size":        SV_SIZE,
                "captured_utc":      run_ts,
                "sv_capture_date":   sv_date,
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
                "pitch":          pitch,
            })

            time.sleep(0.15)

    point_bar.close()
    image_bar.close()

    results_df = pd.DataFrame(results)

    # ── 5. Upload manifest ──────────────────────────────────────────────────
    manifest_local = os.path.join(
        os.path.dirname(__file__) or ".", f"manifest_{run_ts}.csv",
    )
    results_df.to_csv(manifest_local, index=False)
    manifest_blob = f"{gcs_prefix}/manifest.csv"
    upload_file(manifest_local, manifest_blob)
    print(f"\n  Manifest uploaded -> gs://{GCS_BUCKET_NAME}/{manifest_blob}")

    # ── 6. Upload run metadata JSON ─────────────────────────────────────────
    run_meta = {
        "label": run_label,
        "network_type": network_type,
        "sample_interval_m": interval_m,
        "total_sample_points": len(df),
        "total_segments": int(df["segment_id"].nunique()),
        "junction_points": int((df["point_type"] == "junction").sum()),
        "unique_panoramas": len(unique_rows),
        "duplicate_panos_removed": dup_count,
        "images_uploaded": int((results_df["status"] == "uploaded").sum())
            if len(results_df) else 0,
        "directions_per_point": len(CAMERA_DIRECTIONS),
        "run_utc": run_ts,
        "version": "v2.1",
    }
    meta_blob = f"{gcs_prefix}/run_metadata.json"
    upload_bytes(
        json.dumps(run_meta, indent=2).encode(),
        meta_blob,
        content_type="application/json",
    )
    print(f"  Metadata uploaded  -> gs://{GCS_BUCKET_NAME}/{meta_blob}")

    # ── 7. Summary ──────────────────────────────────────────────────────────
    uploaded = (int((results_df["status"] == "uploaded").sum())
                if len(results_df) else 0)
    skipped_total = len(results_df) - uploaded if len(results_df) else 0
    print("\n-- Summary ---------------------------------------------------")
    print(f"  Area                 : {run_label}")
    print(f"  Total sample points  : {len(df)}")
    print(f"  Segments             : {int(df['segment_id'].nunique())}")
    print(f"  Junction points      : {int((df['point_type'] == 'junction').sum())}")
    print(f"  Unique panoramas     : {len(unique_rows)}")
    print(f"  Duplicates removed   : {dup_count}")
    print(f"  Directions / point   : {len(CAMERA_DIRECTIONS)}")
    print(f"  Images uploaded      : {uploaded}")
    print(f"  Skipped / failed     : {skipped_total}")

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
