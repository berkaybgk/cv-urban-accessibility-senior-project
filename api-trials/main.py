#!/usr/bin/env python3
"""
Street View Sampling Pipeline – Zone-Aware
───────────────────────────────────────────
Sample points along streets, classify each as *midblock* (sidewalk
analysis) or *junction_approach* (crossing / signal analysis), fetch
Google Street View images with zone-appropriate cameras, and upload
everything to GCS.

Metadata is saved in three places:
  1. **Per-image** – custom metadata on each GCS blob (direction, zone,
     snap distance, pano coordinates, heading, etc.)
  2. **manifest.csv** – one row per image attempt (uploaded / failed /
     skipped), uploaded to GCS and returned as the results DataFrame
  3. **run_metadata.json** – run-level summary (zone counts, total
     images, config used), uploaded to GCS

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
    CAMERA_PRESETS,
    DEFAULT_CAMERA_PRESET,
    SV_FOV,
    SV_PITCH,
    SV_SIZE,
    SV_MAX_SNAP_DISTANCE_M,
    DEFAULT_SAMPLE_INTERVAL_M,
    DEFAULT_NETWORK_TYPE,
    DEFAULT_EDGE_INDEX,
    JUNCTION_ZONE_M,
)
from gcs_utils import upload_bytes, upload_file
from street_sampler import (
    sample_street,
    sample_street_bbox,
    sample_street_point,
    sample_street_polygon,
)
from streetview import check_availability, fetch_image, PanoMetadata


def _slug(text: str) -> str:
    """Turn an arbitrary string into a URL/path-safe slug."""
    return text.split(",")[0].strip().lower().replace(" ", "_")


def run_pipeline(
    *,
    place: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
    radius_m: float = 500,
    vertices: list[tuple[float, float]] | None = None,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
    max_snap_distance_m: float = SV_MAX_SNAP_DISTANCE_M,
) -> pd.DataFrame:
    """
    End-to-end pipeline.

    Returns a results DataFrame (= the manifest).  The same data is also
    uploaded to GCS as ``manifest.csv``, and a ``run_metadata.json``
    summary is stored alongside it.
    """
    sampling_kwargs = dict(
        network_type=network_type,
        interval_m=interval_m,
        edge_index=edge_index,
        sample_all=sample_all,
        junction_zone_m=junction_zone_m,
    )

    # ── 1. Sample points ────────────────────────────────────────────────────
    if place is not None:
        df = sample_street(place, **sampling_kwargs)
        run_label = place
    elif bbox is not None:
        north, south, east, west = bbox
        df = sample_street_bbox(north, south, east, west, **sampling_kwargs)
        run_label = f"bbox_{north}_{south}_{east}_{west}"
    elif center is not None:
        lat_c, lon_c = center
        df = sample_street_point(lat_c, lon_c, radius_m, **sampling_kwargs)
        run_label = f"point_{lat_c}_{lon_c}_{radius_m}m"
    elif vertices is not None:
        df = sample_street_polygon(vertices, **sampling_kwargs)
        run_label = f"polygon_{len(vertices)}v"
    else:
        raise ValueError("Provide one of: place, bbox, center, or vertices")

    if df.empty:
        print("  No sample points produced — nothing to fetch.")
        return pd.DataFrame()

    print(f"\n  Sampled {len(df)} points  "
          f"(midblock: {(df['zone_type'] == 'midblock').sum()}, "
          f"junction: {(df['zone_type'] == 'junction_approach').sum()})\n")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    place_slug = _slug(run_label)
    gcs_prefix = f"streetview/{place_slug}/{run_ts}"
    results: list[dict] = []

    # ── 2. Image counts (midblock→2, junction→1) ───────────────────────────
    total_images = sum(
        len(CAMERA_PRESETS.get(z, DEFAULT_CAMERA_PRESET))
        for z in df["zone_type"]
    )

    uploaded_count = 0
    skipped_count = 0

    point_bar = tqdm(df.iterrows(), total=len(df), desc="Points",
                     unit="pt", position=0)
    image_bar = tqdm(total=total_images, desc="Images", unit="img",
                     position=1, leave=True)

    for idx, row in point_bar:
        lat = row["latitude"]
        lon = row["longitude"]
        bearing = row["bearing"]
        dist = row["distance_along_m"]
        street = row["street_name"]
        edge_id = row["edge_id"]
        zone = row["zone_type"]
        nearest_junc = row["nearest_junction_m"]
        point_id = f"{idx:04d}"

        camera_dirs = CAMERA_PRESETS.get(zone, DEFAULT_CAMERA_PRESET)

        point_bar.set_postfix_str(
            f"{point_id} {zone} ({lat:.4f}, {lon:.4f})  "
            f"✓{uploaded_count} ⏭{skipped_count}"
        )

        # ── Metadata check (free, validates snap distance) ────────────
        pano: PanoMetadata | None = check_availability(
            lat, lon, max_snap_m=max_snap_distance_m,
        )
        if pano is None:
            skipped_count += len(camera_dirs)
            image_bar.update(len(camera_dirs))
            results.append({
                "point_id": point_id, "direction": "",
                "status": "no_coverage_or_too_far",
                "zone_type": zone, "edge_id": edge_id,
            })
            continue

        # ── Fetch images (using pano_id for exact match) ──────────────
        for direction_label, offset_deg in camera_dirs:
            heading = (bearing + offset_deg) % 360

            img_bytes = fetch_image(heading, pano_id=pano.pano_id)
            if img_bytes is None:
                skipped_count += 1
                image_bar.update(1)
                results.append({
                    "point_id": point_id,
                    "direction": direction_label,
                    "status": "fetch_failed",
                    "zone_type": zone,
                    "edge_id": edge_id,
                })
                continue

            blob_name = (
                f"{gcs_prefix}/"
                f"{point_id}_{lat}_{lon}-{direction_label}.jpg"
            )

            # ── Per-image metadata (stored on the GCS blob) ───────────
            custom_metadata = {
                "point_id":           point_id,
                "direction":          direction_label,
                "zone_type":          zone,
                "edge_id":            edge_id,
                "requested_lat":      str(lat),
                "requested_lon":      str(lon),
                "pano_lat":           str(pano.pano_lat),
                "pano_lng":           str(pano.pano_lng),
                "snap_distance_m":    str(pano.snap_distance_m),
                "street_bearing":     str(bearing),
                "heading":            str(round(heading, 2)),
                "heading_offset":     str(offset_deg),
                "distance_along_m":   str(dist),
                "nearest_junction_m": str(nearest_junc),
                "street_name":        str(street),
                "place":              run_label,
                "sample_interval_m":  str(interval_m),
                "junction_zone_m":    str(junction_zone_m),
                "max_snap_distance_m": str(max_snap_distance_m),
                "fov":                str(SV_FOV),
                "pitch":              str(SV_PITCH),
                "image_size":         SV_SIZE,
                "captured_utc":       run_ts,
                "sv_pano_id":         pano.pano_id,
                "sv_capture_date":    pano.capture_date,
            }

            upload_bytes(img_bytes, blob_name, metadata=custom_metadata)
            uploaded_count += 1
            image_bar.update(1)

            results.append({
                "point_id":        point_id,
                "direction":       direction_label,
                "status":          "uploaded",
                "zone_type":       zone,
                "edge_id":         edge_id,
                "gcs_uri":         f"gs://{GCS_BUCKET_NAME}/{blob_name}",
                "requested_lat":   lat,
                "requested_lon":   lon,
                "pano_lat":        pano.pano_lat,
                "pano_lng":        pano.pano_lng,
                "snap_distance_m": pano.snap_distance_m,
                "street_bearing":  bearing,
                "heading":         round(heading, 2),
                "pano_id":         pano.pano_id,
            })

            time.sleep(0.15)

    point_bar.close()
    image_bar.close()

    results_df = pd.DataFrame(results)

    # ── 3. Upload manifest (= results_df as CSV) ───────────────────────────
    manifest_local = os.path.join(
        os.path.dirname(__file__) or ".", f"manifest_{run_ts}.csv",
    )
    results_df.to_csv(manifest_local, index=False)
    manifest_blob = f"{gcs_prefix}/manifest.csv"
    upload_file(manifest_local, manifest_blob)
    print(f"\n  Manifest uploaded → gs://{GCS_BUCKET_NAME}/{manifest_blob}")

    # ── 4. Upload run metadata JSON ─────────────────────────────────────────
    zone_counts = df["zone_type"].value_counts().to_dict()
    run_meta = {
        "label":               run_label,
        "network_type":        network_type,
        "sample_interval_m":   interval_m,
        "junction_zone_m":     junction_zone_m,
        "max_snap_distance_m": max_snap_distance_m,
        "edge_index":          edge_index,
        "total_sample_points": len(df),
        "zone_distribution":   zone_counts,
        "camera_presets":      {
            k: [(lbl, off) for lbl, off in v]
            for k, v in CAMERA_PRESETS.items()
        },
        "images_uploaded":     int(
            (results_df["status"] == "uploaded").sum()
        ) if len(results_df) else 0,
        "run_utc":             run_ts,
    }
    meta_blob = f"{gcs_prefix}/run_metadata.json"
    upload_bytes(
        json.dumps(run_meta, indent=2).encode(),
        meta_blob,
        content_type="application/json",
    )
    print(f"  Metadata uploaded  → gs://{GCS_BUCKET_NAME}/{meta_blob}")

    # ── 5. Summary ──────────────────────────────────────────────────────────
    uploaded = int(
        (results_df["status"] == "uploaded").sum()
    ) if len(results_df) else 0
    skipped = len(results_df) - uploaded
    print("\n── Summary ───────────────────────────────────────────────────────")
    print(f"  Area                : {run_label}")
    print(f"  Total sample points : {len(df)}")
    print(f"  Zone distribution   : {zone_counts}")
    print(f"  Images uploaded     : {uploaded}")
    print(f"  Skipped / failed    : {skipped}")

    return results_df


# ── YAML config loading ─────────────────────────────────────────────────────

def _run_job(job: dict) -> pd.DataFrame:
    """
    Execute a single pipeline job described by a dict (one YAML entry).

    Recognised keys
    ───────────────
    mode (required): "place" | "bbox" | "point" | "polygon"

    Mode-specific:
      place   → name: str
      bbox    → north, south, east, west: float
      point   → lat, lon: float; radius (default 500): float
      polygon → vertices: list of [lat, lon] pairs (≥ 3)

    Common (all optional, have defaults):
      interval, network_type, edge_index, sample_all,
      junction_zone_m, max_snap_distance_m
    """
    mode = job.get("mode")
    if mode is None:
        raise ValueError(
            "Each job must have a 'mode' key (place / bbox / point / polygon)"
        )

    common = dict(
        network_type=job.get("network_type", DEFAULT_NETWORK_TYPE),
        interval_m=float(job.get("interval", DEFAULT_SAMPLE_INTERVAL_M)),
        edge_index=int(job.get("edge_index", DEFAULT_EDGE_INDEX)),
        sample_all=bool(job.get("sample_all", False)),
        junction_zone_m=float(job.get("junction_zone_m", JUNCTION_ZONE_M)),
        max_snap_distance_m=float(
            job.get("max_snap_distance_m", SV_MAX_SNAP_DISTANCE_M)
        ),
    )

    if mode == "place":
        name = job.get("name")
        if not name:
            raise ValueError("mode 'place' requires a 'name' key")
        return run_pipeline(place=name, **common)

    elif mode == "bbox":
        for k in ("north", "south", "east", "west"):
            if k not in job:
                raise ValueError(f"mode 'bbox' requires a '{k}' key")
        return run_pipeline(
            bbox=(float(job["north"]), float(job["south"]),
                  float(job["east"]),  float(job["west"])),
            **common,
        )

    elif mode == "point":
        for k in ("lat", "lon"):
            if k not in job:
                raise ValueError(f"mode 'point' requires a '{k}' key")
        return run_pipeline(
            center=(float(job["lat"]), float(job["lon"])),
            radius_m=float(job.get("radius", 500)),
            **common,
        )

    elif mode == "polygon":
        raw = job.get("vertices")
        if not raw or not isinstance(raw, list) or len(raw) < 3:
            raise ValueError(
                "mode 'polygon' requires a 'vertices' list with ≥ 3 "
                "[lat, lon] pairs"
            )
        verts = [(float(v[0]), float(v[1])) for v in raw]
        return run_pipeline(vertices=verts, **common)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use place / bbox / point / polygon"
        )


def load_and_run(yaml_path: str) -> list[pd.DataFrame]:
    """Load a YAML file and execute every job defined in it."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        jobs = [data]
    elif isinstance(data, list):
        jobs = data
    else:
        raise ValueError(
            "YAML must be a mapping (single job) or a list of mappings"
        )

    results = []
    for i, job in enumerate(jobs, 1):
        print(f"\n{'═' * 70}")
        print(f"  Job {i}/{len(jobs)}")
        print(f"{'═' * 70}\n")
        results.append(_run_job(job))

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sample Street View images for an area and upload to GCS.",
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
