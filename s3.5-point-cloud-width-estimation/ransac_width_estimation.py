#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RANSAC-based sidewalk width estimation for sidewalk-facing images.

Unlike the manual approach (manual_width_estimation.py), this script:
  - Loads images and their pre-computed sidewalk masks directly from GCS
  - Detects the upper and lower pixel boundaries of the sidewalk mask
    using RANSAC line fitting (adapted from the left/right boundary logic
    in continuous_sidewalk_rectification_strip.ipynb)
  - Projects the two horizontal boundary lines back to physical distances
    using the same camera-geometry model as manual_width_estimation.py

The images are *sidewalk-facing* (camera roughly perpendicular to the
walking direction), so:
  - "upper boundary"  -> top edge of the sidewalk mask  (wall / building side)
  - "lower boundary"  -> bottom edge of the sidewalk mask (road / curb side)

These map to y_wall and y_curb in the calculate_width() projection formula.

Usage (standalone):
    python ransac_width_estimation.py \\
        --manifest gs://my-bucket/streetview/.../manifest.csv \\
        --point_ids 0258 0259 0260 \\
        --directions left right \\
        --masks_root v3/segmentation-results \\
        --fov 90 --pitch 20 --cam_height 2.5 \\
        --out_dir ransac_width_output

Or import individual functions and call them from a Jupyter notebook:
    from ransac_width_estimation import (
        load_manifest, GCSClient,
        load_image_and_mask, find_horizontal_boundaries,
        estimate_width_from_boundaries, process_image
    )

Jupyter notebook segmentation guide
------------------------------------
This script is written so it can be split cleanly into notebook cells.
Look for the  ── CELL BREAK ──  comments to find natural split points.
"""

from __future__ import annotations

# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Imports and configuration

import csv
import io
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from sklearn.linear_model import LinearRegression, RANSACRegressor

# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## User configuration — edit these before running

MANIFEST_CSV = "https://storage.cloud.google.com/cv-urban-accessibility-bucket/streetview/polygon_4v/20260516T081917Z/manifest.csv"
POINT_IDS = ["0011"]  # 0258 – 0286 inclusive
DIRECTIONS = ["right"]  # which camera directions to process

MASKS_ROOT = "v4-20260516T081917Z/segmentation-results"

# Camera intrinsics / extrinsics
FOV_DEG = 90.0       # vertical field-of-view in degrees
PITCH_DEG = -38.0     # camera downward pitch in degrees (negative = looking down)
CAM_HEIGHT_M = 2.5   # camera height above ground in metres

# RANSAC boundary fitting parameters
RANSAC_RESIDUAL_THRESHOLD = 2.0   # pixels; inlier tolerance for boundary lines
RANSAC_MIN_SAMPLES = 0.25         # fraction of valid rows used as min_samples
BORDER_MARGIN = 3                 # pixels; columns/rows within this of the edge
                                  # are considered "clipped" and extrapolated
SHIFT_PERCENTILE_UPPER = 10       # shift RANSAC line toward the tightest upper bound
SHIFT_PERCENTILE_LOWER = 90       # shift RANSAC line toward the tightest lower bound
MIN_MASK_AREA = 500               # pixels; skip masks smaller than this
MIN_ROW_COVERAGE = 0.05           # fraction of rows that must contain sidewalk

# Output
OUTPUT_DIR = "ransac_width_output"
SAVE_VISUALIZATIONS = True

# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Environment / GCS helpers

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Filename parsing helpers

_FILENAME_RE_NEW = re.compile(
    r"^(\d+)-(\d+)_(forward|backward|left|right)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)
_FILENAME_RE_OLD = re.compile(
    r"^(\d+)_(forward|backward|left|right)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)


def normalize_point_id(point_id: str | int) -> str:
    s = str(point_id).strip()
    return s.zfill(4) if s.isdigit() else s


def parse_gcs_path(uri_or_blob: str | Path) -> tuple[str | None, str]:
    """Return (bucket, blob). bucket=None means this is not an explicit GCS URI."""
    s = str(uri_or_blob).strip()
    if s.startswith("gs://"):
        parts = s.split("/", 3)
        bucket = parts[2] if len(parts) > 2 else ""
        blob = parts[3] if len(parts) > 3 else ""
        return bucket, blob

    if s.startswith(("http://", "https://")):
        parsed = urlparse(s)
        host = parsed.netloc
        path = unquote(parsed.path.lstrip("/"))

        if host in {"storage.cloud.google.com", "storage.googleapis.com"}:
            bucket, _, blob = path.partition("/")
            return bucket, blob

        suffix = ".storage.googleapis.com"
        if host.endswith(suffix):
            bucket = host[: -len(suffix)]
            return bucket, path

    return None, s.lstrip("/")


def blob_from_gcs_uri(uri_or_blob: str) -> str:
    _, blob = parse_gcs_path(uri_or_blob)
    return blob


def parse_image_filename(filename: str) -> dict[str, str] | None:
    name = Path(filename).name
    m_new = _FILENAME_RE_NEW.match(name)
    if m_new:
        street_id, point_id, direction, lat, lon, heading = m_new.groups()
        return {
            "street_id": street_id,
            "point_id": point_id,
            "index": point_id,
            "direction": direction,
            "lat": lat,
            "lon": lon,
            "heading": heading,
            "coordinate_folder": f"{point_id}_{lat}_{lon}",
        }

    m_old = _FILENAME_RE_OLD.match(name)
    if not m_old:
        return None
    point_id, direction, lat, lon, heading = m_old.groups()
    return {
        "street_id": "",
        "point_id": point_id,
        "index": point_id,
        "direction": direction,
        "lat": lat,
        "lon": lon,
        "heading": heading,
        "coordinate_folder": f"{point_id}_{lat}_{lon}",
    }


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## GCS client + manifest loading

class GCSClient:
    def __init__(self, project_id: str, bucket_name: str) -> None:
        self._client = storage.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def download_as_bytes(self, blob_name: str) -> bytes:
        return self._bucket.blob(blob_name).download_as_bytes()

    def list_blobs(self, prefix: str) -> list[str]:
        return [b.name for b in self._bucket.list_blobs(prefix=prefix)]


def _open_manifest_text(path: str | Path, gcs: GCSClient | None = None):
    """Open a manifest from a local path, gs:// URI, or HTTPS GCS URL."""
    bucket_name, blob_name = parse_gcs_path(path)
    if bucket_name:
        client = gcs or GCSClient(GCP_PROJECT_ID, bucket_name)
        data = client.download_as_bytes(blob_name)
        print(f"Manifest source: gs://{bucket_name}/{blob_name}")
        return io.StringIO(data.decode("utf-8-sig"))

    local_path = Path(path).expanduser()
    if local_path.exists():
        print(f"Manifest source: {local_path}")
        return local_path.open(newline="")

    if GCS_BUCKET_NAME:
        data = (gcs or GCSClient(GCP_PROJECT_ID, GCS_BUCKET_NAME)).download_as_bytes(blob_name)
        print(f"Manifest source: gs://{GCS_BUCKET_NAME}/{blob_name}")
        return io.StringIO(data.decode("utf-8-sig"))

    raise FileNotFoundError(f"Manifest not found locally and GCS_BUCKET_NAME is not set: {path}")


def load_manifest(path: str | Path, gcs: GCSClient | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Load manifest CSV and return {point_id: {direction: row_dict}}."""
    rows_by_point: dict[str, dict[str, dict[str, Any]]] = {}
    with _open_manifest_text(path, gcs) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") and row["status"] != "uploaded":
                continue
            point_id = normalize_point_id(row.get("point_id", ""))
            direction = row.get("direction", "").strip()
            gcs_uri = row.get("gcs_uri", "").strip()
            if not point_id or direction not in {"forward", "backward", "left", "right"} or not gcs_uri:
                continue
            blob = blob_from_gcs_uri(gcs_uri)
            parsed = parse_image_filename(blob) or {
                "point_id": point_id,
                "direction": direction,
                "lat": row.get("latitude", ""),
                "lon": row.get("longitude", ""),
                "heading": row.get("heading", ""),
                "coordinate_folder": f"{point_id}_{row.get('latitude', '')}_{row.get('longitude', '')}",
            }
            row = dict(row)
            row["blob_name"] = blob
            row["parsed"] = parsed
            rows_by_point.setdefault(point_id, {})[direction] = row
    return rows_by_point


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Image + mask loading helpers

def bytes_to_image(data: bytes) -> np.ndarray:
    """Decode image bytes → RGB numpy array."""
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def bytes_to_mask(data: bytes) -> np.ndarray:
    """Decode mask bytes → boolean numpy array (True = sidewalk)."""
    mask = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return (mask > 127).astype(bool)


def resolve_masks_prefix(
    gcs: GCSClient, masks_root: str, parsed: dict[str, str]
) -> tuple[str, str]:
    """Find the GCS prefix where masks live for a given image."""
    direction = parsed["direction"]
    point_id = parsed["point_id"]
    preferred_coord = parsed["coordinate_folder"]
    preferred = f"{masks_root.rstrip('/')}/{preferred_coord}/{direction}"

    exact = [b for b in gcs.list_blobs(preferred + "/") if b.endswith(".png")]
    if exact:
        return preferred, preferred_coord

    all_for_point = gcs.list_blobs(f"{masks_root.rstrip('/')}/{point_id}_")
    coord_candidates: set[str] = set()
    marker = f"/{direction}/"
    for blob in all_for_point:
        if marker not in blob or not blob.endswith(".png"):
            continue
        rel = blob[len(masks_root.rstrip("/") + "/"):]
        coord_candidates.add(rel.split("/", 1)[0])

    if not coord_candidates:
        return preferred, preferred_coord

    if preferred_coord in coord_candidates:
        return preferred, preferred_coord

    def coord_distance_sq(coord_folder: str) -> float:
        m = re.match(r"^\d+_([-\d.]+)_([-\d.]+)$", coord_folder)
        if not m:
            return float("inf")
        try:
            d_lat = float(m.group(1)) - float(parsed.get("lat", 0))
            d_lon = float(m.group(2)) - float(parsed.get("lon", 0))
        except (TypeError, ValueError):
            return float("inf")
        return d_lat * d_lat + d_lon * d_lon

    chosen = sorted(coord_candidates, key=lambda c: (coord_distance_sq(c), c))[0]
    return f"{masks_root.rstrip('/')}/{chosen}/{direction}", chosen


def load_sidewalk_masks(
    gcs: GCSClient, masks_prefix: str, shape: tuple[int, int]
) -> list[dict[str, Any]]:
    """Download all sidewalk mask PNGs for an image and return sorted by area."""
    blobs = [b for b in gcs.list_blobs(masks_prefix + "/sidewalk/") if b.endswith(".png")]
    masks = []
    H, W = shape
    for blob in blobs:
        mask = bytes_to_mask(gcs.download_as_bytes(blob))
        if mask.shape != shape:
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        cols = np.where(mask.any(axis=0))[0]
        if len(cols) == 0:
            continue
        side = "left" if float(np.mean(cols)) < W / 2 else "right"
        masks.append(
            {
                "mask": mask,
                "side": side,
                "blob_name": blob,
                "area": int(mask.sum()),
            }
        )
    masks.sort(key=lambda m: m["area"], reverse=True)
    return masks


def load_image_and_mask(
    gcs: GCSClient,
    row: dict[str, Any],
    masks_root: str,
) -> tuple[np.ndarray, np.ndarray | None, str, dict[str, Any]]:
    """
    Download image + best sidewalk mask for a manifest row.

    Returns
    -------
    image       : RGB uint8 ndarray
    mask        : bool ndarray, or None if no mask found
    masks_prefix: GCS prefix where masks were found
    info        : dict with provenance (blob names, area, etc.)
    """
    blob = row["blob_name"]
    parsed = row["parsed"]
    img = bytes_to_image(gcs.download_as_bytes(blob))
    masks_prefix, coord = resolve_masks_prefix(gcs, masks_root, parsed)
    masks = load_sidewalk_masks(gcs, masks_prefix, img.shape[:2])

    if not masks:
        return img, None, masks_prefix, {"image_blob": blob, "mask_blob": None, "coord": coord}

    # For sidewalk-facing images (left/right), pick the largest mask.
    best = masks[0]
    return img, best["mask"], masks_prefix, {
        "image_blob": blob,
        "mask_blob": best["blob_name"],
        "coord": coord,
        "mask_area": best["area"],
        "mask_side": best["side"],
    }


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## RANSAC boundary fitting (upper / lower edges in image space)
#
# For sidewalk-facing images the sidewalk appears as a roughly horizontal
# band in the image.  We want to fit two lines:
#
#   upper_boundary(col) = a_U * col + b_U   ← top edge of the mask band
#   lower_boundary(col) = a_L * col + b_L   ← bottom edge of the mask band
#
# The approach mirrors what `find_row_edges` does in the rectification
# notebook, but we swap the role of rows and columns:
#   • In the rectification notebook: scan each *row*, find left/right column edges.
#   • Here:                          scan each *column*, find upper/lower row edges.


def _fit_boundary_line(
    valid_cols: np.ndarray,
    edge_rows: np.ndarray,
    side: str,  # "upper" or "lower"
    ransac_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
    ransac_min_samples: float = RANSAC_MIN_SAMPLES,
) -> tuple[float, float, np.ndarray]:
    """
    Fit a line  row = a * col + b  to (col, row) boundary samples using RANSAC.

    For the upper boundary we want the line to hug the *lowest* inlier rows
    (i.e. the highest pixel y that still belongs to the mask top edge).
    For the lower boundary we want the highest row values (furthest down).

    Returns (a, b, inlier_mask) where inlier_mask has length len(valid_cols).
    """
    is_upper = side == "upper"
    shift_pct = SHIFT_PERCENTILE_UPPER if is_upper else SHIFT_PERCENTILE_LOWER

    n = len(valid_cols)
    if n < 2:
        # Degenerate: return a horizontal line at the mean edge row.
        a = 0.0
        b = float(np.mean(edge_rows))
        return a, b, np.ones(n, dtype=bool)

    if n >= 4:
        X = valid_cols.reshape(-1, 1).astype(float)
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=ransac_threshold,
            min_samples=max(2, int(ransac_min_samples * n)),
            random_state=42,
        )
        ransac.fit(X, edge_rows.astype(float))
        a = float(ransac.estimator_.coef_[0])
        b = float(ransac.estimator_.intercept_)

        fitted = a * valid_cols.astype(float) + b
        residuals = edge_rows.astype(float) - fitted
        # Inlier selection: for upper, keep rows that are ≥ fitted (below);
        # for lower, keep rows that are ≤ fitted (above).
        if is_upper:
            inlier_mask = residuals > -ransac_threshold
        else:
            inlier_mask = residuals < ransac_threshold

        # Refit on clean inliers with a percentile shift to tighten the boundary.
        if inlier_mask.sum() >= 2:
            clean_cols = valid_cols[inlier_mask]
            clean_rows = edge_rows[inlier_mask]
            a, b = np.polyfit(clean_cols.astype(float), clean_rows.astype(float), 1)
            fitted_clean = a * clean_cols.astype(float) + b
            residuals_clean = clean_rows.astype(float) - fitted_clean
            b += np.percentile(residuals_clean, shift_pct)

            # Final inlier mask relative to the shifted line.
            fitted_all = a * valid_cols.astype(float) + b
            residuals_all = edge_rows.astype(float) - fitted_all
            if is_upper:
                inlier_mask = residuals_all > -ransac_threshold
            else:
                inlier_mask = residuals_all < ransac_threshold
    else:
        a, b = np.polyfit(valid_cols.astype(float), edge_rows.astype(float), 1)
        inlier_mask = np.ones(n, dtype=bool)

    return a, b, inlier_mask


@dataclass
class BoundaryResult:
    """Horizontal RANSAC boundary lines for a sidewalk-facing image."""
    # Line parameters:  y_boundary(col) = a * col + b
    a_upper: float = float("nan")
    b_upper: float = float("nan")
    a_lower: float = float("nan")
    b_lower: float = float("nan")

    # Per-column observations
    valid_cols: np.ndarray = field(default_factory=lambda: np.array([]))
    upper_edge_rows: np.ndarray = field(default_factory=lambda: np.array([]))
    lower_edge_rows: np.ndarray = field(default_factory=lambda: np.array([]))
    inlier_upper: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    inlier_lower: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Derived representative y values (median of inlier boundary rows,
    # evaluated at the horizontal image centre)
    y_upper_mid: float = float("nan")  # wall side  → y_wall in calculate_width
    y_lower_mid: float = float("nan")  # road side  → y_curb in calculate_width

    success: bool = False
    reason: str = ""


def find_horizontal_boundaries(
    mask: np.ndarray,
    border_margin: int = BORDER_MARGIN,
    ransac_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
    ransac_min_samples: float = RANSAC_MIN_SAMPLES,
    min_mask_area: int = MIN_MASK_AREA,
    min_row_coverage: float = MIN_ROW_COVERAGE,
) -> BoundaryResult:
    """
    Detect upper (wall side) and lower (road side) pixel boundaries of a
    sidewalk mask in a sidewalk-facing image using RANSAC line fitting.

    For each image column that contains mask pixels, we record:
      • upper_edge_row[col] — top-most mask pixel row in that column
      • lower_edge_row[col] — bottom-most mask pixel row in that column

    We then fit a line through each set of column-wise edge rows using RANSAC.

    Parameters
    ----------
    mask : bool ndarray [H, W]
        Binary sidewalk mask (True = sidewalk).
    border_margin : int
        Columns within this many pixels of the image border are skipped when
        determining whether an edge is "clipped" (touching the border).
    ransac_threshold : float
        Pixel tolerance for RANSAC inlier classification.
    ransac_min_samples : float
        Fraction of valid columns used as RANSAC min_samples.
    min_mask_area : int
        Minimum number of mask pixels; smaller masks are rejected.
    min_row_coverage : float
        Minimum fraction of image rows that must contain mask pixels.

    Returns
    -------
    BoundaryResult
        Contains fitted line parameters and representative y-values.
    """
    H, W = mask.shape
    result = BoundaryResult()

    # ── Quality checks ─────────────────────────────────────────────────────
    mask_area = int(mask.sum())
    if mask_area < min_mask_area:
        result.reason = f"mask_area_too_small ({mask_area} < {min_mask_area})"
        return result

    row_coverage = float(np.mean(mask.any(axis=1)))
    if row_coverage < min_row_coverage:
        result.reason = f"row_coverage_too_low ({row_coverage:.3f} < {min_row_coverage})"
        return result

    # ── Per-column edge extraction ─────────────────────────────────────────
    valid_cols = []
    upper_edge_rows = []   # smallest row index (top of mask)
    lower_edge_rows = []   # largest row index (bottom of mask)
    upper_clipped = []     # True if upper edge touches image top border
    lower_clipped = []     # True if lower edge touches image bottom border

    for col in range(W):
        col_mask = mask[:, col]
        rows_with_mask = np.where(col_mask)[0]
        if len(rows_with_mask) == 0:
            continue
        r_top = int(rows_with_mask[0])
        r_bot = int(rows_with_mask[-1])
        valid_cols.append(col)
        upper_edge_rows.append(r_top)
        lower_edge_rows.append(r_bot)
        upper_clipped.append(r_top < border_margin)
        lower_clipped.append(r_bot >= H - border_margin)

    if len(valid_cols) < 4:
        result.reason = f"too_few_valid_columns ({len(valid_cols)})"
        return result

    valid_cols = np.array(valid_cols, dtype=np.float64)
    upper_edge_rows = np.array(upper_edge_rows, dtype=np.float64)
    lower_edge_rows = np.array(lower_edge_rows, dtype=np.float64)
    upper_clipped = np.array(upper_clipped, dtype=bool)
    lower_clipped = np.array(lower_clipped, dtype=bool)

    # ── Filter: only use non-clipped columns for initial fitting ──────────
    upper_fit_cols = valid_cols[~upper_clipped]
    upper_fit_rows = upper_edge_rows[~upper_clipped]
    lower_fit_cols = valid_cols[~lower_clipped]
    lower_fit_rows = lower_edge_rows[~lower_clipped]

    # Fall back to all columns if too many are clipped.
    if len(upper_fit_cols) < 4:
        upper_fit_cols = valid_cols
        upper_fit_rows = upper_edge_rows
    if len(lower_fit_cols) < 4:
        lower_fit_cols = valid_cols
        lower_fit_rows = lower_edge_rows

    # ── RANSAC fitting ────────────────────────────────────────────────────
    a_U, b_U, inlier_U = _fit_boundary_line(
        upper_fit_cols, upper_fit_rows, "upper", ransac_threshold, ransac_min_samples
    )
    a_L, b_L, inlier_L = _fit_boundary_line(
        lower_fit_cols, lower_fit_rows, "lower", ransac_threshold, ransac_min_samples
    )

    # ── Extrapolate clipped edges using the fitted lines ──────────────────
    upper_edge_rows_final = upper_edge_rows.copy()
    lower_edge_rows_final = lower_edge_rows.copy()
    for i, col in enumerate(valid_cols):
        if upper_clipped[i]:
            upper_edge_rows_final[i] = a_U * col + b_U
        if lower_clipped[i]:
            lower_edge_rows_final[i] = a_L * col + b_L

    # ── Representative y values at horizontal image centre ────────────────
    col_mid = float(W / 2.0)
    y_upper_mid = a_U * col_mid + b_U
    y_lower_mid = a_L * col_mid + b_L

    # Map inlier masks back to the full valid_cols array for storage.
    inlier_upper_full = np.zeros(len(valid_cols), dtype=bool)
    inlier_lower_full = np.zeros(len(valid_cols), dtype=bool)
    # upper fit was on upper_fit_cols — find which indices correspond.
    upper_fit_indices = np.where(~upper_clipped)[0] if (~upper_clipped).any() else np.arange(len(valid_cols))
    if len(upper_fit_indices) == len(inlier_U):
        inlier_upper_full[upper_fit_indices] = inlier_U
    else:
        inlier_upper_full[:] = True  # fallback
    lower_fit_indices = np.where(~lower_clipped)[0] if (~lower_clipped).any() else np.arange(len(valid_cols))
    if len(lower_fit_indices) == len(inlier_L):
        inlier_lower_full[lower_fit_indices] = inlier_L
    else:
        inlier_lower_full[:] = True  # fallback

    result.a_upper = a_U
    result.b_upper = b_U
    result.a_lower = a_L
    result.b_lower = b_L
    result.valid_cols = valid_cols
    result.upper_edge_rows = upper_edge_rows_final
    result.lower_edge_rows = lower_edge_rows_final
    result.inlier_upper = inlier_upper_full
    result.inlier_lower = inlier_lower_full
    result.y_upper_mid = y_upper_mid
    result.y_lower_mid = y_lower_mid
    result.success = True
    result.reason = "ok"
    return result


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Camera-geometry projection (from manual_width_estimation.py)

def calculate_width(
    y_wall: float,
    y_curb: float,
    image_height: int,
    fov_deg: float = FOV_DEG,
    pitch_deg: float = PITCH_DEG,
    cam_height: float = CAM_HEIGHT_M,
) -> tuple[float, float, float]:
    """
    Calculate physical sidewalk width from pixel boundary positions.

    Parameters
    ----------
    y_wall : float
        Pixel row of the upper boundary (building/wall side).
    y_curb : float
        Pixel row of the lower boundary (curb/road side).
    image_height : int
        Image height in pixels.
    fov_deg : float
        Vertical field-of-view in degrees.
    pitch_deg : float
        Camera downward pitch in degrees (positive = looking down).
    cam_height : float
        Camera height above ground in metres.

    Returns
    -------
    (width_m, z_wall, z_curb)
        Estimated sidewalk width and ground distances to each boundary.
    """
    fov_rad = math.radians(fov_deg)
    pitch_rad = math.radians(pitch_deg)

    f_y = image_height / (2 * math.tan(fov_rad / 2))
    c_y = image_height / 2.0

    def get_gamma(y: float) -> float:
        return math.atan((y - c_y) / f_y)

    def get_z(y: float) -> float:
        gamma_y = get_gamma(y)
        downward_pitch = abs(pitch_rad)
        angle = downward_pitch + gamma_y
        if angle <= 0:
            return float("inf")
        return cam_height / math.tan(angle)

    z_wall = get_z(y_wall)
    z_curb = get_z(y_curb)
    width = z_wall - z_curb
    return width, z_wall, z_curb


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Per-image processing

@dataclass
class WidthEstimationResult:
    """Full result for one image."""
    point_id: str
    direction: str
    image_blob: str
    mask_blob: str | None
    status: str  # "ok", "no_mask", "boundary_fail", "geometry_fail"
    reason: str

    # Boundary detection outputs
    y_wall: float = float("nan")   # upper boundary row (wall side)
    y_curb: float = float("nan")   # lower boundary row (road side)
    image_height: int = 0

    # Projection outputs
    width_m: float | None = None
    z_wall: float | None = None
    z_curb: float | None = None

    # Boundary model for visualisation
    boundary: BoundaryResult | None = None

    # Camera parameters used
    fov_deg: float = FOV_DEG
    pitch_deg: float = PITCH_DEG
    cam_height: float = CAM_HEIGHT_M


def process_image(
    gcs: GCSClient,
    row: dict[str, Any],
    masks_root: str = MASKS_ROOT,
    fov_deg: float = FOV_DEG,
    pitch_deg: float = PITCH_DEG,
    cam_height: float = CAM_HEIGHT_M,
    min_mask_area: int = MIN_MASK_AREA,
    min_row_coverage: float = MIN_ROW_COVERAGE,
    ransac_threshold: float = RANSAC_RESIDUAL_THRESHOLD,
    ransac_min_samples: float = RANSAC_MIN_SAMPLES,
    border_margin: int = BORDER_MARGIN,
) -> WidthEstimationResult:
    """
    End-to-end processing for a single manifest row:
      1. Download image + sidewalk mask from GCS.
      2. Detect upper/lower boundaries with RANSAC.
      3. Project boundaries to physical width via camera geometry.

    Parameters
    ----------
    gcs : GCSClient
    row : dict
        A manifest row dict (as returned by load_manifest()).
    masks_root : str
        GCS path prefix for segmentation masks.
    fov_deg, pitch_deg, cam_height : float
        Camera intrinsics/extrinsics.

    Returns
    -------
    WidthEstimationResult
    """
    parsed = row.get("parsed", {})
    point_id = parsed.get("point_id", "?")
    direction = parsed.get("direction", "?")
    image_blob = row.get("blob_name", "")

    base = WidthEstimationResult(
        point_id=point_id,
        direction=direction,
        image_blob=image_blob,
        mask_blob=None,
        status="ok",
        reason="ok",
        fov_deg=fov_deg,
        pitch_deg=pitch_deg,
        cam_height=cam_height,
    )

    # ── Step 1: load image + mask ─────────────────────────────────────────
    try:
        img, mask, masks_prefix, info = load_image_and_mask(gcs, row, masks_root)
    except Exception as exc:
        base.status = "load_fail"
        base.reason = f"load_fail: {exc}"
        return base

    base.mask_blob = info.get("mask_blob")
    base.image_height = img.shape[0]

    if mask is None:
        base.status = "no_mask"
        base.reason = f"no sidewalk mask found under {masks_prefix}/sidewalk/"
        return base

    # ── Step 2: detect boundaries ─────────────────────────────────────────
    boundary = find_horizontal_boundaries(
        mask,
        border_margin=border_margin,
        ransac_threshold=ransac_threshold,
        ransac_min_samples=ransac_min_samples,
        min_mask_area=min_mask_area,
        min_row_coverage=min_row_coverage,
    )
    base.boundary = boundary

    if not boundary.success:
        base.status = "boundary_fail"
        base.reason = f"boundary detection failed: {boundary.reason}"
        return base

    base.y_wall = boundary.y_upper_mid
    base.y_curb = boundary.y_lower_mid

    # Sanity: upper must be above lower (smaller y = higher in image)
    if base.y_wall >= base.y_curb:
        base.status = "geometry_fail"
        base.reason = (
            f"upper boundary (y={base.y_wall:.1f}) is not above "
            f"lower boundary (y={base.y_curb:.1f})"
        )
        return base

    # ── Step 3: project to physical width ─────────────────────────────────
    try:
        width, z_wall, z_curb = calculate_width(
            y_wall=base.y_wall,
            y_curb=base.y_curb,
            image_height=base.image_height,
            fov_deg=fov_deg,
            pitch_deg=pitch_deg,
            cam_height=cam_height,
        )
    except Exception as exc:
        base.status = "geometry_fail"
        base.reason = f"geometry projection failed: {exc}"
        return base

    if math.isinf(z_wall) or z_wall < 0 or math.isinf(z_curb) or z_curb < 0:
        base.status = "geometry_fail"
        base.reason = (
            f"projection out of range: z_wall={z_wall:.2f} z_curb={z_curb:.2f}"
        )
        return base

    if width <= 0:
        base.status = "geometry_fail"
        base.reason = f"non-positive width ({width:.3f} m)"
        return base

    base.width_m = width
    base.z_wall = z_wall
    base.z_curb = z_curb
    return base


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Visualisation

def visualize_boundaries(
    image_rgb: np.ndarray,
    boundary: BoundaryResult,
    result: WidthEstimationResult,
    out_path: str | None = None,
) -> np.ndarray:
    """
    Draw the fitted upper/lower RANSAC boundary lines on the image.

    Upper boundary → green (wall side)
    Lower boundary → red   (road / curb side)

    Parameters
    ----------
    image_rgb : np.ndarray  (H, W, 3)
    boundary  : BoundaryResult
    result    : WidthEstimationResult
    out_path  : optional file path to save the overlay

    Returns
    -------
    Annotated BGR image.
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    if boundary.success:
        # Draw boundary lines across the full image width
        for col in range(W):
            r_upper = int(round(boundary.a_upper * col + boundary.b_upper))
            r_lower = int(round(boundary.a_lower * col + boundary.b_lower))
            if 0 <= r_upper < H:
                img_bgr[max(0, r_upper - 1): min(H, r_upper + 2), col] = (0, 255, 0)
            if 0 <= r_lower < H:
                img_bgr[max(0, r_lower - 1): min(H, r_lower + 2), col] = (0, 0, 255)

        # Draw inlier scatter points
        for i, col in enumerate(boundary.valid_cols):
            col_i = int(col)
            if boundary.inlier_upper[i]:
                r = int(boundary.upper_edge_rows[i])
                cv2.circle(img_bgr, (col_i, r), 2, (0, 200, 0), -1)
            if boundary.inlier_lower[i]:
                r = int(boundary.lower_edge_rows[i])
                cv2.circle(img_bgr, (col_i, r), 2, (0, 0, 200), -1)

    # Annotation text
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"point={result.point_id}  dir={result.direction}",
        f"status={result.status}",
    ]
    if result.width_m is not None:
        lines += [
            f"y_wall={result.y_wall:.1f}px  z_wall={result.z_wall:.2f}m",
            f"y_curb={result.y_curb:.1f}px  z_curb={result.z_curb:.2f}m",
            f"WIDTH = {result.width_m:.3f} m",
        ]
    else:
        lines.append(result.reason[:80])

    y_text = 22
    for line in lines:
        cv2.putText(img_bgr, line, (10, y_text), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_bgr, line, (10, y_text), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += 22

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, img_bgr)

    return img_bgr


def plot_boundary_debug(
    image_rgb: np.ndarray,
    boundary: BoundaryResult,
    result: WidthEstimationResult,
    out_path: str | None = None,
) -> None:
    """
    Multi-panel matplotlib debug figure:
      Left  – original image with mask overlay + boundary lines
      Right – column-wise edge scatter + fitted lines
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: image + mask + lines
    H, W = image_rgb.shape[:2]
    overlay = image_rgb.copy()

    # Find mask from boundary inlier data (only used for display)
    ax = axes[0]
    ax.imshow(overlay)
    ax.set_title(
        f"point={result.point_id}  dir={result.direction}\n"
        f"width={result.width_m:.3f}m" if result.width_m else f"status={result.status}"
    )

    if boundary.success:
        cols_line = np.arange(W)
        upper_line = boundary.a_upper * cols_line + boundary.b_upper
        lower_line = boundary.a_lower * cols_line + boundary.b_lower
        ax.plot(cols_line, upper_line, color="lime", linewidth=2, label="Upper (wall)")
        ax.plot(cols_line, lower_line, color="red", linewidth=2, label="Lower (curb)")
        ax.axvline(W / 2, color="yellow", linestyle="--", alpha=0.6, label="Image centre")
        ax.scatter(
            boundary.valid_cols[boundary.inlier_upper],
            boundary.upper_edge_rows[boundary.inlier_upper],
            s=4, color="lime", alpha=0.6,
        )
        ax.scatter(
            boundary.valid_cols[boundary.inlier_lower],
            boundary.lower_edge_rows[boundary.inlier_lower],
            s=4, color="red", alpha=0.6,
        )
        # Outliers in dimmer colours
        ax.scatter(
            boundary.valid_cols[~boundary.inlier_upper],
            boundary.upper_edge_rows[~boundary.inlier_upper],
            s=4, color="darkgreen", alpha=0.3,
        )
        ax.scatter(
            boundary.valid_cols[~boundary.inlier_lower],
            boundary.lower_edge_rows[~boundary.inlier_lower],
            s=4, color="darkred", alpha=0.3,
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Panel 2: column-wise edge scatter + lines
    ax2 = axes[1]
    if boundary.success:
        ax2.scatter(
            boundary.valid_cols[boundary.inlier_upper],
            boundary.upper_edge_rows[boundary.inlier_upper],
            s=8, color="lime", label="Upper inliers",
        )
        ax2.scatter(
            boundary.valid_cols[boundary.inlier_lower],
            boundary.lower_edge_rows[boundary.inlier_lower],
            s=8, color="red", label="Lower inliers",
        )
        ax2.scatter(
            boundary.valid_cols[~boundary.inlier_upper],
            boundary.upper_edge_rows[~boundary.inlier_upper],
            s=4, color="darkgreen", alpha=0.3, label="Upper outliers",
        )
        ax2.scatter(
            boundary.valid_cols[~boundary.inlier_lower],
            boundary.lower_edge_rows[~boundary.inlier_lower],
            s=4, color="darkred", alpha=0.3, label="Lower outliers",
        )
        cols_line = np.linspace(0, W, 200)
        ax2.plot(cols_line, boundary.a_upper * cols_line + boundary.b_upper,
                 color="lime", linewidth=2)
        ax2.plot(cols_line, boundary.a_lower * cols_line + boundary.b_lower,
                 color="red", linewidth=2)
        ax2.axvline(W / 2, color="yellow", linestyle="--", alpha=0.6, label="Image centre")
        ax2.set_xlabel("Column (px)")
        ax2.set_ylabel("Row (px)")
        ax2.set_title("Column-wise boundary edge scatter")
        ax2.invert_yaxis()
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, f"Boundary failed:\n{boundary.reason}",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Batch runner

def run_batch(
    manifest_path: str,
    point_ids: list[str],
    directions: list[str],
    masks_root: str = MASKS_ROOT,
    fov_deg: float = FOV_DEG,
    pitch_deg: float = PITCH_DEG,
    cam_height: float = CAM_HEIGHT_M,
    out_dir: str = OUTPUT_DIR,
    save_visualizations: bool = SAVE_VISUALIZATIONS,
    verbose: bool = True,
) -> list[WidthEstimationResult]:
    """
    Process a list of point IDs from a GCS manifest and estimate sidewalk widths.

    Returns a list of WidthEstimationResult objects (one per image).
    A CSV summary is also written to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Set up GCS ────────────────────────────────────────────────────────
    bucket_name, _ = parse_gcs_path(manifest_path)
    bucket_name = bucket_name or GCS_BUCKET_NAME
    if not bucket_name:
        raise ValueError(
            "Cannot determine GCS bucket. Set GCS_BUCKET_NAME in .env or "
            "use a gs:// or HTTPS GCS URL for --manifest."
        )
    gcs = GCSClient(GCP_PROJECT_ID, bucket_name)

    # ── Load manifest ─────────────────────────────────────────────────────
    manifest = load_manifest(manifest_path, gcs)
    point_ids_norm = [normalize_point_id(p) for p in point_ids]

    # ── Collect manifest rows to process ─────────────────────────────────
    tasks: list[dict[str, Any]] = []
    for pid in point_ids_norm:
        dirs = manifest.get(pid, {})
        for direction in directions:
            row = dirs.get(direction)
            if row is None:
                if verbose:
                    print(f"  [SKIP] point={pid} dir={direction}: not in manifest")
                continue
            tasks.append(row)

    print(f"\nProcessing {len(tasks)} images for {len(point_ids_norm)} points × "
          f"{len(directions)} direction(s).")

    # ── Process each image ────────────────────────────────────────────────
    results: list[WidthEstimationResult] = []
    for i, row in enumerate(tasks, start=1):
        parsed = row.get("parsed", {})
        pid = parsed.get("point_id", "?")
        direction = parsed.get("direction", "?")
        print(f"[{i}/{len(tasks)}] point={pid}  dir={direction}  blob={row.get('blob_name', '')}")

        # We need the image for visualisation even if we fail, but we don't
        # want to download it twice.  Download lazily in process_image.
        result = process_image(
            gcs=gcs,
            row=row,
            masks_root=masks_root,
            fov_deg=fov_deg,
            pitch_deg=pitch_deg,
            cam_height=cam_height,
        )
        results.append(result)

        if verbose:
            if result.width_m is not None:
                print(
                    f"  -> y_wall={result.y_wall:.1f}  y_curb={result.y_curb:.1f}  "
                    f"width={result.width_m:.3f}m  z_wall={result.z_wall:.2f}m  z_curb={result.z_curb:.2f}m"
                )
            else:
                print(f"  -> FAILED  [{result.status}]  {result.reason}")

        # Visualisation: download the image again for the overlay.
        if save_visualizations:
            try:
                img_bytes = gcs.download_as_bytes(row["blob_name"])
                img_rgb = bytes_to_image(img_bytes)
                stem = f"point{pid}_{direction}"
                vis_path = os.path.join(out_dir, f"{stem}_overlay.jpg")
                debug_path = os.path.join(out_dir, f"{stem}_debug.png")
                visualize_boundaries(img_rgb, result.boundary or BoundaryResult(), result, out_path=vis_path)
                if result.boundary is not None:
                    plot_boundary_debug(img_rgb, result.boundary, result, out_path=debug_path)
            except Exception as exc:
                print(f"  [WARN] visualisation failed for {pid}/{direction}: {exc}")

    # ── Save CSV summary ──────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "ransac_width_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "point_id", "direction", "status", "reason",
            "width_m", "z_wall_m", "z_curb_m",
            "y_wall_px", "y_curb_px", "image_height_px",
            "fov_deg", "pitch_deg", "cam_height_m",
            "image_blob", "mask_blob",
            "processed_utc",
        ])
        now_utc = datetime.now(timezone.utc).isoformat()
        for r in results:
            writer.writerow([
                r.point_id, r.direction, r.status, r.reason,
                f"{r.width_m:.4f}" if r.width_m is not None else "",
                f"{r.z_wall:.4f}" if r.z_wall is not None else "",
                f"{r.z_curb:.4f}" if r.z_curb is not None else "",
                f"{r.y_wall:.2f}" if not math.isnan(r.y_wall) else "",
                f"{r.y_curb:.2f}" if not math.isnan(r.y_curb) else "",
                r.image_height,
                r.fov_deg, r.pitch_deg, r.cam_height,
                r.image_blob, r.mask_blob or "",
                now_utc,
            ])

    print(f"\nResults saved to {csv_path}")

    # ── Print summary stats ───────────────────────────────────────────────
    ok = [r for r in results if r.width_m is not None]
    widths = np.array([r.width_m for r in ok])
    print(f"\n--- Summary ---")
    print(f"  Total images:       {len(results)}")
    print(f"  Successful:         {len(ok)}")
    print(f"  Failed:             {len(results) - len(ok)}")
    if len(ok) > 0:
        print(f"  Width range:        {widths.min():.3f} – {widths.max():.3f} m")
        print(f"  Median width:       {np.median(widths):.3f} m")
        print(f"  Mean width:         {np.mean(widths):.3f} m")
        print(f"  Std:                {np.std(widths):.3f} m")

    return results


# ── CELL BREAK ──────────────────────────────────────────────────────────────
# ## Run
#
# Edit the constants at the top of this file (MANIFEST_CSV, POINT_IDS,
# DIRECTIONS, MASKS_ROOT, FOV_DEG, PITCH_DEG, CAM_HEIGHT_M, OUTPUT_DIR, …)
# then simply run:  python ransac_width_estimation.py

if __name__ == "__main__":
    run_batch(
        manifest_path=MANIFEST_CSV,
        point_ids=POINT_IDS,
        directions=DIRECTIONS,
        masks_root=MASKS_ROOT,
        fov_deg=FOV_DEG,
        pitch_deg=PITCH_DEG,
        cam_height=CAM_HEIGHT_M,
        out_dir=OUTPUT_DIR,
        save_visualizations=SAVE_VISUALIZATIONS,
        verbose=True,
    )
