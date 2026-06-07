"""Manifest parsing and GCS access. Ported from cell 4 of the notebook.

Globals (``GCP_PROJECT_ID``, ``GCS_BUCKET_NAME``, ``manifest``, ``gcs``) are
replaced by explicit parameters / a :class:`GCSClient` instance.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from google.cloud import storage
from PIL import Image

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


def download_gcs_bytes(bucket_name: str, blob_name: str, project_id: str = "") -> bytes:
    return storage.Client(project=project_id or None).bucket(bucket_name).blob(blob_name).download_as_bytes()


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


def open_manifest_text(path: str | Path, project_id: str = "", bucket_name: str = ""):
    """Open a manifest from a local path, gs:// URI, HTTPS GCS URL, or bucket blob path."""
    src_bucket, blob_name = parse_gcs_path(path)

    if src_bucket:
        data = download_gcs_bytes(src_bucket, blob_name, project_id)
        return io.StringIO(data.decode("utf-8-sig"))

    local_path = Path(path).expanduser()
    if local_path.exists():
        return local_path.open(newline="")

    if bucket_name:
        data = download_gcs_bytes(bucket_name, blob_name, project_id)
        return io.StringIO(data.decode("utf-8-sig"))

    raise FileNotFoundError(f"Manifest not found locally and no bucket configured: {path}")


def load_manifest(
    path: str | Path, project_id: str = "", bucket_name: str = ""
) -> dict[str, dict[str, dict[str, Any]]]:
    rows_by_point: dict[str, dict[str, dict[str, Any]]] = {}
    with open_manifest_text(path, project_id=project_id, bucket_name=bucket_name) as f:
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


class GCSClient:
    def __init__(self, project_id: str, bucket_name: str):
        self._client = storage.Client(project=project_id or None)
        self._bucket = self._client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def download_as_bytes(self, blob_name: str) -> bytes:
        return self._bucket.blob(blob_name).download_as_bytes()

    def list_blobs(self, prefix: str) -> list[str]:
        return [b.name for b in self._bucket.list_blobs(prefix=prefix)]

    def upload_bytes(self, blob_name: str, data: bytes, content_type: str = "image/png") -> str:
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{blob_name}"


def bytes_to_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def bytes_to_mask(data: bytes) -> np.ndarray:
    mask = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return (mask > 127).astype(bool)


def resolve_masks_prefix(gcs: GCSClient, masks_root: str, parsed: dict[str, str]) -> tuple[str, str]:
    direction = parsed["direction"]
    point_id = parsed["point_id"]
    preferred_coord = parsed["coordinate_folder"]
    preferred = f'{masks_root.rstrip("/")}/{preferred_coord}/{direction}'

    exact = [b for b in gcs.list_blobs(preferred + "/") if b.endswith(".png")]
    if exact:
        return preferred, preferred_coord

    all_for_point = gcs.list_blobs(f'{masks_root.rstrip("/")}/{point_id}_')
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
            d_lat = float(m.group(1)) - float(parsed["lat"])
            d_lon = float(m.group(2)) - float(parsed["lon"])
        except (TypeError, ValueError):
            return float("inf")
        return d_lat * d_lat + d_lon * d_lon

    chosen = sorted(coord_candidates, key=lambda c: (coord_distance_sq(c), c))[0]
    return f'{masks_root.rstrip("/")}/{chosen}/{direction}', chosen


def load_individual_sidewalk_masks(gcs: GCSClient, masks_prefix: str, shape: tuple[int, int]) -> list[dict[str, Any]]:
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
        masks.append({"mask": mask, "side": side, "blob_name": blob, "area": int(mask.sum())})
    masks.sort(key=lambda m: m["area"], reverse=True)
    return masks


def load_obstacle_masks(gcs: GCSClient, masks_prefix: str, shape: tuple[int, int]) -> list[dict[str, Any]]:
    blobs = [b for b in gcs.list_blobs(masks_prefix + "/") if b.endswith(".png")]
    H, W = shape
    obstacles: list[dict[str, Any]] = []
    for blob in sorted(blobs):
        parts = blob.split("/")
        if len(parts) < 2:
            continue
        class_name = parts[-2]
        if class_name in {"sidewalk", "road"}:
            continue
        mask = bytes_to_mask(gcs.download_as_bytes(blob))
        if mask.shape != shape:
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():
            continue
        obstacles.append({"mask": mask, "class_name": class_name, "blob_name": blob, "area": int(mask.sum())})
    obstacles.sort(key=lambda m: (m["class_name"], m["blob_name"]))
    return obstacles
