"""
Google Cloud Storage utilities (in-memory only, no temp files).
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


class GCSClient:
    """Thin wrapper around google.cloud.storage for in-memory I/O."""

    def __init__(self, project_id: str, bucket_name: str):
        from google.cloud import storage
        self._client = storage.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def download_as_bytes(self, blob_name: str) -> bytes:
        return self._bucket.blob(blob_name).download_as_bytes()

    def upload_bytes(self, data: bytes, blob_name: str,
                     content_type: str = "image/png") -> str:
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{blob_name}"

    def list_blobs(self, prefix: str) -> list[str]:
        return [b.name for b in self._bucket.list_blobs(prefix=prefix)]


def bytes_to_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to an RGB numpy array."""
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def bytes_to_mask(data: bytes) -> np.ndarray:
    """Decode grayscale PNG bytes to a boolean mask."""
    mask = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return mask > 127


def image_array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a numpy image array to PNG bytes."""
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
