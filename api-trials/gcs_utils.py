"""
Google Cloud Storage utilities
──────────────────────────────
Upload / download / list / delete blobs, with optional custom metadata.

Authentication: uses Application Default Credentials.
Run ``gcloud auth application-default login`` before first use.
"""

import os
from typing import Optional

from google.cloud import storage

from config import GCS_BUCKET_NAME, GCP_PROJECT_ID


def _get_client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT_ID)


def get_bucket() -> storage.Bucket:
    """Return the configured GCS bucket object."""
    return _get_client().bucket(GCS_BUCKET_NAME)


# ── Upload ──────────────────────────────────────────────────────────────────

def upload_bytes(
    data: bytes,
    blob_name: str,
    content_type: str = "image/jpeg",
    metadata: Optional[dict[str, str]] = None,
) -> str:
    """
    Upload raw bytes to GCS.

    Parameters
    ----------
    data : bytes
        Raw file content to upload.
    blob_name : str
        Destination path inside the bucket.
    content_type : str
        MIME type.
    metadata : dict, optional
        Custom metadata key/value pairs attached to the blob.

    Returns
    -------
    str
        The ``gs://`` URI of the uploaded blob.
    """
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    if metadata:
        blob.metadata = metadata
    blob.upload_from_string(data, content_type=content_type)
    uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    return uri


def upload_file(
    local_path: str,
    blob_name: Optional[str] = None,
    metadata: Optional[dict[str, str]] = None,
) -> str:
    """
    Upload a local file to GCS.

    Parameters
    ----------
    local_path : str
        Path to the file on disk.
    blob_name : str, optional
        Destination path in the bucket.  Defaults to the file's basename.
    metadata : dict, optional
        Custom metadata key/value pairs attached to the blob.

    Returns
    -------
    str
        The ``gs://`` URI of the uploaded blob.
    """
    if blob_name is None:
        blob_name = os.path.basename(local_path)
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    if metadata:
        blob.metadata = metadata
    blob.upload_from_filename(local_path)
    uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    return uri


# ── Download / List / Delete ────────────────────────────────────────────────

def download_file(blob_name: str, local_path: Optional[str] = None) -> str:
    """Download a blob to a local file.  Returns the local path."""
    if local_path is None:
        local_path = os.path.basename(blob_name)
    bucket = get_bucket()
    bucket.blob(blob_name).download_to_filename(local_path)
    return local_path


def list_blobs(prefix: Optional[str] = None) -> list[str]:
    """List blob names in the bucket, optionally filtered by *prefix*."""
    return [b.name for b in get_bucket().list_blobs(prefix=prefix)]


def delete_blob(blob_name: str) -> None:
    """Delete a single blob from the bucket."""
    get_bucket().blob(blob_name).delete()
