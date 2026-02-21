"""
Google Cloud Storage utility functions for uploading and downloading images.

Uses Application Default Credentials for authentication.
Run `gcloud auth application-default login` before using this module.
"""

import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv(".env")

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")


def get_bucket():
    """Returns the GCS bucket object."""
    client = storage.Client(project=GCP_PROJECT_ID)
    return client.bucket(BUCKET_NAME)


def upload_image(local_path: str, destination_blob_name: str | None = None) -> str:
    """
    Upload an image file to the GCS bucket.

    Args:
        local_path: Path to the local image file.
        destination_blob_name: Name for the blob in GCS.
            Defaults to the local filename if not provided.

    Returns:
        The GCS URI (gs://bucket/blob) of the uploaded file.
    """
    if destination_blob_name is None:
        destination_blob_name = os.path.basename(local_path)

    bucket = get_bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
    print(f"Uploaded {local_path} -> {gcs_uri}")
    return gcs_uri


def upload_image_from_bytes(data: bytes, destination_blob_name: str, content_type: str = "image/jpeg") -> str:
    """
    Upload image bytes directly to the GCS bucket (no local file needed).

    Args:
        data: Raw image bytes.
        destination_blob_name: Name for the blob in GCS.
        content_type: MIME type of the image.

    Returns:
        The GCS URI of the uploaded file.
    """
    bucket = get_bucket()
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(data, content_type=content_type)

    gcs_uri = f"gs://{BUCKET_NAME}/{destination_blob_name}"
    print(f"Uploaded bytes -> {gcs_uri}")
    return gcs_uri


def download_image(blob_name: str, local_path: str | None = None) -> str:
    """
    Download an image from the GCS bucket to a local file.

    Args:
        blob_name: Name of the blob in GCS.
        local_path: Local path to save the file.
            Defaults to the blob name (flattened) in the current directory.

    Returns:
        The local file path where the image was saved.
    """
    if local_path is None:
        local_path = os.path.basename(blob_name)

    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

    print(f"Downloaded gs://{BUCKET_NAME}/{blob_name} -> {local_path}")
    return local_path


def list_images(prefix: str | None = None) -> list[str]:
    """
    List all blobs in the bucket, optionally filtered by a prefix.

    Args:
        prefix: Only list blobs whose names start with this prefix.
            Useful for listing images in a specific "folder", e.g. "streetview/".

    Returns:
        A list of blob names.
    """
    bucket = get_bucket()
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]


def delete_image(blob_name: str) -> None:
    """Delete a blob from the bucket."""
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    blob.delete()
    print(f"Deleted gs://{BUCKET_NAME}/{blob_name}")
