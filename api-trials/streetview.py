"""
Street View helpers
───────────────────
Thin wrappers around the Google Street View Static API for:
  • checking panorama availability (metadata endpoint – free)
  • fetching JPEG images at a given location + heading
"""

from typing import Optional

import requests

from config import (
    GOOGLE_MAPS_API_KEY,
    STREETVIEW_URL,
    STREETVIEW_META_URL,
    SV_SIZE,
    SV_FOV,
    SV_PITCH,
)


def check_availability(lat: float, lon: float) -> Optional[dict]:
    """
    Query the Street View *metadata* endpoint (free, no image quota used).

    Returns the metadata dict (contains ``pano_id``, ``date``, ``location``,
    etc.) when imagery is available, or ``None`` otherwise.
    """
    params = {
        "location": f"{lat},{lon}",
        "key": GOOGLE_MAPS_API_KEY,
    }
    resp = requests.get(STREETVIEW_META_URL, params=params, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("status") == "OK":
            return data
    return None


def fetch_image(
    lat: float,
    lon: float,
    heading: float,
    size: str = SV_SIZE,
    fov: int = SV_FOV,
    pitch: int = SV_PITCH,
) -> Optional[bytes]:
    """
    Fetch a Street View JPEG image.

    Returns the raw image bytes, or ``None`` on failure.
    """
    params = {
        "size": size,
        "location": f"{lat},{lon}",
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "key": GOOGLE_MAPS_API_KEY,
    }
    resp = requests.get(STREETVIEW_URL, params=params, timeout=30)
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image"):
        return resp.content
    print(f"  ⚠ Street View request failed ({resp.status_code}) for ({lat}, {lon})")
    return None
