"""
Street View helpers v2
──────────────────────
Wrappers around the Google Street View Static API:
  - check_availability: metadata endpoint (free) — returns pano_id AND
    the actual panorama location (lat/lng), not just the requested one.
  - fetch_image: fetch JPEG by pano_id (not by lat/lon) so we always get
    exactly the panorama we checked.
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
    Query the Street View metadata endpoint (free, no image quota used).

    Returns a dict with:
      - pano_id   : str   (unique panorama identifier)
      - pano_lat  : float (actual panorama latitude)
      - pano_lng  : float (actual panorama longitude)
      - date      : str   (capture date, e.g. "2023-10")
    or ``None`` when no imagery is available near (lat, lon).
    """
    params = {
        "location": f"{lat},{lon}",
        "key": GOOGLE_MAPS_API_KEY,
    }
    resp = requests.get(STREETVIEW_META_URL, params=params, timeout=15)
    if resp.status_code != 200:
        return None

    data = resp.json()
    if data.get("status") != "OK":
        return None

    location = data.get("location", {})
    return {
        "pano_id":  data.get("pano_id", ""),
        "pano_lat": location.get("lat", lat),
        "pano_lng": location.get("lng", lon),
        "date":     data.get("date", ""),
    }


def fetch_image(
    pano_id: str,
    heading: float,
    size: str = SV_SIZE,
    fov: int = SV_FOV,
    pitch: int = SV_PITCH,
) -> Optional[bytes]:
    """
    Fetch a Street View JPEG by panorama ID.

    Using ``pano=`` instead of ``location=`` guarantees we get the exact
    panorama that was checked with ``check_availability``, avoiding the
    50 m snap-to-nearest behaviour of the location parameter.

    Returns the raw image bytes, or ``None`` on failure.
    """
    params = {
        "size": size,
        "pano": pano_id,
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "key": GOOGLE_MAPS_API_KEY,
    }
    resp = requests.get(STREETVIEW_URL, params=params, timeout=30)
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image"):
        return resp.content
    print(f"  WARNING: Street View fetch failed ({resp.status_code}) "
          f"for pano_id={pano_id}")
    return None
