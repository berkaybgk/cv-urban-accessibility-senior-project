"""
Street View helpers
───────────────────
Thin wrappers around the Google Street View Static API for:
  • checking panorama availability  (metadata endpoint – free)
  • validating snap distance        (actual pano vs requested location)
  • fetching JPEG images by pano_id (guarantees the validated panorama)
"""

import math
from dataclasses import dataclass
from typing import Optional

import requests

from config import (
    GOOGLE_MAPS_API_KEY,
    STREETVIEW_URL,
    STREETVIEW_META_URL,
    SV_SIZE,
    SV_FOV,
    SV_PITCH,
    SV_MAX_SNAP_DISTANCE_M,
)


# ── Geometry ────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6_371_000
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Metadata / availability ─────────────────────────────────────────────────

@dataclass
class PanoMetadata:
    """Parsed result of a Street View metadata query."""
    pano_id: str
    pano_lat: float
    pano_lng: float
    capture_date: str
    snap_distance_m: float
    raw: dict


def check_availability(
    lat: float,
    lon: float,
    max_snap_m: float = SV_MAX_SNAP_DISTANCE_M,
) -> Optional[PanoMetadata]:
    """
    Query the Street View *metadata* endpoint (free, no image quota used).

    Returns a ``PanoMetadata`` when a panorama exists **and** its actual
    location is within *max_snap_m* of the requested coordinate.
    Returns ``None`` if there is no coverage or the snap is too far.
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

    pano_loc = data.get("location", {})
    pano_lat = pano_loc.get("lat", lat)
    pano_lng = pano_loc.get("lng", lon)

    snap_dist = _haversine_m(lat, lon, pano_lat, pano_lng)

    if snap_dist > max_snap_m:
        return None

    return PanoMetadata(
        pano_id=data.get("pano_id", ""),
        pano_lat=pano_lat,
        pano_lng=pano_lng,
        capture_date=data.get("date", ""),
        snap_distance_m=round(snap_dist, 2),
        raw=data,
    )


# ── Image fetching ──────────────────────────────────────────────────────────

def fetch_image(
    heading: float,
    *,
    pano_id: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    size: str = SV_SIZE,
    fov: int = SV_FOV,
    pitch: int = SV_PITCH,
) -> Optional[bytes]:
    """
    Fetch a Street View JPEG image.

    Prefer passing *pano_id* (from ``check_availability``) so that the
    image comes from the exact panorama you validated.  Falls back to
    *lat*/*lon* if no pano_id is given.

    Returns the raw image bytes, or ``None`` on failure.
    """
    params = {
        "size": size,
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "key": GOOGLE_MAPS_API_KEY,
    }
    if pano_id:
        params["pano"] = pano_id
    elif lat is not None and lon is not None:
        params["location"] = f"{lat},{lon}"
    else:
        raise ValueError("Provide either pano_id or both lat and lon")

    resp = requests.get(STREETVIEW_URL, params=params, timeout=30)
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image"):
        return resp.content
    loc = pano_id or f"({lat}, {lon})"
    print(f"  ⚠ Street View request failed ({resp.status_code}) for {loc}")
    return None
