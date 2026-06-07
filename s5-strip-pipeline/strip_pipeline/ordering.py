"""Heading-aware point ordering and tile-sequence construction.

The notebook hard-coded ``POINT_IDS`` order and the forward/backward roles in
``strip_sequence`` (cell 10). That silently breaks when points are given in an
order that does not follow the street direction. Here we derive the walking
order from coordinates and orient it with the ``heading`` value so that the
"forward" camera always looks ahead along travel — no forward/backward
confusion regardless of selection order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest_gcs import normalize_point_id


@dataclass
class TileSpec:
    side_strip: str
    point_id: str
    direction: str
    selected_side: str
    method: str
    flip_180: bool

    @property
    def tile_id(self) -> str:
        return f"{self.side_strip}:{self.point_id}:{self.direction}:{self.selected_side}:{self.method}"


def _circular_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference a-b in degrees, in (-180, 180]."""
    return ((a - b + 180.0) % 360.0) - 180.0


def _bearing_to_vec(bearing_deg: float) -> np.ndarray:
    """Compass bearing -> (east, north) unit vector. 0=N, 90=E."""
    rad = math.radians(bearing_deg)
    return np.array([math.sin(rad), math.cos(rad)], dtype=float)


def _point_latlon(manifest: dict[str, Any], point_id: str) -> tuple[float, float] | None:
    dirs = manifest.get(point_id, {})
    for row in dirs.values():
        parsed = row.get("parsed", {})
        try:
            return float(parsed.get("lat")), float(parsed.get("lon"))
        except (TypeError, ValueError):
            try:
                return float(row.get("latitude")), float(row.get("longitude"))
            except (TypeError, ValueError):
                continue
    return None


def _forward_bearing(manifest: dict[str, Any], point_id: str) -> float | None:
    """Bearing the 'forward' camera looks along (== street bearing)."""
    dirs = manifest.get(point_id, {})
    fwd = dirs.get("forward")
    if fwd is not None:
        for key in ("street_bearing", "heading"):
            val = fwd.get(key) or fwd.get("parsed", {}).get(key if key != "street_bearing" else "")
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        try:
            return float(fwd["parsed"]["heading"])
        except (KeyError, TypeError, ValueError):
            pass
    # Fall back to any row's street_bearing.
    for row in dirs.values():
        try:
            return float(row.get("street_bearing"))
        except (TypeError, ValueError):
            continue
    return None


def order_points(manifest: dict[str, Any], point_ids: list[str]) -> list[str]:
    """Return point ids sorted back-to-front along the heading-oriented street axis."""
    ids = [normalize_point_id(p) for p in point_ids]
    ids = [p for p in ids if p in manifest]
    if len(ids) <= 1:
        return ids

    coords = {p: _point_latlon(manifest, p) for p in ids}
    usable = [p for p in ids if coords[p] is not None]
    if len(usable) <= 1:
        return ids

    lat0 = float(np.mean([coords[p][0] for p in usable]))
    lon0 = float(np.mean([coords[p][1] for p in usable]))
    cos_lat = math.cos(math.radians(lat0))
    # Local east/north meters (equirectangular approximation).
    xy = {
        p: np.array([
            (coords[p][1] - lon0) * cos_lat * 111320.0,
            (coords[p][0] - lat0) * 110540.0,
        ])
        for p in usable
    }

    pts = np.array([xy[p] for p in usable])
    centered = pts - pts.mean(axis=0)

    # Mean forward heading vector for orienting the axis.
    bearings = [b for b in (_forward_bearing(manifest, p) for p in usable) if b is not None]
    if bearings:
        heading_vec = np.mean([_bearing_to_vec(b) for b in bearings], axis=0)
        if np.linalg.norm(heading_vec) < 1e-6:
            heading_vec = None
    else:
        heading_vec = None

    if np.allclose(centered, 0):
        axis = heading_vec if heading_vec is not None else np.array([0.0, 1.0])
    else:
        # Principal axis of the point cloud.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        if heading_vec is not None and float(np.dot(axis, heading_vec)) < 0:
            axis = -axis
        elif heading_vec is None:
            axis = axis  # arbitrary but stable

    axis = axis / (np.linalg.norm(axis) or 1.0)
    projections = {p: float(np.dot(xy[p], axis)) for p in usable}

    ordered_usable = sorted(usable, key=lambda p: projections[p])
    # Append any points that lacked coordinates at the end, preserving input order.
    leftover = [p for p in ids if p not in set(usable)]
    return ordered_usable + leftover


def strip_sequence(side_strip: str, ordered_point_ids: list[str]) -> list[TileSpec]:
    """Bottom-to-top tile specs for one side, mirroring notebook cell 10.

    ``side_strip`` is in travel frame: 'left' = left-of-travel sidewalk. Because
    points are pre-ordered so the forward camera looks ahead, literal camera
    directions match travel roles and the forward/backward roles are unambiguous.
    """
    if side_strip == "left":
        side_view = ("left", "largest", "side-view-fan")
        forward = ("forward", "left", "geometry")
        backward_bridge_side = "right"
    elif side_strip == "right":
        side_view = ("right", "largest", "side-view-fan")
        forward = ("forward", "right", "geometry")
        backward_bridge_side = "left"
    else:
        raise ValueError(f"Unknown side strip: {side_strip}")

    seq: list[TileSpec] = []
    for idx, point_id in enumerate(ordered_point_ids):
        seq.append(TileSpec(side_strip, point_id, *side_view, flip_180=False))
        seq.append(TileSpec(side_strip, point_id, *forward, flip_180=False))
        if idx + 1 < len(ordered_point_ids):
            next_point = ordered_point_ids[idx + 1]
            seq.append(TileSpec(side_strip, next_point, "backward", backward_bridge_side, "geometry", flip_180=True))
    return seq
