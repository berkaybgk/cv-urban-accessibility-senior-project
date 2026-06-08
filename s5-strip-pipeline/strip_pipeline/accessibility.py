"""Shared accessibility scoring + GeoJSON feature builders.

Single source of truth for how a strip's ``*_METRICS.json`` becomes a scored
map segment. Used by:

  * ``strip_pipeline/outputs.py`` — auto-emits one
    ``*_sidewalk_strip_ACCESSIBILITY.geojson`` per strip while the pipeline runs.
  * ``build_accessibility_geojson.py`` — batch-converts many METRICS.json files.

Both consume the same functions so per-strip files and batch files always agree.

This module is intentionally dependency-free (stdlib only) so the batch CLI can
load it without importing the heavy ``strip_pipeline`` package (cv2/torch/etc).

Each segment carries **two** scores so a single uploaded GeoJSON can drive both
the web app's Walkability map and Wheelchair map via a metric toggle:

  * ``walkability_score`` — graded in [0, 1] from the count of >=60 cm clear-width
    drops. Fewer drops => more walkable => higher (greener) score.
  * ``wheelchair_score``  — binary 1.0 / 0.0 from ``wheelchair_passable_65cm``.

``score`` mirrors ``walkability_score`` as the default metric for backward
compatibility with files/readers that only know about a single ``score`` field.
"""
from __future__ import annotations

from typing import Any


def walkability_score(metrics: dict[str, Any]) -> float:
    """Map the count of >=60 cm width-drop events to a [0, 1] walkability score.

    With the app's thresholds (green >= 0.66, yellow 0.33-0.66, red < 0.33):
        0 drops -> 1.00 (green)
        1 drop  -> 0.50 (yellow)
        2 drops -> 0.33 (yellow/edge)
        3 drops -> 0.25 (red)
    """
    count = int(metrics.get("width_drop_60cm_count", 0) or 0)
    return round(1.0 / (1.0 + max(0, count)), 4)


def wheelchair_score(metrics: dict[str, Any]) -> float:
    """Binary 1.0 / 0.0 from the wheelchair-passability flag (>=65 cm clear)."""
    return 1.0 if bool(metrics.get("wheelchair_passable_65cm", False)) else 0.0


def has_drawable_geometry(metrics: dict[str, Any]) -> bool:
    """True if metrics embed a LineString geometry with >= 2 coordinates."""
    geom = metrics.get("geometry") or {}
    coords = geom.get("coordinates") if isinstance(geom, dict) else None
    return (
        isinstance(geom, dict)
        and geom.get("type") == "LineString"
        and isinstance(coords, list)
        and len(coords) >= 2
    )


def accessibility_feature(metrics: dict[str, Any], seg_id: str) -> dict[str, Any]:
    """Build one GeoJSON LineString feature carrying both scores + key metrics."""
    walk = walkability_score(metrics)
    wheel = wheelchair_score(metrics)
    return {
        "type": "Feature",
        "properties": {
            "id": seg_id,
            "score": walk,  # default metric (back-compat)
            "walkability_score": walk,
            "wheelchair_score": wheel,
            "min_clear_width_m": metrics.get("min_clear_width_m"),
            "width_drop_60cm_count": metrics.get("width_drop_60cm_count"),
            "wheelchair_passable_65cm": metrics.get("wheelchair_passable_65cm"),
            "ada_accessible_90cm": metrics.get("ada_accessible_90cm"),
            "strip_length_m": metrics.get("strip_length_m"),
        },
        "geometry": metrics.get("geometry"),
    }
