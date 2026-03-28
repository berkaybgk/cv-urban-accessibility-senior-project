"""
Street Sampler
──────────────
Download an OSM street network by **place name**, **bounding box**, or
**center-point + radius**, pick a street edge, and return evenly-spaced
sample points with lat/lon, compass bearing, and distance along the edge.

Three ways to define the area
─────────────────────────────
1. Place name  → ``graph_from_place``  (must be a polygon area, not a street)
2. Bounding box → ``graph_from_bbox``   (north, south, east, west)
3. Center + radius → ``graph_from_point`` (lat, lon, dist_m)
"""

import math
from typing import Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, Polygon

from config import (
    DEFAULT_NETWORK_TYPE,
    DEFAULT_SAMPLE_INTERVAL_M,
    DEFAULT_EDGE_INDEX,
)


# ── Geometry helpers ────────────────────────────────────────────────────────

def compute_bearing(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Initial compass bearing (0–360°) from point 1 to point 2."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)

    x = math.sin(d_lon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon))

    return math.degrees(math.atan2(x, y)) % 360


# ── Graph download helpers ──────────────────────────────────────────────────

def _graph_from_place(place: str, network_type: str):
    """Download graph using a geocodable area name (neighbourhood, district…)."""
    print(f"Downloading '{network_type}' network for place '{place}' …")
    return ox.graph_from_place(place, network_type=network_type)


def _graph_from_bbox(
    north: float, south: float, east: float, west: float,
    network_type: str,
):
    """Download graph for an explicit lat/lon bounding box."""
    print(f"Downloading '{network_type}' network for bbox "
          f"N={north}, S={south}, E={east}, W={west} …")
    return ox.graph_from_bbox(bbox=(north, south, east, west),
                              network_type=network_type)


def _graph_from_point(
    lat: float, lon: float, dist_m: float, network_type: str,
):
    """Download graph within *dist_m* metres of a centre point."""
    print(f"Downloading '{network_type}' network within {dist_m} m of "
          f"({lat}, {lon}) …")
    return ox.graph_from_point((lat, lon), dist=dist_m,
                               network_type=network_type)


def _graph_from_polygon(
    vertices: list[tuple[float, float]], network_type: str,
):
    """
    Download graph inside an arbitrary polygon.

    *vertices* is a list of (lat, lon) tuples – at least 3.
    The polygon is closed automatically.
    """
    # Shapely Polygon expects (lon, lat) = (x, y)
    ring = [(lon, lat) for lat, lon in vertices]
    poly = Polygon(ring)
    print(f"Downloading '{network_type}' network for custom polygon "
          f"({len(vertices)} vertices) …")
    return ox.graph_from_polygon(poly, network_type=network_type)


# ── Edge → sample points ───────────────────────────────────────────────────

def _interpolate_edge(edge_geom, interval_m: float, crs_metric,
                      street_name) -> pd.DataFrame:
    """Interpolate points along a single projected edge geometry."""
    # OSM edges can have name as a list (e.g. ["Bağdat Caddesi", "D010"])
    if isinstance(street_name, list):
        street_name = ", ".join(str(n) for n in street_name)
    street_name = str(street_name) if street_name else "unknown"

    edge_length = edge_geom.length
    if edge_length == 0:
        return pd.DataFrame()

    distances = np.arange(0, edge_length, interval_m)
    if distances[-1] != edge_length:
        distances = np.append(distances, edge_length)
    pts = [edge_geom.interpolate(d) for d in distances]

    transformer = Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)
    lats, lons = [], []
    for p in pts:
        lon, lat = transformer.transform(p.x, p.y)
        lats.append(lat)
        lons.append(lon)

    bearings: list[float] = []
    for i in range(len(lats)):
        if i < len(lats) - 1:
            bearings.append(
                round(compute_bearing(lats[i], lons[i],
                                      lats[i + 1], lons[i + 1]), 2)
            )
        else:
            bearings.append(bearings[-1] if bearings else 0.0)

    return pd.DataFrame({
        "latitude":         [round(la, 7) for la in lats],
        "longitude":        [round(lo, 7) for lo in lons],
        "bearing":          bearings,
        "distance_along_m": [round(d, 2) for d in distances],
        "street_name":      street_name,
    })


def _sample_edges(G, interval_m: float, edge_index: int,
                  sample_all: bool = False, label: str = "") -> pd.DataFrame:
    """
    Project graph *G*, sample points along edge(s), and return a DataFrame
    with latitude, longitude, bearing, distance_along_m, street_name.

    If *sample_all* is True, every edge in the graph is sampled.
    Otherwise only the edge at *edge_index* (sorted by length desc) is used.
    """
    G_undir = ox.convert.to_undirected(G)
    G_proj = ox.project_graph(G_undir)
    edges_proj = ox.graph_to_gdfs(G_proj, nodes=False)
    crs_metric = edges_proj.crs

    edges_proj = (edges_proj
                  .sort_values("length", ascending=False)
                  .reset_index(drop=True))

    if sample_all:
        print(f"  Sampling ALL {len(edges_proj)} edges …")
        frames = []
        for i in range(len(edges_proj)):
            geom = edges_proj.geometry.iloc[i]
            name = edges_proj.iloc[i].get("name", "unknown")
            frames.append(_interpolate_edge(geom, interval_m, crs_metric, name))
        df = pd.concat(frames, ignore_index=True)
        print(f"  Total: {len(df)} sample points across {len(edges_proj)} edges")
        return df

    # Single-edge mode
    if edge_index >= len(edges_proj):
        raise IndexError(
            f"edge_index={edge_index} but only {len(edges_proj)} edges exist"
        )
    geom = edges_proj.geometry.iloc[edge_index]
    name = edges_proj.iloc[edge_index].get("name", "unknown")
    print(f"  Street: {name}  |  length: {geom.length:.1f} m")
    return _interpolate_edge(geom, interval_m, crs_metric, name)


# ── Public API ──────────────────────────────────────────────────────────────

def sample_street(
    place: str,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
) -> pd.DataFrame:
    """Sample points along edges inside a *place name* area."""
    G = _graph_from_place(place, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all,
                         label=place)


def sample_street_bbox(
    north: float, south: float, east: float, west: float,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
) -> pd.DataFrame:
    """
    Sample points along edges inside an explicit bounding box.

    Example – Bağdat Caddesi corridor::

        sample_street_bbox(
            north=40.975, south=40.960, east=29.090, west=29.040,
        )
    """
    G = _graph_from_bbox(north, south, east, west, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all)


def sample_street_point(
    lat: float, lon: float, dist_m: float = 500,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
) -> pd.DataFrame:
    """
    Sample points along edges within *dist_m* metres of (lat, lon).

    Example – 500 m around a point on Bağdat Caddesi::

        sample_street_point(40.9667, 29.0640, dist_m=500)
    """
    G = _graph_from_point(lat, lon, dist_m, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all)


def sample_street_polygon(
    vertices: list[tuple[float, float]],
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
) -> pd.DataFrame:
    """
    Sample points along edges inside an arbitrary polygon.

    *vertices* is a list of ``(lat, lon)`` tuples (≥ 3).

    Example – quadrilateral around Bağdat Caddesi::

        sample_street_polygon([
            (40.9730, 29.0450),
            (40.9730, 29.0900),
            (40.9600, 29.0900),
            (40.9600, 29.0450),
        ])
    """
    if len(vertices) < 3:
        raise ValueError("A polygon needs at least 3 vertices")
    G = _graph_from_polygon(vertices, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all)
