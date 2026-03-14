"""
Street Sampler – Direction-Aware with Zone Classification
─────────────────────────────────────────────────────────
Download an OSM street network, classify nodes by topology (junction /
through / dead-end), sample evenly-spaced points along edges, and label
each point with a *zone type* that tells the downstream CV pipeline which
analysis to apply:

  • **midblock** – away from intersections → sidewalk analysis
    (camera angles capture left / right sidewalks consistently)
  • **junction_approach** – near a high-degree node → crossing analysis
    (camera angles look for crosswalks, traffic lights, ramps)
  • **bend** – sharp direction change within a single edge

Edge geometries are canonicalised (west→east, south→north) so that
"right" always refers to the same physical side of the street regardless
of the original OSM digitisation direction.

Three ways to define the area
─────────────────────────────
1. Place name  → ``graph_from_place``
2. Bounding box → ``graph_from_bbox``
3. Center + radius → ``graph_from_point``
4. Custom polygon → ``graph_from_polygon``
"""

import math
from typing import Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

from config import (
    DEFAULT_NETWORK_TYPE,
    DEFAULT_SAMPLE_INTERVAL_M,
    DEFAULT_EDGE_INDEX,
    JUNCTION_ZONE_M,
    CORNER_THRESHOLD_DEG,
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


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference *b* − *a*, in (−180, 180]."""
    d = (b - a) % 360
    return d - 360 if d > 180 else d


def _circular_mean(angles_deg: list[float]) -> float:
    """Circular (vector) mean of angles in degrees → [0, 360)."""
    rads = [math.radians(a) for a in angles_deg]
    s = sum(math.sin(r) for r in rads)
    c = sum(math.cos(r) for r in rads)
    return math.degrees(math.atan2(s, c)) % 360


def _smooth_bearings(bearings: list[float], window: int = 3) -> list[float]:
    """Smooth a sequence of compass bearings with a circular moving average."""
    n = len(bearings)
    if n <= 1:
        return [round(b, 2) for b in bearings]
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(round(_circular_mean(bearings[lo:hi]), 2))
    return smoothed


# ── Graph topology helpers ──────────────────────────────────────────────────

def _classify_nodes(G) -> dict[int, str]:
    """
    Classify every node in *G* by its topological role.

    Returns a dict  ``node_id → "junction" | "through" | "dead_end"``.
    A junction is any node where 3+ edges meet – i.e. an intersection.
    """
    types: dict[int, str] = {}
    for node, degree in G.degree():
        if degree >= 3:
            types[node] = "junction"
        elif degree == 1:
            types[node] = "dead_end"
        else:
            types[node] = "through"
    return types


def _canonicalize_geometry(geom: LineString) -> tuple[LineString, bool]:
    """
    Ensure a LineString runs in a canonical geographic direction:
    west → east (increasing easting / x), with south → north as tie-break.

    Returns ``(geometry, was_reversed)``.
    """
    coords = list(geom.coords)
    sx, sy = coords[0][:2]
    ex, ey = coords[-1][:2]
    if sx > ex or (sx == ex and sy > ey):
        return LineString(coords[::-1]), True
    return geom, False


# ── Graph download helpers ──────────────────────────────────────────────────

def _graph_from_place(place: str, network_type: str):
    print(f"Downloading '{network_type}' network for place '{place}' …")
    return ox.graph_from_place(place, network_type=network_type)


def _graph_from_bbox(
    north: float, south: float, east: float, west: float,
    network_type: str,
):
    print(f"Downloading '{network_type}' network for bbox "
          f"N={north}, S={south}, E={east}, W={west} …")
    return ox.graph_from_bbox(bbox=(north, south, east, west),
                              network_type=network_type)


def _graph_from_point(
    lat: float, lon: float, dist_m: float, network_type: str,
):
    print(f"Downloading '{network_type}' network within {dist_m} m of "
          f"({lat}, {lon}) …")
    return ox.graph_from_point((lat, lon), dist=dist_m,
                               network_type=network_type)


def _graph_from_polygon(
    vertices: list[tuple[float, float]], network_type: str,
):
    ring = [(lon, lat) for lat, lon in vertices]
    poly = Polygon(ring)
    print(f"Downloading '{network_type}' network for custom polygon "
          f"({len(vertices)} vertices) …")
    return ox.graph_from_polygon(poly, network_type=network_type)


# ── Edge → sample points (direction-aware) ─────────────────────────────────

def _interpolate_edge(
    edge_geom: LineString,
    interval_m: float,
    crs_metric,
    street_name,
    edge_id: str,
    start_node_type: str,
    end_node_type: str,
    junction_zone_m: float,
    corner_threshold_deg: float,
) -> pd.DataFrame:
    """
    Interpolate evenly-spaced points along *edge_geom* (already projected
    to a metric CRS and canonicalised west→east) and classify each point
    into a zone type.
    """
    if isinstance(street_name, list):
        street_name = ", ".join(str(n) for n in street_name)
    street_name = str(street_name) if street_name else "unknown"

    edge_length = edge_geom.length
    if edge_length == 0:
        return pd.DataFrame()

    distances = np.arange(0, edge_length, interval_m)
    if len(distances) == 0:
        return pd.DataFrame()
    if distances[-1] != edge_length:
        distances = np.append(distances, edge_length)

    pts = [edge_geom.interpolate(d) for d in distances]

    transformer = Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)
    lats, lons = [], []
    for p in pts:
        lon, lat = transformer.transform(p.x, p.y)
        lats.append(lat)
        lons.append(lon)

    # Raw bearings follow the canonical edge direction
    raw_bearings: list[float] = []
    for i in range(len(lats)):
        if i < len(lats) - 1:
            raw_bearings.append(
                compute_bearing(lats[i], lons[i], lats[i + 1], lons[i + 1])
            )
        else:
            raw_bearings.append(raw_bearings[-1] if raw_bearings else 0.0)

    smoothed = _smooth_bearings(raw_bearings, window=3)

    # ── Zone classification ─────────────────────────────────────────────
    zone_types: list[str] = []
    nearest_junc: list[float] = []
    bearing_deltas: list[float] = []

    for i, d in enumerate(distances):
        dist_to_start = d
        dist_to_end = edge_length - d

        # Distance to nearest junction node
        junc_candidates: list[float] = []
        if start_node_type == "junction":
            junc_candidates.append(dist_to_start)
        if end_node_type == "junction":
            junc_candidates.append(dist_to_end)
        nj = min(junc_candidates) if junc_candidates else -1.0
        nearest_junc.append(round(nj, 2))

        # Bearing change at this point
        if i > 0:
            delta = abs(_angle_diff(raw_bearings[i - 1], raw_bearings[i]))
        else:
            delta = 0.0
        bearing_deltas.append(round(delta, 2))

        # Classification
        near_junction = nj >= 0 and nj <= junction_zone_m
        is_bend = delta > corner_threshold_deg

        if near_junction:
            zone_types.append("junction_approach")
        elif is_bend:
            zone_types.append("bend")
        else:
            zone_types.append("midblock")

    return pd.DataFrame({
        "latitude":           [round(la, 7) for la in lats],
        "longitude":          [round(lo, 7) for lo in lons],
        "bearing":            smoothed,
        "distance_along_m":   [round(float(d), 2) for d in distances],
        "street_name":        street_name,
        "edge_id":            edge_id,
        "zone_type":          zone_types,
        "nearest_junction_m": nearest_junc,
        "bearing_change_deg": bearing_deltas,
    })


def _sample_edges(
    G,
    interval_m: float,
    edge_index: int,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
    corner_threshold_deg: float = CORNER_THRESHOLD_DEG,
    label: str = "",
) -> pd.DataFrame:
    """
    Project graph *G*, classify nodes, canonicalise edge directions, sample
    points along edge(s), and return a DataFrame with zone labels.
    """
    G_undir = ox.convert.to_undirected(G)
    node_types = _classify_nodes(G_undir)

    n_junctions = sum(1 for v in node_types.values() if v == "junction")
    n_dead = sum(1 for v in node_types.values() if v == "dead_end")
    print(f"  Graph: {G_undir.number_of_nodes()} nodes "
          f"({n_junctions} junctions, {n_dead} dead-ends), "
          f"{G_undir.number_of_edges()} edges")

    G_proj = ox.project_graph(G_undir)
    edges_proj = ox.graph_to_gdfs(G_proj, nodes=False)
    crs_metric = edges_proj.crs

    # Keep u, v, key from the MultiIndex as columns
    edges_proj = (edges_proj
                  .reset_index()
                  .sort_values("length", ascending=False)
                  .reset_index(drop=True))

    if sample_all:
        print(f"  Sampling ALL {len(edges_proj)} edges …")
        frames = []
        for i in range(len(edges_proj)):
            row = edges_proj.iloc[i]
            u, v, key = int(row["u"]), int(row["v"]), int(row["key"])
            geom = row.geometry
            name = row.get("name", "unknown")

            canon_geom, was_reversed = _canonicalize_geometry(geom)
            if was_reversed:
                stype = node_types.get(v, "through")
                etype = node_types.get(u, "through")
            else:
                stype = node_types.get(u, "through")
                etype = node_types.get(v, "through")

            edge_id = f"{u}_{v}_{key}"
            frames.append(_interpolate_edge(
                canon_geom, interval_m, crs_metric, name, edge_id,
                stype, etype, junction_zone_m, corner_threshold_deg,
            ))
        df = pd.concat(frames, ignore_index=True)
        zone_counts = df["zone_type"].value_counts().to_dict()
        print(f"  Total: {len(df)} sample points across {len(edges_proj)} edges")
        print(f"  Zones: {zone_counts}")
        return df

    # ── Single-edge mode ────────────────────────────────────────────────
    if edge_index >= len(edges_proj):
        raise IndexError(
            f"edge_index={edge_index} but only {len(edges_proj)} edges exist"
        )
    row = edges_proj.iloc[edge_index]
    u, v, key = int(row["u"]), int(row["v"]), int(row["key"])
    geom = row.geometry
    name = row.get("name", "unknown")

    canon_geom, was_reversed = _canonicalize_geometry(geom)
    if was_reversed:
        stype = node_types.get(v, "through")
        etype = node_types.get(u, "through")
    else:
        stype = node_types.get(u, "through")
        etype = node_types.get(v, "through")

    edge_id = f"{u}_{v}_{key}"
    print(f"  Street: {name}  |  length: {geom.length:.1f} m  |  "
          f"start: {stype}, end: {etype}")
    return _interpolate_edge(
        canon_geom, interval_m, crs_metric, name, edge_id,
        stype, etype, junction_zone_m, corner_threshold_deg,
    )


# ── Public API ──────────────────────────────────────────────────────────────

def sample_street(
    place: str,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
    corner_threshold_deg: float = CORNER_THRESHOLD_DEG,
) -> pd.DataFrame:
    """Sample points along edges inside a *place name* area."""
    G = _graph_from_place(place, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all,
                         junction_zone_m=junction_zone_m,
                         corner_threshold_deg=corner_threshold_deg,
                         label=place)


def sample_street_bbox(
    north: float, south: float, east: float, west: float,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
    corner_threshold_deg: float = CORNER_THRESHOLD_DEG,
) -> pd.DataFrame:
    """Sample points along edges inside an explicit bounding box."""
    G = _graph_from_bbox(north, south, east, west, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all,
                         junction_zone_m=junction_zone_m,
                         corner_threshold_deg=corner_threshold_deg)


def sample_street_point(
    lat: float, lon: float, dist_m: float = 500,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
    corner_threshold_deg: float = CORNER_THRESHOLD_DEG,
) -> pd.DataFrame:
    """Sample points along edges within *dist_m* metres of (lat, lon)."""
    G = _graph_from_point(lat, lon, dist_m, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all,
                         junction_zone_m=junction_zone_m,
                         corner_threshold_deg=corner_threshold_deg)


def sample_street_polygon(
    vertices: list[tuple[float, float]],
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
    corner_threshold_deg: float = CORNER_THRESHOLD_DEG,
) -> pd.DataFrame:
    """
    Sample points along edges inside an arbitrary polygon.

    *vertices* is a list of ``(lat, lon)`` tuples (≥ 3).
    """
    if len(vertices) < 3:
        raise ValueError("A polygon needs at least 3 vertices")
    G = _graph_from_polygon(vertices, network_type)
    return _sample_edges(G, interval_m, edge_index, sample_all=sample_all,
                         junction_zone_m=junction_zone_m,
                         corner_threshold_deg=corner_threshold_deg)
