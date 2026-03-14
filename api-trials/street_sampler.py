"""
Street Sampler – Topology-Aware Sampling
─────────────────────────────────────────
Download an OSM street network and produce two distinct kinds of sample
points, each with a clear purpose for the downstream CV pipeline:

  **midblock**
      Evenly-spaced points along the interior of each edge, away from any
      intersection.  Bearing follows the canonical edge direction (west→east,
      south→north) so that +offset is always the *right* sidewalk.
      → 2 images: forward-right, forward-left (sidewalk analysis)

  **junction_approach**
      One point per edge per junction, placed ``junction_zone_m`` before the
      intersection node, with its bearing aimed *toward* the junction.
      → 1 image: straight ahead into the junction (crossing / signal analysis)

The two types are produced by separate logic — junction points are never
the accidental by-product of interpolation landing near a node.
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
    return [
        round(_circular_mean(bearings[max(0, i - half):min(n, i + half + 1)]), 2)
        for i in range(n)
    ]


def _normalize_street_name(name) -> str:
    if isinstance(name, list):
        return ", ".join(str(n) for n in name)
    return str(name) if name else "unknown"


# ── Graph topology ──────────────────────────────────────────────────────────

def _classify_nodes(G) -> dict[int, str]:
    """
    Classify every node by degree.

    Returns ``node_id → "junction" | "through" | "dead_end"``.
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
    Orient a LineString west→east (south→north as tie-break) so that
    bearing + 90° is always the *right-hand* side of the street.

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


# ── Midblock interpolation ─────────────────────────────────────────────────

def _interpolate_midblock(
    edge_geom: LineString,
    interval_m: float,
    crs_metric,
    street_name: str,
    edge_id: str,
    trim_start_m: float,
    trim_end_m: float,
) -> pd.DataFrame:
    """
    Interpolate evenly-spaced points along the *midblock* portion of an edge.

    The first ``trim_start_m`` and last ``trim_end_m`` are excluded — those
    regions belong to junction approach points (generated separately).
    Every returned point has ``zone_type = "midblock"``.
    """
    edge_length = edge_geom.length
    start = trim_start_m
    end = edge_length - trim_end_m

    if edge_length == 0 or start >= end:
        return pd.DataFrame()

    distances = np.arange(start, end, interval_m)
    if len(distances) == 0:
        return pd.DataFrame()
    if float(distances[-1]) < end - 0.5:
        distances = np.append(distances, end)

    pts = [edge_geom.interpolate(d) for d in distances]

    transformer = Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)
    lats, lons = [], []
    for p in pts:
        lon, lat = transformer.transform(p.x, p.y)
        lats.append(lat)
        lons.append(lon)

    raw_bearings: list[float] = []
    for i in range(len(lats)):
        if i < len(lats) - 1:
            raw_bearings.append(
                compute_bearing(lats[i], lons[i], lats[i + 1], lons[i + 1])
            )
        else:
            raw_bearings.append(raw_bearings[-1] if raw_bearings else 0.0)

    smoothed = _smooth_bearings(raw_bearings, window=3)

    return pd.DataFrame({
        "latitude":           [round(la, 7) for la in lats],
        "longitude":          [round(lo, 7) for lo in lons],
        "bearing":            smoothed,
        "distance_along_m":   [round(float(d), 2) for d in distances],
        "street_name":        street_name,
        "edge_id":            edge_id,
        "zone_type":          "midblock",
        "nearest_junction_m": -1.0,
    })


# ── Core sampling logic ────────────────────────────────────────────────────

def _sample_edges(
    G,
    interval_m: float,
    edge_index: int,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
    label: str = "",
) -> pd.DataFrame:
    """
    Build sample points from graph *G*:
      1. Classify nodes (junction / through / dead-end)
      2. For each edge, produce **midblock** points on the interior
      3. For each junction endpoint, produce one **junction_approach** point
         with bearing aimed at the junction
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
    transformer = Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)

    edges_proj = (edges_proj
                  .reset_index()
                  .sort_values("length", ascending=False)
                  .reset_index(drop=True))

    def _process_edge(idx: int):
        """Return (midblock_df, junction_rows) for edge at *idx*."""
        row = edges_proj.iloc[idx]
        u, v, key = int(row["u"]), int(row["v"]), int(row["key"])
        geom = row.geometry
        name = _normalize_street_name(row.get("name", "unknown"))
        edge_id = f"{u}_{v}_{key}"

        canon_geom, was_reversed = _canonicalize_geometry(geom)
        if was_reversed:
            start_node, end_node = v, u
        else:
            start_node, end_node = u, v

        start_type = node_types.get(start_node, "through")
        end_type = node_types.get(end_node, "through")
        edge_length = canon_geom.length

        # ── Midblock ────────────────────────────────────────────────
        trim_start = junction_zone_m if start_type == "junction" else 0.0
        trim_end = junction_zone_m if end_type == "junction" else 0.0

        mid_df = _interpolate_midblock(
            canon_geom, interval_m, crs_metric, name, edge_id,
            trim_start, trim_end,
        )

        # ── Junction approach points ────────────────────────────────
        junc_rows: list[dict] = []

        for is_start, ntype in [(True, start_type), (False, end_type)]:
            if ntype != "junction":
                continue

            # Place the approach point junction_zone_m away from the
            # junction along this edge.  For very short edges, use 70 %
            # of the edge length so we don't overshoot or land on top of
            # the junction node.
            approach_dist = min(junction_zone_m, edge_length * 0.7)

            if is_start:
                sample_at = approach_dist
                junc_coords = canon_geom.coords[0]
            else:
                sample_at = edge_length - approach_dist
                junc_coords = canon_geom.coords[-1]

            approach_pt = canon_geom.interpolate(sample_at)

            a_lon, a_lat = transformer.transform(approach_pt.x, approach_pt.y)
            j_lon, j_lat = transformer.transform(junc_coords[0], junc_coords[1])

            toward_bearing = compute_bearing(a_lat, a_lon, j_lat, j_lon)

            junc_rows.append({
                "latitude":           round(a_lat, 7),
                "longitude":          round(a_lon, 7),
                "bearing":            round(toward_bearing, 2),
                "distance_along_m":   round(sample_at, 2),
                "street_name":        name,
                "edge_id":            edge_id,
                "zone_type":          "junction_approach",
                "nearest_junction_m": round(approach_dist, 2),
            })

        return mid_df, junc_rows

    # ── Iterate over edges ──────────────────────────────────────────────
    if sample_all:
        print(f"  Sampling ALL {len(edges_proj)} edges …")
        all_midblock: list[pd.DataFrame] = []
        all_junction: list[dict] = []

        for i in range(len(edges_proj)):
            mid_df, junc_rows = _process_edge(i)
            if len(mid_df) > 0:
                all_midblock.append(mid_df)
            all_junction.extend(junc_rows)

        frames: list[pd.DataFrame] = []
        if all_midblock:
            frames.append(pd.concat(all_midblock, ignore_index=True))
        if all_junction:
            frames.append(pd.DataFrame(all_junction))
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

    else:
        if edge_index >= len(edges_proj):
            raise IndexError(
                f"edge_index={edge_index} but only {len(edges_proj)} edges exist"
            )
        mid_df, junc_rows = _process_edge(edge_index)
        frames = []
        if len(mid_df) > 0:
            frames.append(mid_df)
        if junc_rows:
            frames.append(pd.DataFrame(junc_rows))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

    zone_counts = df["zone_type"].value_counts().to_dict()
    print(f"  Total: {len(df)} sample points  |  Zones: {zone_counts}")
    return df


# ── Public API ──────────────────────────────────────────────────────────────

def sample_street(
    place: str,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
) -> pd.DataFrame:
    """Sample points along edges inside a *place name* area."""
    G = _graph_from_place(place, network_type)
    return _sample_edges(G, interval_m, edge_index,
                         sample_all=sample_all,
                         junction_zone_m=junction_zone_m,
                         label=place)


def sample_street_bbox(
    north: float, south: float, east: float, west: float,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
) -> pd.DataFrame:
    """Sample points along edges inside an explicit bounding box."""
    G = _graph_from_bbox(north, south, east, west, network_type)
    return _sample_edges(G, interval_m, edge_index,
                         sample_all=sample_all,
                         junction_zone_m=junction_zone_m)


def sample_street_point(
    lat: float, lon: float, dist_m: float = 500,
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
) -> pd.DataFrame:
    """Sample points along edges within *dist_m* metres of (lat, lon)."""
    G = _graph_from_point(lat, lon, dist_m, network_type)
    return _sample_edges(G, interval_m, edge_index,
                         sample_all=sample_all,
                         junction_zone_m=junction_zone_m)


def sample_street_polygon(
    vertices: list[tuple[float, float]],
    network_type: str = DEFAULT_NETWORK_TYPE,
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    edge_index: int = DEFAULT_EDGE_INDEX,
    sample_all: bool = False,
    junction_zone_m: float = JUNCTION_ZONE_M,
) -> pd.DataFrame:
    """
    Sample points along edges inside an arbitrary polygon.

    *vertices* is a list of ``(lat, lon)`` tuples (≥ 3).
    """
    if len(vertices) < 3:
        raise ValueError("A polygon needs at least 3 vertices")
    G = _graph_from_polygon(vertices, network_type)
    return _sample_edges(G, interval_m, edge_index,
                         sample_all=sample_all,
                         junction_zone_m=junction_zone_m)
