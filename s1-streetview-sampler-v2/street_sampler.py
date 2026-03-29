"""
Street Sampler v2
─────────────────
Download an OSM street network inside an arbitrary polygon, decompose it
into straight segments (maximal chains of degree-2 nodes), and return
evenly-spaced sample points with consistent bearings and junction labels.

Key improvements over v1:
  - Polygon-only mode (no place / bbox / point).
  - Edges are deduplicated so no street is sampled twice.
  - Each segment gets a *canonical* traversal direction (deterministic,
    independent of OSM way ordering) so "forward" and "backward" are
    always consistent.
  - Points touching degree-≥3 nodes are labelled ``point_type="junction"``
    for downstream special handling.
"""

import math
from collections import defaultdict

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge

from config import DEFAULT_NETWORK_TYPE, DEFAULT_SAMPLE_INTERVAL_M


# ── Geometry helpers ────────────────────────────────────────────────────────

def compute_bearing(lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
    """Initial compass bearing (0-360) from point 1 to point 2."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)
    x = math.sin(d_lon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon))
    return math.degrees(math.atan2(x, y)) % 360


# ── Graph acquisition ──────────────────────────────────────────────────────

def _graph_from_polygon(
    vertices: list[tuple[float, float]],
    network_type: str,
):
    """
    Download an OSM graph inside an arbitrary polygon.

    *vertices* is a list of (lat, lon) tuples (at least 3).
    The polygon is closed automatically by Shapely.
    """
    ring = [(lon, lat) for lat, lon in vertices]
    poly = Polygon(ring)
    print(f"Downloading '{network_type}' network for custom polygon "
          f"({len(vertices)} vertices) …")
    return ox.graph_from_polygon(poly, network_type=network_type)


# ── Segment decomposition ──────────────────────────────────────────────────

def _deduplicate_edges(G_undir):
    """
    Remove parallel edges between the same (u, v) pair, keeping the one
    with the longest geometry.  Operates in-place on the MultiGraph.
    """
    seen: dict[tuple, int] = {}
    to_remove: list[tuple[int, int, int]] = []

    for u, v, key, data in G_undir.edges(keys=True, data=True):
        pair = (min(u, v), max(u, v))
        length = data.get("length", 0)
        if pair in seen:
            prev_key, prev_len = seen[pair]
            if length > prev_len:
                to_remove.append((pair[0], pair[1], prev_key))
                seen[pair] = (key, length)
            else:
                to_remove.append((u, v, key))
        else:
            seen[pair] = (key, length)

    for u, v, k in to_remove:
        if G_undir.has_edge(u, v, k):
            G_undir.remove_edge(u, v, k)


def _build_segment_chains(G_undir):
    """
    Walk the undirected graph and extract maximal chains of degree-2 nodes.

    Returns a list of node-lists.  Each chain runs from one "terminal" node
    (degree != 2, i.e. a junction or dead-end) to another.  Every edge in
    the graph appears in exactly one chain.
    """
    visited_edges: set[tuple[int, int]] = set()
    segments: list[list[int]] = []

    terminal_nodes = {n for n in G_undir.nodes() if G_undir.degree(n) != 2}

    for start in terminal_nodes:
        for neighbor in G_undir.neighbors(start):
            edge_key = (min(start, neighbor), max(start, neighbor))
            if edge_key in visited_edges:
                continue

            chain = [start, neighbor]
            visited_edges.add(edge_key)

            current = neighbor
            prev = start
            while G_undir.degree(current) == 2:
                nexts = [n for n in G_undir.neighbors(current) if n != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                ek = (min(current, nxt), max(current, nxt))
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                chain.append(nxt)
                prev = current
                current = nxt

            segments.append(chain)

    # Handle pure cycles (rings of degree-2 nodes with no terminal)
    all_edges = {(min(u, v), max(u, v)) for u, v in G_undir.edges()}
    remaining = all_edges - visited_edges
    if remaining:
        remaining_graph = nx.Graph()
        for u, v in remaining:
            remaining_graph.add_edge(u, v)
        for component in nx.connected_components(remaining_graph):
            cycle_nodes = list(component)
            if len(cycle_nodes) < 2:
                continue
            try:
                cycle = list(nx.find_cycle(remaining_graph.subgraph(cycle_nodes)))
                chain = [cycle[0][0]] + [e[1] for e in cycle]
                segments.append(chain)
            except nx.NetworkXNoCycle:
                pass

    return segments


def _chain_to_geometry(chain: list[int], G_proj) -> LineString:
    """
    Merge the individual edge geometries along *chain* into one LineString.
    """
    edge_geoms = []
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        data = G_proj.get_edge_data(u, v)
        if data is None:
            continue
        if isinstance(data, dict) and 0 in data:
            data = data[0]
        geom = data.get("geometry", None)
        if geom is None:
            u_data = G_proj.nodes[u]
            v_data = G_proj.nodes[v]
            geom = LineString([(u_data["x"], u_data["y"]),
                               (v_data["x"], v_data["y"])])
        edge_geoms.append(geom)

    if not edge_geoms:
        return LineString()

    merged = linemerge(edge_geoms)
    if merged.geom_type == "MultiLineString":
        coords = []
        for part in merged.geoms:
            coords.extend(list(part.coords))
        merged = LineString(coords)

    return merged


def _canonical_direction(geom: LineString, crs_metric) -> LineString:
    """
    Return the geometry oriented in a canonical direction: the endpoint
    with the smaller (lat, lon) tuple in WGS84 is always the start.
    This guarantees deterministic, reproducible forward/backward bearings.
    """
    transformer = Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)
    start_coords = geom.coords[0]
    end_coords = geom.coords[-1]

    start_lon, start_lat = transformer.transform(start_coords[0], start_coords[1])
    end_lon, end_lat = transformer.transform(end_coords[0], end_coords[1])

    if (start_lat, start_lon) > (end_lat, end_lon):
        return LineString(list(reversed(geom.coords)))
    return geom


def _get_street_name(chain: list[int], G) -> str:
    """Extract a street name from any edge along the chain."""
    for i in range(len(chain) - 1):
        data = G.get_edge_data(chain[i], chain[i + 1])
        if data is None:
            continue
        if isinstance(data, dict) and 0 in data:
            data = data[0]
        name = data.get("name", None)
        if name is not None:
            if isinstance(name, list):
                return ", ".join(str(n) for n in name)
            return str(name)
    return "unknown"


def decompose_into_segments(G, crs_metric):
    """
    Decompose a projected undirected graph into straight segments.

    Returns a list of dicts, each with:
      - segment_id   : int
      - geometry     : LineString (projected CRS, canonical direction)
      - street_name  : str
      - start_is_junction : bool
      - end_is_junction   : bool
      - chain        : list[int] (node IDs, for debugging)
    """
    G_undir = ox.convert.to_undirected(G)
    _deduplicate_edges(G_undir)

    G_proj = ox.project_graph(G_undir)
    crs = G_proj.graph.get("crs", crs_metric)

    junction_nodes = {n for n in G_proj.nodes() if G_proj.degree(n) >= 3}

    chains = _build_segment_chains(G_proj)

    segments = []
    for seg_idx, chain in enumerate(chains):
        geom = _chain_to_geometry(chain, G_proj)
        if geom.is_empty or geom.length < 1.0:
            continue

        geom = _canonical_direction(geom, crs)
        name = _get_street_name(chain, G_proj)

        segments.append({
            "segment_id": seg_idx,
            "geometry": geom,
            "street_name": name,
            "start_is_junction": chain[0] in junction_nodes,
            "end_is_junction": chain[-1] in junction_nodes,
            "chain": chain,
        })

    # Re-index segment_ids contiguously after filtering
    for i, seg in enumerate(segments):
        seg["segment_id"] = i

    print(f"  Decomposed into {len(segments)} segments "
          f"({len(junction_nodes)} junction nodes)")

    return segments, crs


# ── Interpolation ──────────────────────────────────────────────────────────

def _interpolate_segment(segment: dict, interval_m: float,
                         crs_metric) -> pd.DataFrame:
    """
    Interpolate evenly-spaced points along a single segment geometry.

    Returns a DataFrame with columns:
      segment_id, point_type, latitude, longitude, bearing,
      distance_along_m, street_name
    """
    geom = segment["geometry"]
    seg_id = segment["segment_id"]
    edge_length = geom.length

    if edge_length == 0:
        return pd.DataFrame()

    distances = np.arange(0, edge_length, interval_m).tolist()
    if distances[-1] != edge_length:
        distances.append(edge_length)
    distances = np.array(distances)

    pts = [geom.interpolate(d) for d in distances]

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

    n_points = len(lats)
    point_types = ["street"] * n_points

    if segment["start_is_junction"]:
        point_types[0] = "junction"
    if segment["end_is_junction"] and n_points > 1:
        point_types[-1] = "junction"

    return pd.DataFrame({
        "segment_id":       seg_id,
        "point_type":       point_types,
        "latitude":         [round(la, 7) for la in lats],
        "longitude":        [round(lo, 7) for lo in lons],
        "bearing":          bearings,
        "distance_along_m": [round(d, 2) for d in distances],
        "street_name":      segment["street_name"],
    })


# ── Public API ──────────────────────────────────────────────────────────────

def sample_polygon(
    vertices: list[tuple[float, float]],
    interval_m: float = DEFAULT_SAMPLE_INTERVAL_M,
    network_type: str = DEFAULT_NETWORK_TYPE,
) -> pd.DataFrame:
    """
    Sample points along all streets inside a polygon area.

    Parameters
    ----------
    vertices : list of (lat, lon) tuples
        At least 3 vertices defining the sampling area.
    interval_m : float
        Distance in metres between consecutive sample points.
    network_type : str
        OSM network type passed to OSMnx.

    Returns
    -------
    pd.DataFrame
        Columns: segment_id, point_id, point_type, latitude, longitude,
        bearing, distance_along_m, street_name
    """
    if len(vertices) < 3:
        raise ValueError("A polygon needs at least 3 vertices")

    G = _graph_from_polygon(vertices, network_type)

    G_proj = ox.project_graph(G)
    crs_metric = G_proj.graph.get("crs", None)
    if crs_metric is None:
        edges_proj = ox.graph_to_gdfs(G_proj, nodes=False)
        crs_metric = edges_proj.crs

    segments, crs = decompose_into_segments(G, crs_metric)

    frames = []
    for seg in segments:
        df_seg = _interpolate_segment(seg, interval_m, crs)
        if not df_seg.empty:
            frames.append(df_seg)

    if not frames:
        print("  WARNING: No sample points generated!")
        return pd.DataFrame(columns=[
            "segment_id", "point_id", "point_type",
            "latitude", "longitude", "bearing",
            "distance_along_m", "street_name",
        ])

    df = pd.concat(frames, ignore_index=True)
    df.insert(1, "point_id", [f"{i:04d}" for i in range(len(df))])

    print(f"  Total: {len(df)} sample points across {len(segments)} segments")
    junction_count = (df["point_type"] == "junction").sum()
    print(f"  Junction points: {junction_count}")

    return df
