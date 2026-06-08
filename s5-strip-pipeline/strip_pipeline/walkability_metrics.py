"""Walkability and accessibility metrics for stitched sidewalk strips.

Scans horizontal slices of the FOOTPRINT_BOXES strip to compute effective
clear widths, detect significant narrowing events, and quantify obstacle
frequency.  All spatial measurements use a uniform px→m factor derived from
the camera-geometry average sidewalk width and the target pixel width.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Thresholds (sensible defaults, overridable via function args)
# ---------------------------------------------------------------------------
WHEELCHAIR_WIDTH_M = 0.65          # minimum for a standard wheelchair
ADA_ACCESSIBLE_WIDTH_M = 0.90     # ADA / European standard
SIGNIFICANT_DROP_M = 0.60         # ≥60 cm reduction from base counts as an event
DISTURBING_OBSTACLE_THRESHOLD_M = 0.20  # obstacle reducing width by ≥20 cm
ENCOUNTER_MERGE_GAP_PX = 30       # rows between obstacle boxes to merge into one encounter


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class WalkabilityMetrics:
    """All walkability / accessibility metrics for one strip side."""

    base_sidewalk_width_m: float = 0.0
    sidewalk_width_px: int = 0
    px_to_m: float = 0.0
    min_clear_width_m: float = 0.0
    min_clear_width_location_y: int = 0
    wheelchair_passable_65cm: bool = True
    ada_accessible_90cm: bool = True
    width_drop_60cm_count: int = 0
    width_drop_60cm_events: list[dict[str, Any]] = field(default_factory=list)
    obstacle_encounter_count: int = 0
    obstacle_frequency_per_meter: float = 0.0
    mean_clear_width_m: float = 0.0
    strip_length_m: float = 0.0
    clear_width_profile_m: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dictionary (excludes the large per-row array)."""
        d: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if k == "clear_width_profile_m":
                # Include summary stats only
                if len(self.clear_width_profile_m) > 0:
                    arr = self.clear_width_profile_m
                    d["clear_width_profile_summary"] = {
                        "rows": int(len(arr)),
                        "min_m": float(np.nanmin(arr)),
                        "max_m": float(np.nanmax(arr)),
                        "mean_m": float(np.nanmean(arr)),
                        "median_m": float(np.nanmedian(arr)),
                        "std_m": float(np.nanstd(arr)),
                    }
                continue
            d[k] = v
        return d


# ---------------------------------------------------------------------------
# Per-row helpers
# ---------------------------------------------------------------------------

def find_largest_free_gap(sidewalk_left: int, sidewalk_right: int,
                          obstacle_cols: np.ndarray) -> int:
    """Return the width (px) of the largest contiguous obstacle-free span
    within [sidewalk_left, sidewalk_right].

    Parameters
    ----------
    sidewalk_left, sidewalk_right : int
        Inclusive column range of the sidewalk in this row.
    obstacle_cols : np.ndarray
        Sorted array of column indices occupied by obstacles in this row.
        May be empty.

    Returns
    -------
    int
        Width in pixels of the largest free gap.
    """
    total_width = sidewalk_right - sidewalk_left + 1
    if total_width <= 0:
        return 0

    if len(obstacle_cols) == 0:
        return total_width

    # Clip obstacle columns to the sidewalk extent
    obs = obstacle_cols[(obstacle_cols >= sidewalk_left) & (obstacle_cols <= sidewalk_right)]
    if len(obs) == 0:
        return total_width

    # Build list of free-gap boundaries
    # Gaps exist: [sidewalk_left..first_obs-1], between consecutive obs runs,
    # [last_obs+1..sidewalk_right]
    obs_sorted = np.unique(obs)

    # Find contiguous runs of obstacle columns
    diffs = np.diff(obs_sorted)
    break_indices = np.where(diffs > 1)[0]

    # Build obstacle run boundaries [(start, end), ...]
    runs: list[tuple[int, int]] = []
    run_start = int(obs_sorted[0])
    for bi in break_indices:
        run_end = int(obs_sorted[bi])
        runs.append((run_start, run_end))
        run_start = int(obs_sorted[bi + 1])
    runs.append((run_start, int(obs_sorted[-1])))

    # Compute free gaps
    max_gap = 0

    # Gap before first obstacle run
    gap_before = runs[0][0] - sidewalk_left
    max_gap = max(max_gap, gap_before)

    # Gaps between consecutive obstacle runs
    for i in range(len(runs) - 1):
        gap = runs[i + 1][0] - runs[i][1] - 1
        max_gap = max(max_gap, gap)

    # Gap after last obstacle run
    gap_after = sidewalk_right - runs[-1][1]
    max_gap = max(max_gap, gap_after)

    return max(0, max_gap)


def compute_row_clear_widths(sidewalk_mask: np.ndarray,
                             boxes: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Compute the effective clear width (px) for every row of the strip.

    Uses the **global** sidewalk column extent (min/max columns across all
    rows) as fixed left/right boundaries.  This avoids artifacts from rows
    where the sidewalk mask tapers to just a few pixels at tile seams.

    Parameters
    ----------
    sidewalk_mask : np.ndarray
        Boolean 2-D mask, True where the sidewalk is.
    boxes : list[dict]
        Deduplicated footprint boxes from ``collect_footprint_boxes``.
        Each has ``bbox = (min_row, min_col, max_row, max_col)``.

    Returns
    -------
    clear_widths : np.ndarray
        1-D float array of shape ``(H,)`` with per-row clear width in pixels.
        Rows outside the sidewalk band are set to NaN.
    sidewalk_width_px : int
        The global sidewalk width in pixels (``sw_right - sw_left + 1``).
    """
    H, W = sidewalk_mask.shape[:2]
    clear_widths = np.full(H, np.nan, dtype=np.float64)

    # Global sidewalk extent — same approach as render_footprint_box_strip()
    sidewalk_rows = np.where(sidewalk_mask.any(axis=1))[0]
    sidewalk_cols = np.where(sidewalk_mask.any(axis=0))[0]
    if len(sidewalk_rows) == 0 or len(sidewalk_cols) == 0:
        return clear_widths, 0

    sw_left = int(sidewalk_cols[0])
    sw_right = int(sidewalk_cols[-1])
    sw_y_start = int(sidewalk_rows[0])
    sw_y_end = int(sidewalk_rows[-1])
    sidewalk_width_px = sw_right - sw_left + 1

    # Build a combined obstacle occupancy mask from boxes
    obstacle_mask = np.zeros((H, W), dtype=bool)
    for box in boxes:
        r0, c0, r1, c1 = box["bbox"]
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(H, r1)
        c1 = min(W, c1)
        obstacle_mask[r0:r1, c0:c1] = True

    # Scan only rows within the sidewalk band, using fixed column bounds
    for y in range(sw_y_start, sw_y_end + 1):
        obs_cols = np.where(obstacle_mask[y])[0]
        clear_widths[y] = float(find_largest_free_gap(sw_left, sw_right, obs_cols))

    return clear_widths, sidewalk_width_px


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def count_width_drop_events(clear_width_m: np.ndarray,
                            base_width_m: float,
                            drop_threshold_m: float = SIGNIFICANT_DROP_M,
                            min_event_rows: int = 3) -> list[dict[str, Any]]:
    """Find contiguous regions where clear width drops ≥ *drop_threshold_m*
    below *base_width_m*.

    Parameters
    ----------
    clear_width_m : np.ndarray
        Per-row clear width in metres (NaN for rows without sidewalk).
    base_width_m : float
        Reference sidewalk width from camera geometry.
    drop_threshold_m : float
        Minimum drop from base to count as an event.
    min_event_rows : int
        Ignore drops shorter than this many rows (noise filter).

    Returns
    -------
    list[dict]
        Each dict has ``y_start``, ``y_end``, ``narrowest_m``,
        ``mean_width_m``, ``drop_from_base_m``.
    """
    threshold = base_width_m - drop_threshold_m
    is_dropped = np.zeros(len(clear_width_m), dtype=bool)
    for i, w in enumerate(clear_width_m):
        if not np.isnan(w) and w < threshold:
            is_dropped[i] = True

    events: list[dict[str, Any]] = []
    in_event = False
    start = 0

    for i in range(len(is_dropped) + 1):
        if i < len(is_dropped) and is_dropped[i]:
            if not in_event:
                in_event = True
                start = i
        else:
            if in_event:
                in_event = False
                length = i - start
                if length >= min_event_rows:
                    segment = clear_width_m[start:i]
                    valid = segment[~np.isnan(segment)]
                    if len(valid) > 0:
                        narrowest = float(np.min(valid))
                        events.append({
                            "y_start": int(start),
                            "y_end": int(i),
                            "rows": int(length),
                            "narrowest_m": round(narrowest, 4),
                            "mean_width_m": round(float(np.mean(valid)), 4),
                            "drop_from_base_m": round(base_width_m - narrowest, 4),
                        })

    return events


def count_obstacle_encounters(boxes: list[dict[str, Any]],
                              merge_gap_px: int = ENCOUNTER_MERGE_GAP_PX) -> list[dict[str, Any]]:
    """Cluster obstacle boxes by their y-span into distinct *encounters*.

    Two boxes whose y-ranges are within *merge_gap_px* of each other are
    merged into a single encounter.  Returns a list of encounter dicts, each
    with the merged y-span and the class names involved.
    """
    if not boxes:
        return []

    # Extract (y_min, y_max, class_name) per box
    intervals = []
    for box in boxes:
        r0, _, r1, _ = box["bbox"]
        intervals.append((int(r0), int(r1), box.get("class_name", "unknown")))

    # Sort by y_min
    intervals.sort(key=lambda t: t[0])

    # Merge overlapping / close intervals
    encounters: list[dict[str, Any]] = []
    cur_start, cur_end = intervals[0][0], intervals[0][1]
    cur_classes: list[str] = [intervals[0][2]]

    for y0, y1, cls in intervals[1:]:
        if y0 <= cur_end + merge_gap_px:
            # Extend current encounter
            cur_end = max(cur_end, y1)
            cur_classes.append(cls)
        else:
            encounters.append({
                "y_start": cur_start,
                "y_end": cur_end,
                "classes": sorted(set(cur_classes)),
                "box_count": len(cur_classes),
            })
            cur_start, cur_end = y0, y1
            cur_classes = [cls]

    encounters.append({
        "y_start": cur_start,
        "y_end": cur_end,
        "classes": sorted(set(cur_classes)),
        "box_count": len(cur_classes),
    })

    return encounters


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_walkability_metrics(sidewalk_mask: np.ndarray,
                                boxes: list[dict[str, Any]],
                                px_to_m: float,
                                base_width_m: float,
                                drop_threshold_m: float = SIGNIFICANT_DROP_M,
                                disturbing_threshold_m: float = DISTURBING_OBSTACLE_THRESHOLD_M) -> WalkabilityMetrics:
    """Compute all walkability metrics for one strip side.

    Parameters
    ----------
    sidewalk_mask : np.ndarray
        Boolean (H, W) mask of sidewalk pixels in the stitched strip.
    boxes : list[dict]
        Deduplicated FOOTPRINT_BOXES with ``bbox`` keys.
    px_to_m : float
        Pixel-to-metre conversion factor (``avg_sidewalk_width_m / target_sidewalk_width_px``).
        Used as fallback; the function prefers deriving it from the actual
        global sidewalk pixel width when available.
    base_width_m : float
        Average sidewalk width from camera geometry.
    drop_threshold_m : float
        Width drop ≥ this from base triggers an event.
    disturbing_threshold_m : float
        Obstacle reducing width by ≥ this counts as disturbing.
    """
    H = sidewalk_mask.shape[0]
    metrics = WalkabilityMetrics(base_sidewalk_width_m=base_width_m)

    # --- Per-row clear widths (using global sidewalk bounds) ---
    clear_widths_px, sidewalk_width_px = compute_row_clear_widths(sidewalk_mask, boxes)

    # Derive px_to_m from the actual sidewalk pixel extent:
    # sidewalk_width_px pixels == base_width_m metres
    if sidewalk_width_px > 0:
        px_to_m = base_width_m / sidewalk_width_px
    metrics.sidewalk_width_px = sidewalk_width_px
    metrics.px_to_m = round(px_to_m, 6)

    clear_widths_m = clear_widths_px * px_to_m
    metrics.clear_width_profile_m = clear_widths_m

    # --- Global stats ---
    valid_mask = ~np.isnan(clear_widths_m)
    valid_widths = clear_widths_m[valid_mask]

    if len(valid_widths) == 0:
        return metrics

    metrics.min_clear_width_m = round(float(np.min(valid_widths)), 4)
    metrics.min_clear_width_location_y = int(np.nanargmin(clear_widths_m))
    metrics.mean_clear_width_m = round(float(np.mean(valid_widths)), 4)
    metrics.wheelchair_passable_65cm = metrics.min_clear_width_m >= WHEELCHAIR_WIDTH_M
    metrics.ada_accessible_90cm = metrics.min_clear_width_m >= ADA_ACCESSIBLE_WIDTH_M

    # --- Strip length in metres (count of rows with sidewalk × px_to_m) ---
    metrics.strip_length_m = round(float(np.sum(valid_mask)) * px_to_m, 4)

    # --- Width-drop events ---
    events = count_width_drop_events(clear_widths_m, base_width_m, drop_threshold_m)
    metrics.width_drop_60cm_count = len(events)
    metrics.width_drop_60cm_events = events

    # --- Obstacle encounters ---
    encounters = count_obstacle_encounters(boxes)
    metrics.obstacle_encounter_count = len(encounters)
    if metrics.strip_length_m > 0:
        metrics.obstacle_frequency_per_meter = round(
            len(encounters) / metrics.strip_length_m, 4
        )

    return metrics


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def generate_metrics_debug_strip(footprint_strip: np.ndarray,
                                 metrics: WalkabilityMetrics,
                                 px_to_m: float) -> np.ndarray:
    """Render a debug image: footprint strip + clear-width heatmap + annotations.

    Left: the existing footprint-box rendered strip.
    Right: a narrow (80 px) colour-coded column showing per-row clear width.
    """
    H, W = footprint_strip.shape[:2]
    heatmap_w = 80
    annotation_w = 200
    total_w = W + heatmap_w + annotation_w
    out = np.zeros((H, total_w, 3), dtype=np.uint8)
    out[:, :, :] = 30  # dark background

    # Copy footprint strip on the left
    out[:, :W, :] = footprint_strip[:, :, :3] if footprint_strip.ndim == 3 else np.repeat(footprint_strip[:, :, None], 3, axis=2)

    # Build heatmap column
    profile = metrics.clear_width_profile_m
    base = metrics.base_sidewalk_width_m

    for y in range(H):
        if y >= len(profile) or np.isnan(profile[y]):
            # No sidewalk — dark grey
            out[y, W:W + heatmap_w, :] = (50, 50, 50)
            continue

        w_m = profile[y]
        if w_m >= base * 0.95:
            # Full width — green
            color = (60, 200, 80)
        elif w_m >= ADA_ACCESSIBLE_WIDTH_M:
            # Narrowed but ADA-compliant — yellow
            color = (220, 200, 40)
        elif w_m >= WHEELCHAIR_WIDTH_M:
            # Below ADA, above wheelchair — orange
            color = (240, 140, 30)
        else:
            # Below wheelchair threshold — red
            color = (220, 50, 50)

        out[y, W:W + heatmap_w, :] = color

    # Draw event markers and annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    annotation_x = W + heatmap_w + 8

    for event in metrics.width_drop_60cm_events:
        y_start = event["y_start"]
        y_end = event["y_end"]
        narrowest = event["narrowest_m"]

        # Horizontal marker lines
        cv2.line(out, (W, y_start), (W + heatmap_w + annotation_w - 1, y_start), (255, 100, 100), 1)
        cv2.line(out, (W, y_end), (W + heatmap_w + annotation_w - 1, y_end), (255, 100, 100), 1)

        # Label
        mid_y = max(14, min(H - 8, (y_start + y_end) // 2))
        text = f"{narrowest:.2f}m"
        cv2.putText(out, text, (annotation_x, mid_y), font, 0.45, (255, 180, 180), 1, cv2.LINE_AA)

    # Top-right summary box
    summary_lines = [
        f"Base width: {metrics.base_sidewalk_width_m:.2f} m",
        f"Min clear:  {metrics.min_clear_width_m:.2f} m",
        f"Mean clear: {metrics.mean_clear_width_m:.2f} m",
        f"WC 65cm:    {'PASS' if metrics.wheelchair_passable_65cm else 'FAIL'}",
        f"ADA 90cm:   {'PASS' if metrics.ada_accessible_90cm else 'FAIL'}",
        f"Drop events: {metrics.width_drop_60cm_count}",
        f"Obstacles:   {metrics.obstacle_encounter_count}",
        f"Freq: {metrics.obstacle_frequency_per_meter:.2f}/m",
        f"Length: {metrics.strip_length_m:.1f} m",
    ]

    # Background box
    box_h = 20 * len(summary_lines) + 16
    box_y0 = 6
    overlay = out.copy()
    cv2.rectangle(overlay, (annotation_x - 4, box_y0),
                  (total_w - 4, box_y0 + box_h), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)

    text_y = box_y0 + 18
    for line in summary_lines:
        cv2.putText(out, line, (annotation_x, text_y), font, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        text_y += 20

    # Heatmap legend at bottom-right
    legend_y = H - 90
    legend_items = [
        ((60, 200, 80), "Full width"),
        ((220, 200, 40), ">= 90cm (ADA)"),
        ((240, 140, 30), ">= 65cm (WC)"),
        ((220, 50, 50), "< 65cm (blocked)"),
    ]
    for color, label in legend_items:
        if legend_y + 16 > H:
            break
        cv2.rectangle(out, (annotation_x, legend_y), (annotation_x + 12, legend_y + 12), color, -1)
        cv2.putText(out, label, (annotation_x + 18, legend_y + 11), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        legend_y += 18

    return out
