"""Unit tests for walkability_metrics module.

Tests the core algorithmic functions with synthetic data — no GCS or heavy
pipeline dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from strip_pipeline.walkability_metrics import (
    find_largest_free_gap,
    compute_row_clear_widths,
    count_width_drop_events,
    count_obstacle_encounters,
    compute_walkability_metrics,
    WalkabilityMetrics,
)


# ---------------------------------------------------------------------------
# find_largest_free_gap
# ---------------------------------------------------------------------------

class TestFindLargestFreeGap:
    def test_no_obstacles(self):
        """Full sidewalk, no obstacles → gap = sidewalk width."""
        assert find_largest_free_gap(10, 109, np.array([])) == 100

    def test_single_obstacle_in_middle(self):
        """Obstacle cols 50-59 in sidewalk 0-99 → gaps of 50 and 40."""
        obs = np.arange(50, 60)
        assert find_largest_free_gap(0, 99, obs) == 50  # gap before obstacle

    def test_two_obstacles_gap_between(self):
        """Obstacles at edges, free gap in the middle."""
        obs = np.concatenate([np.arange(0, 20), np.arange(80, 100)])
        assert find_largest_free_gap(0, 99, obs) == 60  # gap between 20..79

    def test_obstacle_fills_entire_width(self):
        """Obstacle fills the whole sidewalk → gap = 0."""
        obs = np.arange(0, 100)
        assert find_largest_free_gap(0, 99, obs) == 0

    def test_obstacle_outside_sidewalk(self):
        """Obstacle columns outside sidewalk extent → ignored."""
        obs = np.array([200, 201, 202])
        assert find_largest_free_gap(0, 99, obs) == 100

    def test_single_pixel_obstacle(self):
        """One pixel obstacle at col 50 → gaps of 50 and 49."""
        obs = np.array([50])
        assert find_largest_free_gap(0, 99, obs) == 50  # 0..49 = 50 px

    def test_zero_width_sidewalk(self):
        """Sidewalk left > right → 0."""
        assert find_largest_free_gap(50, 49, np.array([])) == 0

    def test_adjacent_obstacles_multiple_gaps(self):
        """Three obstacle clusters with two gaps."""
        # Obstacle: 10-19, 40-49, 80-89  in sidewalk 0-99
        obs = np.concatenate([np.arange(10, 20), np.arange(40, 50), np.arange(80, 90)])
        # Gaps: 0-9 (10), 20-39 (20), 50-79 (30), 90-99 (10)
        assert find_largest_free_gap(0, 99, obs) == 30


# ---------------------------------------------------------------------------
# compute_row_clear_widths
# ---------------------------------------------------------------------------

class TestComputeRowClearWidths:
    def test_simple_strip_no_obstacles(self):
        """100-row strip, sidewalk cols 20-79 (60px), no boxes."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 20:80] = True
        widths, sw_px = compute_row_clear_widths(mask, [])
        assert widths.shape == (100,)
        assert sw_px == 60
        assert np.all(widths == 60.0)

    def test_strip_with_one_obstacle(self):
        """Obstacle box in rows 30-50, cols 40-60 (20px wide)."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 10:90] = True  # 80px wide sidewalk
        boxes = [{"bbox": (30, 40, 50, 60), "class_name": "bollard"}]
        widths, sw_px = compute_row_clear_widths(mask, boxes)
        assert sw_px == 80

        # Rows 0-29 and 50-99: full width = 80
        assert np.all(widths[:30] == 80.0)
        assert np.all(widths[50:] == 80.0)

        # Rows 30-49: obstacle cols 40-59 → gaps: 10-39 (30px) and 60-89 (30px)
        assert np.all(widths[30:50] == 30.0)

    def test_no_sidewalk_rows(self):
        """Rows outside the global sidewalk band should be NaN."""
        mask = np.zeros((50, 100), dtype=bool)
        mask[10:40, 20:80] = True
        widths, sw_px = compute_row_clear_widths(mask, [])
        assert sw_px == 60
        assert np.all(np.isnan(widths[:10]))
        assert np.all(np.isnan(widths[40:]))
        assert np.all(widths[10:40] == 60.0)

    def test_global_bounds_ignore_per_row_taper(self):
        """Even if some rows have a narrow mask, the global bounds are used."""
        mask = np.zeros((100, 200), dtype=bool)
        mask[:, 50:150] = True  # 100px wide globally
        # Taper a few rows to just 2 pixels
        mask[45, :] = False
        mask[45, 99:101] = True  # only 2px at row 45
        widths, sw_px = compute_row_clear_widths(mask, [])
        assert sw_px == 100  # global extent, not per-row
        assert widths[45] == 100.0  # still 100px because global bounds used


# ---------------------------------------------------------------------------
# count_width_drop_events
# ---------------------------------------------------------------------------

class TestCountWidthDropEvents:
    def test_no_drops(self):
        """Constant width at base → no events."""
        profile = np.full(100, 2.0)
        events = count_width_drop_events(profile, base_width_m=2.0)
        assert len(events) == 0

    def test_single_drop(self):
        """One 10-row region drops from 2.0 to 1.0 (drop = 1.0 > 0.6)."""
        profile = np.full(100, 2.0)
        profile[40:50] = 1.0
        events = count_width_drop_events(profile, base_width_m=2.0)
        assert len(events) == 1
        assert events[0]["y_start"] == 40
        assert events[0]["y_end"] == 50
        assert events[0]["narrowest_m"] == 1.0

    def test_drop_below_threshold_ignored(self):
        """Drop of 0.5m < 0.6m threshold → no event."""
        profile = np.full(100, 2.0)
        profile[40:50] = 1.5  # only 0.5m drop
        events = count_width_drop_events(profile, base_width_m=2.0)
        assert len(events) == 0

    def test_two_separate_drops(self):
        """Two distinct drop regions."""
        profile = np.full(200, 3.0)
        profile[20:40] = 1.0
        profile[120:150] = 0.5
        events = count_width_drop_events(profile, base_width_m=3.0)
        assert len(events) == 2

    def test_nan_rows_in_drop(self):
        """NaN rows should not break event detection."""
        profile = np.full(100, 2.0)
        profile[40:50] = 1.0
        profile[45] = np.nan  # NaN in the middle of the drop
        events = count_width_drop_events(profile, base_width_m=2.0)
        # The event should still be detected (the NaN row is not dropped, but
        # the rows around it are)
        assert len(events) >= 1

    def test_short_drops_filtered(self):
        """Drop shorter than min_event_rows is ignored."""
        profile = np.full(100, 2.0)
        profile[50:52] = 0.5  # only 2 rows
        events = count_width_drop_events(profile, base_width_m=2.0, min_event_rows=3)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# count_obstacle_encounters
# ---------------------------------------------------------------------------

class TestCountObstacleEncounters:
    def test_no_boxes(self):
        assert count_obstacle_encounters([]) == []

    def test_single_box(self):
        boxes = [{"bbox": (10, 20, 30, 40), "class_name": "bollard"}]
        encounters = count_obstacle_encounters(boxes)
        assert len(encounters) == 1
        assert encounters[0]["box_count"] == 1

    def test_two_close_boxes_merge(self):
        """Two boxes within merge_gap → one encounter."""
        boxes = [
            {"bbox": (10, 20, 30, 40), "class_name": "bollard"},
            {"bbox": (35, 50, 55, 70), "class_name": "tree"},  # 5px gap from first
        ]
        encounters = count_obstacle_encounters(boxes, merge_gap_px=30)
        assert len(encounters) == 1
        assert set(encounters[0]["classes"]) == {"bollard", "tree"}

    def test_two_far_boxes_separate(self):
        """Two boxes far apart → two encounters."""
        boxes = [
            {"bbox": (10, 20, 30, 40), "class_name": "bollard"},
            {"bbox": (200, 50, 220, 70), "class_name": "tree"},
        ]
        encounters = count_obstacle_encounters(boxes, merge_gap_px=30)
        assert len(encounters) == 2


# ---------------------------------------------------------------------------
# compute_walkability_metrics (integration)
# ---------------------------------------------------------------------------

class TestComputeWalkabilityMetrics:
    def test_unobstructed_sidewalk(self):
        """2m sidewalk, no obstacles → min = base, wheelchair pass."""
        mask = np.zeros((500, 480), dtype=bool)
        mask[:, 40:440] = True  # 400px wide
        px_to_m = 2.0 / 400
        metrics = compute_walkability_metrics(mask, [], px_to_m, base_width_m=2.0)
        assert metrics.sidewalk_width_px == 400
        assert metrics.px_to_m == pytest.approx(2.0 / 400)
        assert metrics.min_clear_width_m == pytest.approx(2.0, abs=0.01)
        assert metrics.wheelchair_passable_65cm is True
        assert metrics.ada_accessible_90cm is True
        assert metrics.width_drop_60cm_count == 0
        assert metrics.obstacle_encounter_count == 0

    def test_narrow_point_below_wheelchair(self):
        """Obstacle leaves only 0.5m clear → wheelchair fail."""
        mask = np.zeros((200, 400), dtype=bool)
        mask[:, 0:400] = True  # 400px sidewalk = 2m
        px_to_m = 2.0 / 400  # 0.005 m/px

        # Obstacle spanning cols 0-300 in rows 50-60 → only 100px (0.5m) clear
        boxes = [{"bbox": (50, 0, 60, 300), "class_name": "car"}]
        metrics = compute_walkability_metrics(mask, boxes, px_to_m, base_width_m=2.0)
        assert metrics.min_clear_width_m == pytest.approx(0.5, abs=0.01)
        assert metrics.wheelchair_passable_65cm is False

    def test_to_dict_roundtrip(self):
        """to_dict() produces a JSON-serialisable dict without the raw array."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 10:90] = True
        metrics = compute_walkability_metrics(mask, [], 0.01, 0.8)
        d = metrics.to_dict()
        assert "clear_width_profile_m" not in d
        assert "clear_width_profile_summary" in d
        import json
        json.dumps(d)  # must not raise
