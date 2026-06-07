"""Configuration for the continuous sidewalk strip pipeline.

Ported from cell 2 of ``continuous_sidewalk_rectification_strip.ipynb``. The
notebook referenced these as module-level globals, so they are kept as module
constants here (functions in :mod:`rectify`, :mod:`tiles`, etc. read them the
same way). Per-request overridable values live on :class:`StripConfig`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Fixed algorithm constants (ported verbatim from the notebook config cell)
# -----------------------------------------------------------------------------
RECTIFY_INTERPOLATION = cv2.INTER_CUBIC
DEPTH_RATIO = 1.6
# Along-walk resolution multiplier for left/right side-view tiles. 1.0 = original
# (out_height == source image width); >1 resamples finer, <1 coarser. Pure
# resolution knob (no metric meaning), analogous to pixels_per_meter for road tiles.
SIDE_ALONG_WALK_SCALE = 4.0
HFOV_DEG = 90
LABEL_BAR_HEIGHT = 54
WARNING_TILE_HEIGHT = 180
OBSTACLE_IS_TREE = {"tree"}
FOOTPRINT_BASE_SCAN_RATIO = 0.15
TREE_TRUNK_SCAN_RATIO = 0.40
FOOTPRINT_ASPECT_RATIO = 1.0
FOOTPRINT_MAX_HEIGHT = 25

# Edge fitting (cell 6)
BORDER_MARGIN = 3
RANSAC_RESIDUAL_THRESHOLD = 1.0
RANSAC_MIN_SAMPLES = 0.3
SHIFT_PERCENTILE = 10
ROAD_TOUCH_MARGIN = 20
ROAD_CONFIRMED_WEIGHT = 5.0

# Robust horizon rectifier for road-facing forward/backward tiles.
USE_ROBUST_RECTIFIER = True
ROBUST_CAMERA_HEIGHT_M = 2.5
ROBUST_PIXELS_PER_METER = 213.0
ROBUST_Z_MAX_M = 30.0
ROBUST_BOUNDARY_THICKNESS_PX = 3
ROBUST_MIN_MASK_AREA = 1500
ROBUST_MIN_ROW_COVERAGE = 0.05
ROBUST_MAX_OUTPUT_WIDTH = 1600

# LoFTR merge (cells 16-17)
LOFTR_WEIGHTS = "outdoor"
LOFTR_LONG_SIDE = 840
MIN_CONFIDENCE = 0.5
RESTRICT_TO_SIDEWALK = True
ROAD_VIEW_MATCH_KEEP_RATIO = 0.50
SIDE_VIEW_PAIR_KEEP_RATIO = 0.50
ROAD_SIDE_PAIR_KEEP_RATIO = 0.40
ROAD_ROAD_PAIR_KEEP_RATIO = 1.0
MIN_RANSAC_INLIERS = 4

# Defaults that the service may override per request.
DEFAULT_TARGET_SIDEWALK_WIDTH_PX = 480
DEFAULT_MASKS_ROOT = "v4-20260516T081917Z/segmentation-results"
DEFAULT_MANIFEST_CSV = (
    "https://storage.cloud.google.com/cv-urban-accessibility-bucket/"
    "streetview/polygon_4v/20260516T081917Z/manifest.csv"
)


def _load_env() -> None:
    """Load the repo-level .env so GCS credentials/bucket resolve like the notebook."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            break


@dataclass
class StripConfig:
    """Per-request configuration. Fixed algorithm constants stay module-level."""

    manifest_csv: str = DEFAULT_MANIFEST_CSV
    masks_root: str = DEFAULT_MASKS_ROOT
    target_sidewalk_width_px: int = DEFAULT_TARGET_SIDEWALK_WIDTH_PX
    sides_to_render: tuple[str, ...] = ("left", "right")
    gcp_project_id: str = ""
    gcs_bucket_name: str = ""
    strips_gcs_prefix: str = "strips"

    @property
    def canvas_width(self) -> int:
        return self.target_sidewalk_width_px + 2 * int(self.target_sidewalk_width_px * 0.3)

    @classmethod
    def from_env(cls, **overrides) -> "StripConfig":
        _load_env()
        cfg = cls(
            gcp_project_id=os.environ.get("GCP_PROJECT_ID", ""),
            gcs_bucket_name=os.environ.get("GCS_BUCKET_NAME", ""),
            strips_gcs_prefix=os.environ.get("GCS_PREFIX_STRIPS", "strips"),
        )
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
