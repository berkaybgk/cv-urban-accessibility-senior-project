"""
Configuration loader for the sidewalk analysis pipeline v2.

Reads config.yaml and merges with defaults so every key is always present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the pipeline configuration."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    defaults: dict[str, Any] = {
        "gcs": {
            "bucket_name": "",
            "project_id": "",
            "image_prefix": "",
            "masks_prefix": "segmentation-results/",
            "output_prefix": "analysis-results-v2/",
            "manifest_blob": "",
        },
        "batch": {
            "enabled": False,
            "prefix_min": 0,
            "prefix_max": 9999,
        },
        "camera": {
            "height_m": 2.5,
            "hfov_deg": 90,
            "pitch_deg": 0,
        },
        "width_estimation": {
            "min_horizon_dist": 10,
            "exclude_extrapolated": False,
        },
        "sidewalk": {
            "min_valid_rows": 10,
            "gap_fill_px": 5,
            "iqr_factor": 1.5,
        },
        "obstacle": {
            "is_tree": ["tree"],
        },
        "obstacle_footprint": {
            "base_scan_ratio": 0.15,
            "trunk_scan_ratio": 0.40,
            "aspect_ratio": 1.0,
            "max_height": 25,
        },
        "local_output_dir": None,
    }

    for section, section_defaults in defaults.items():
        if isinstance(section_defaults, dict):
            cfg.setdefault(section, {})
            for k, v in section_defaults.items():
                cfg[section].setdefault(k, v)
        else:
            cfg.setdefault(section, section_defaults)

    return cfg
