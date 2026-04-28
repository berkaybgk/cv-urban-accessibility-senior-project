"""
Configuration loader — same sections as s3-visualization-pipeline (v1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
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
            "output_prefix": "visualization-results/",
            "manifest_blob": "",
        },
        "batch": {
            "enabled": False,
            "prefix_min": 0,
            "prefix_max": 9999,
            # Resume: effective min = max(prefix_min, start_prefix) when set.
            "start_prefix": None,
        },
        "camera": {
            "height_m": 1.8,
            "hfov_deg": 90,
            "pitch_deg": 0,
        },
        "footprint": {
            "obstacle_is_tree": ["tree"],
            "base_scan_ratio": 0.15,
            "trunk_scan_ratio": 0.40,
            "aspect_ratio": 1.0,
            "max_height_px": 25,
        },
        "width_estimation": {
            "min_horizon_dist": 20,
            "exclude_extrapolated": True,
            "iqr_factor": 1.5,
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
