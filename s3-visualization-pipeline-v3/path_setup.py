"""
Register sibling pipeline folders on sys.path.

The repo uses hyphenated directory names (e.g. s3-visualization-pipeline-v2), which
are not valid Python package names. Import gcs_utils and other v2 modules after
calling ensure_v2_pipeline_on_path().
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_v2_pipeline_on_path() -> Path:
    """
    Insert ``s3-visualization-pipeline-v2`` at the front of sys.path and return
    that directory. Safe to call multiple times.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    v2_dir = repo_root / "s3-visualization-pipeline-v2"
    if not v2_dir.is_dir():
        raise FileNotFoundError(
            f"Expected s3-visualization-pipeline-v2 at {v2_dir}. "
            "Clone the full repo and run notebooks from this folder."
        )
    p = str(v2_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    return v2_dir
