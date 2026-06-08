#!/usr/bin/env python3
"""Batch-convert strip METRICS.json files into one map-ready scored GeoJSON.

Each input is a ``*_METRICS.json`` produced by the strip pipeline
(``strip_pipeline/outputs.py``). Since that file embeds the strip's street
geometry (a ``geometry`` LineString of ordered [lon, lat] points), we can turn a
batch of them into a single GeoJSON FeatureCollection the web app's
"Accessibility map" mode loads.

Each feature carries BOTH scores so one upload drives both maps via the app's
metric toggle:

  * ``walkability_score`` — graded [0, 1] from the number of significant
    (>= 60 cm) clear-width drops. Fewer drops = higher (greener) score.
  * ``wheelchair_score``  — binary 1.0 (passable) / 0.0 (not) from the
    ``wheelchair_passable_65cm`` flag.

``score`` mirrors ``walkability_score`` (the default metric).

Note: the strip pipeline now also auto-emits a per-strip
``*_sidewalk_strip_ACCESSIBILITY.geojson`` for every strip it builds (same
feature shape). This batch CLI is for converting a folder of METRICS.json files
in one shot; either output can be multi-uploaded to the web app.

Usage
-----
    python build_accessibility_geojson.py path/to/*_METRICS.json \\
        --out-dir ./maps

    # or point at directories (recurses for *_METRICS.json)
    python build_accessibility_geojson.py ./run_outputs --out-dir ./maps
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

# Load the shared scoring module by file path so we don't import the heavy
# ``strip_pipeline`` package (cv2/torch/etc) just to score JSON files.
_acc_path = Path(__file__).resolve().parent / "strip_pipeline" / "accessibility.py"
_spec = importlib.util.spec_from_file_location("strip_accessibility", _acc_path)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"could not load scoring module at {_acc_path}")
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)

accessibility_feature = _acc.accessibility_feature
has_drawable_geometry = _acc.has_drawable_geometry
walkability_score = _acc.walkability_score
wheelchair_score = _acc.wheelchair_score


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def gather_metric_files(inputs: list[str]) -> list[Path]:
    """Expand the CLI inputs (files and/or directories) into METRICS.json paths."""
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*_METRICS.json")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: no such path: {p}", file=sys.stderr)
    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return unique


def segment_id(path: Path, metrics: dict[str, Any]) -> str:
    """Stable id for a strip: '<folder>-<side>' falling back to the file stem."""
    explicit = metrics.get("id")
    if explicit:
        return str(explicit)
    side = str(metrics.get("side", "")).strip()
    parent = path.parent.name
    base = parent if parent not in ("", ".", "/") else path.stem.replace("_METRICS", "")
    return f"{base}-{side}" if side else base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="METRICS.json files and/or directories to recurse")
    ap.add_argument("--out-dir", default=".",
                    help="output directory for the combined GeoJSON file")
    ap.add_argument("--out", default="accessibility.geojson",
                    help="filename for the combined scored GeoJSON")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = gather_metric_files(args.inputs)
    if not files:
        print("error: no METRICS.json files found", file=sys.stderr)
        return 1

    features: list[dict[str, Any]] = []
    skipped = 0

    for path in files:
        try:
            metrics = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if not has_drawable_geometry(metrics):
            print(f"warning: skipping {path}: no LineString geometry with >= 2 points "
                  "(re-run the strip pipeline so METRICS.json includes geometry)",
                  file=sys.stderr)
            skipped += 1
            continue

        features.append(accessibility_feature(metrics, segment_id(path, metrics)))

    if not features:
        print("error: no usable strips with geometry", file=sys.stderr)
        return 1

    fc = {"type": "FeatureCollection", "name": "accessibility", "features": features}
    out_path = out_dir / args.out
    out_path.write_text(json.dumps(fc, indent=2))

    n = len(features)
    wc_pass = sum(1 for f in features if f["properties"]["wheelchair_score"] == 1.0)
    print(f"wrote {out_path}  ({n} segments; both walkability + wheelchair scores)")
    print(f"  wheelchair: {wc_pass} passable, {n - wc_pass} blocked")
    if skipped:
        print(f"skipped {skipped} file(s) without usable geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
