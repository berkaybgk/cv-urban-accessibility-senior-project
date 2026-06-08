# s5-strip-pipeline

Continuous sidewalk strip pipeline, extracted from
`s0-trials/sidewalk-measurement-trials/continuous_sidewalk_rectification_strip.ipynb`
into a reusable package (`strip_pipeline/`) and a FastAPI service (`service.py`).

It powers the web app's **Strip mode**: pick points on the map, review/drag the
per-tile rectification lines, then create + download the merged strip outputs.

## Layout

| File | Notebook source | Role |
|---|---|---|
| `strip_pipeline/config.py` | cell 2 | `StripConfig` + algorithm constants |
| `strip_pipeline/manifest_gcs.py` | cell 4 | manifest parsing, GCS IO, mask loading |
| `strip_pipeline/rectify.py` | cell 6 | edge fitting + rectifiers (`edge_override` added) |
| `strip_pipeline/tiles.py` | cell 8 | tile building (`PipelineContext`, `edge_override`, `flip_180`) |
| `strip_pipeline/ordering.py` | new (replaces cell 10) | heading-aware point ordering |
| `strip_pipeline/merge_loftr.py` | cells 16, 17, 19 | LoFTR seam merge + diagnostics |
| `strip_pipeline/footprints.py` | cell 21 | continuous footprint boxes |
| `strip_pipeline/outputs.py` | cells 12, 14, 17, 19, 21 | assemble all output PNGs |
| `strip_pipeline/pipeline.py` | orchestration | `compute_tile_previews`, `build_strip` |
| `service.py` | — | FastAPI endpoints |

## Setup

```bash
cd s5-strip-pipeline
python -m venv .venv && source .venv/bin/activate   # or reuse repo .venv
pip install -r requirements.txt
```

GCS credentials + bucket come from the repo-level `.env` (`GCP_PROJECT_ID`,
`GCS_BUCKET_NAME`, optional `GCS_PREFIX_STRIPS`), loaded automatically.

## Run the service

```bash
python -m uvicorn service:app --port 8000
```

Point the web app at it with `STRIP_SERVICE_URL=http://localhost:8000`.

### Endpoints

- `POST /strip/edges` — body `{ "pointIds": ["0010","0011"], "manifestCsv?": "...", "targetWidth?": 480 }`.
  Returns `{ order, orderedPoints, tiles:[{ tileId, side, pointId, direction, method,
  frame:{w,h}, previewPng, previewOverlayPng, lines:{left:{xTop,xBottom}, right:{xTop,xBottom}} }] }`.
- `POST /strip/create` — body `{ pointIds, manifestCsv?, targetWidth?, overrides:{ tileId:{left:{xTop,xBottom},right:{xTop,xBottom}} } }`.
  Returns a zip of all output PNGs; also uploads them to `gs://<bucket>/strips/<id>/`.
  Response headers: `X-Strip-Id`, `X-Strip-Gcs-Prefix`.

## Use as a library

```python
from strip_pipeline import StripConfig, create_context, build_strip
ctx = create_context(StripConfig.from_env())
result = build_strip(ctx, ["0010", "0011", "0012"], overrides={}, upload=False)
# result["zip"] holds the bundled PNGs; result["filenames"] lists them.
```
