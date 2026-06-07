"""FastAPI service exposing the continuous sidewalk strip pipeline.

Endpoints
---------
POST /strip/edges   -> per-tile editable rectification lines + preview images
POST /strip/create  -> build all strip outputs, upload to GCS, return a zip

Run:  uvicorn service:app --reload --port 8000   (from s5-strip-pipeline/)
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from strip_pipeline import StripConfig, build_strip, compute_tile_previews, create_context

app = FastAPI(title="Sidewalk Strip Builder")

# The Next.js app proxies server-side, but allow direct browser calls in dev too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Strip-Id", "X-Strip-Gcs-Prefix"],
)


class EdgesRequest(BaseModel):
    pointIds: list[str]
    manifestCsv: str | None = None
    targetWidth: int | None = None


class CreateRequest(BaseModel):
    pointIds: list[str]
    manifestCsv: str | None = None
    targetWidth: int | None = None
    overrides: dict[str, dict[str, dict[str, float]]] = {}


def _config(manifest_csv: str | None, target_width: int | None) -> StripConfig:
    return StripConfig.from_env(
        manifest_csv=manifest_csv,
        target_sidewalk_width_px=target_width,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/strip/edges")
def strip_edges(req: EdgesRequest) -> dict[str, Any]:
    if not req.pointIds:
        raise HTTPException(status_code=400, detail="pointIds is required")
    cfg = _config(req.manifestCsv, req.targetWidth)
    try:
        ctx = create_context(cfg)
        return compute_tile_previews(ctx, req.pointIds)
    except Exception as exc:  # surface as 500 with a useful message
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/strip/create")
def strip_create(req: CreateRequest) -> Response:
    if not req.pointIds:
        raise HTTPException(status_code=400, detail="pointIds is required")
    cfg = _config(req.manifestCsv, req.targetWidth)
    try:
        ctx = create_context(cfg)
        result = build_strip(ctx, req.pointIds, overrides=req.overrides, upload=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return Response(
        content=result["zip"],
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="strip_{result["strip_id"]}.zip"',
            "X-Strip-Id": result["strip_id"],
            "X-Strip-Gcs-Prefix": result["gcs_prefix"],
        },
    )
