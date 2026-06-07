"""High-level pipeline entry points used by the FastAPI service.

``compute_tile_previews`` produces, for each tile in the strip, the editable
left/right rectification lines plus a preview image (in the frame
``find_row_edges`` operates on). ``build_strip`` runs the full pipeline with any
hand-edited line overrides and returns all output PNGs (also zips + uploads).
"""

from __future__ import annotations

import base64
import io
import zipfile
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

import cv2
import numpy as np

from .config import StripConfig
from .manifest_gcs import GCSClient, load_manifest, parse_gcs_path
from .ordering import order_points, strip_sequence
from .outputs import build_all_outputs
from .rectify import find_row_edges
from .tiles import PipelineContext, edge_frame, load_tile_assets, select_mask


def create_context(cfg: StripConfig) -> PipelineContext:
    bucket = cfg.gcs_bucket_name or (parse_gcs_path(cfg.manifest_csv)[0] or "")
    manifest = load_manifest(cfg.manifest_csv, project_id=cfg.gcp_project_id, bucket_name=bucket)
    cfg.gcs_bucket_name = bucket
    gcs = GCSClient(cfg.gcp_project_id, bucket)
    return PipelineContext(gcs=gcs, manifest=manifest, cfg=cfg)


def _auto_edge_model(frame_mask: np.ndarray) -> dict[str, float]:
    """Auto-fit left/right boundary lines; fall back to vertical mask extremes."""
    _, _, _, _, model = find_row_edges(frame_mask)
    if model is not None:
        return {"a_L": float(model["a_L"]), "b_L": float(model["b_L"]),
                "a_R": float(model["a_R"]), "b_R": float(model["b_R"])}
    H, W = frame_mask.shape[:2]
    cols = np.where(frame_mask.any(axis=0))[0]
    lo = float(cols[0]) if len(cols) else 0.0
    hi = float(cols[-1]) if len(cols) else float(W - 1)
    return {"a_L": 0.0, "b_L": lo, "a_R": 0.0, "b_R": hi}


def _valid_band(frame_mask: np.ndarray) -> tuple[int, int]:
    """Row range [yTop, yBottom] where the sidewalk mask has content.

    Handles are placed on this band (not rows 0/H-1) so drag points land on the
    sidewalk even when the fitted line is steep and exits the frame elsewhere.
    """
    H = frame_mask.shape[0]
    rows = np.where(frame_mask.any(axis=1))[0]
    if len(rows) >= 2:
        return int(rows[0]), int(rows[-1])
    return 0, max(1, H - 1)


def _lines_to_endpoints(model: dict[str, float], frame_mask: np.ndarray) -> dict[str, dict[str, float]]:
    y_top, y_bottom = _valid_band(frame_mask)
    return {
        "left": {
            "xTop": model["a_L"] * y_top + model["b_L"], "yTop": float(y_top),
            "xBottom": model["a_L"] * y_bottom + model["b_L"], "yBottom": float(y_bottom),
        },
        "right": {
            "xTop": model["a_R"] * y_top + model["b_R"], "yTop": float(y_top),
            "xBottom": model["a_R"] * y_bottom + model["b_R"], "yBottom": float(y_bottom),
        },
    }


def _side_coeffs(handle: dict[str, float]) -> tuple[float, float]:
    """Two handle points (xTop,yTop)-(xBottom,yBottom) -> (a, b) for x = a*r + b."""
    y_top = float(handle["yTop"])
    y_bottom = float(handle["yBottom"])
    dy = (y_bottom - y_top) or 1.0
    a = (float(handle["xBottom"]) - float(handle["xTop"])) / dy
    b = float(handle["xTop"]) - a * y_top
    return a, b


def endpoints_to_lines(lines: dict[str, dict[str, float]]) -> dict[str, float]:
    """Inverse of :func:`_lines_to_endpoints`: handle points -> a/b coefficients."""
    a_L, b_L = _side_coeffs(lines["left"])
    a_R, b_R = _side_coeffs(lines["right"])
    return {"a_L": a_L, "b_L": b_L, "a_R": a_R, "b_R": b_R}


def _render_preview(frame_img: np.ndarray, frame_mask: np.ndarray, model: dict[str, float]) -> np.ndarray:
    if frame_img.ndim == 2:
        frame_img = np.repeat(frame_img[:, :, None], 3, axis=2)
    out = frame_img.copy()
    H, W = out.shape[:2]
    # Tint the sidewalk mask for context.
    tint = out.copy()
    tint[frame_mask] = (tint[frame_mask] * 0.5 + np.array([60, 180, 255]) * 0.5).astype(np.uint8)
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)

    def draw(a: float, b: float, color: tuple[int, int, int]) -> None:
        x_top = int(round(b))
        x_bottom = int(round(a * (H - 1) + b))
        cv2.line(out, (x_top, 0), (x_bottom, H - 1), color, 2, cv2.LINE_AA)
        cv2.circle(out, (x_top, 0), 6, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x_bottom, H - 1), 6, color, -1, cv2.LINE_AA)

    draw(model["a_L"], model["b_L"], (60, 220, 90))   # left = green
    draw(model["a_R"], model["b_R"], (255, 90, 90))   # right = red
    return out


def _png_b64(arr: np.ndarray) -> str:
    from .outputs import png_bytes
    return base64.b64encode(png_bytes(arr)).decode("ascii")


def compute_tile_previews(ctx: PipelineContext, point_ids: list[str]) -> dict[str, Any]:
    ordered = order_points(ctx.manifest, point_ids)
    tiles: list[dict[str, Any]] = []
    seen: set[str] = set()

    for side in ctx.cfg.sides_to_render:
        for spec in strip_sequence(side, ordered):
            if spec.tile_id in seen:
                continue
            seen.add(spec.tile_id)
            entry: dict[str, Any] = {
                "tileId": spec.tile_id,
                "side": spec.side_strip,
                "pointId": spec.point_id,
                "direction": spec.direction,
                "selectedSide": spec.selected_side,
                "method": spec.method,
            }
            try:
                item = load_tile_assets(ctx, spec.point_id, spec.direction)
                mask_item = select_mask(item["masks"], spec.selected_side, spec.direction)
                if mask_item is None:
                    raise ValueError("no sidewalk mask for this tile")
                frame_img, frame_mask = edge_frame(item["image"], mask_item["mask"], spec.direction, spec.method)
                model = _auto_edge_model(frame_mask)
                H, W = frame_mask.shape[:2]
                entry["frame"] = {"w": int(W), "h": int(H)}
                entry["lines"] = _lines_to_endpoints(model, frame_mask)
                entry["previewPng"] = _png_b64(frame_img if frame_img.ndim == 3 else frame_img)
                entry["previewOverlayPng"] = _png_b64(_render_preview(frame_img, frame_mask, model))
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            tiles.append(entry)

    return {"order": [t["tileId"] for t in tiles], "orderedPoints": ordered, "tiles": tiles}


def _overrides_to_coeffs(ctx: PipelineContext, point_ids: list[str],
                         overrides: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Map tileId -> a/b coefficients, using each tile's frame height for conversion."""
    if not overrides:
        return {}
    # Frame heights per tile come from the source image shape (rotated for side-view).
    coeffs: dict[str, dict[str, float]] = {}
    ordered = order_points(ctx.manifest, point_ids)
    spec_by_id = {}
    for side in ctx.cfg.sides_to_render:
        for spec in strip_sequence(side, ordered):
            spec_by_id[spec.tile_id] = spec

    for tile_id, lines in overrides.items():
        if tile_id not in spec_by_id:
            continue
        try:
            coeffs[tile_id] = endpoints_to_lines(lines)
        except Exception:
            continue
    return coeffs


def build_strip(ctx: PipelineContext, point_ids: list[str],
                overrides: dict[str, dict[str, Any]] | None = None,
                upload: bool = True) -> dict[str, Any]:
    coeffs = _overrides_to_coeffs(ctx, point_ids, overrides or {})
    outputs = build_all_outputs(ctx, point_ids, coeffs)

    strip_id = _make_strip_id(point_ids)
    gcs_prefix = ""
    if upload:
        gcs_prefix = f"{ctx.cfg.strips_gcs_prefix.rstrip('/')}/{strip_id}/"
        for name, data in outputs.items():
            ctx.gcs.upload_bytes(f"{gcs_prefix}{name}", data, content_type="image/png")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in outputs.items():
            zf.writestr(name, data)
    return {"strip_id": strip_id, "gcs_prefix": gcs_prefix, "filenames": sorted(outputs), "zip": zip_buf.getvalue()}


def _make_strip_id(point_ids: list[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = sha1((",".join(point_ids)).encode("utf-8")).hexdigest()[:8]
    return f"{ts}_{digest}"
