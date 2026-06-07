"""Assemble all strip output images for one run. Combines cells 12, 14, 17, 19, 21."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from .footprints import make_footprint_debug_strip, render_footprint_box_strip
from .merge_loftr import make_full_tile_loftr_debug_strip, merge_side_strip
from .ordering import order_points, strip_sequence
from .tiles import PipelineContext, TileResult, build_tile, normalize_canvas_width, warning_tile


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def build_side_tiles(ctx: PipelineContext, side: str, ordered_points: list[str],
                     overrides: dict[str, dict[str, float]]) -> list[TileResult]:
    """Build every TileResult for one side, bottom-to-top, honoring line overrides."""
    results: list[TileResult] = []
    for spec in strip_sequence(side, ordered_points):
        results.append(build_tile(
            ctx, spec.side_strip, spec.point_id, spec.direction, spec.selected_side, spec.method,
            edge_override=overrides.get(spec.tile_id), flip_180=spec.flip_180,
        ))
    return results


def stack_full_strip(tiles: list[TileResult], canvas_width: int) -> np.ndarray:
    imgs = [t.image for t in tiles]
    if not imgs:
        return warning_tile(canvas_width, "No tiles")
    max_w = max(t.shape[1] for t in imgs)
    normalized = [normalize_canvas_width(t, max_w) for t in imgs]
    return np.vstack(list(reversed(normalized)))


def build_all_outputs(ctx: PipelineContext, point_ids: list[str],
                      overrides: dict[str, dict[str, float]] | None = None) -> dict[str, bytes]:
    """Run the full pipeline and return ``{filename: png_bytes}`` for every output."""
    overrides = overrides or {}
    canvas_width = ctx.cfg.canvas_width
    ordered = order_points(ctx.manifest, point_ids)

    outputs: dict[str, bytes] = {}
    for side in ctx.cfg.sides_to_render:
        tiles = build_side_tiles(ctx, side, ordered, overrides)

        full = stack_full_strip(tiles, canvas_width)
        outputs[f"{side}_sidewalk_strip_FULL.png"] = png_bytes(full)

        clean, debug, logs, segments = merge_side_strip(side, tiles, canvas_width, return_segments=True)
        outputs[f"{side}_sidewalk_strip_MERGED.png"] = png_bytes(clean)
        outputs[f"{side}_sidewalk_strip_MERGED_debug.png"] = png_bytes(debug)

        loftr = make_full_tile_loftr_debug_strip(side, tiles, logs, canvas_width)
        outputs[f"{side}_sidewalk_strip_LOFTR_full_tile_matches.png"] = png_bytes(loftr)

        if segments:
            rendered, boxes = render_footprint_box_strip(clean, segments)
            fp_debug = make_footprint_debug_strip(rendered, segments, boxes)
            outputs[f"{side}_sidewalk_strip_FOOTPRINT_BOXES.png"] = png_bytes(rendered)
            outputs[f"{side}_sidewalk_strip_FOOTPRINT_BOXES_debug.png"] = png_bytes(fp_debug)

    return outputs
