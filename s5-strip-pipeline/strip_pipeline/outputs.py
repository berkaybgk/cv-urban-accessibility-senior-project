"""Assemble all strip output images for one run. Combines cells 12, 14, 17, 19, 21."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from .footprints import collect_footprint_boxes, make_footprint_debug_strip, render_footprint_box_strip
from .merge_loftr import make_full_tile_loftr_debug_strip, merge_side_strip
from .ordering import order_points, strip_sequence
from .tiles import PipelineContext, TileResult, build_tile, normalize_canvas_width, warning_tile, select_mask, load_tile_assets


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _calculate_average_sidewalk_width(ctx: PipelineContext, side: str, ordered_points: list[str]) -> tuple[float, dict[str, bytes]]:
    from .width_estimation import find_horizontal_boundaries, calculate_width, generate_width_debug_plot, FOV_DEG, PITCH_DEG, CAM_HEIGHT_M
    import math
    
    successful_widths = []
    debug_imgs = {}
    for pid in ordered_points:
        try:
            item = load_tile_assets(ctx, pid, side)
            masks = item["masks"]
            mask_item = select_mask(masks, "largest" if side in {"left", "right"} else side, side)
            if mask_item is not None:
                mask = mask_item["mask"]
                boundary = find_horizontal_boundaries(mask)
                if boundary.success:
                    width, z_wall, z_curb = calculate_width(
                        y_wall=boundary.y_upper_mid,
                        y_curb=boundary.y_lower_mid,
                        image_height=mask.shape[0]
                    )
                    if width > 0 and not math.isinf(z_wall) and not math.isinf(z_curb):
                        successful_widths.append(width)
                        # Generate step-by-step pinhole geometry debug plot
                        debug_plot_bytes = generate_width_debug_plot(
                            image_rgb=item["image"],
                            mask=mask,
                            boundary=boundary,
                            width=width,
                            z_wall=z_wall,
                            z_curb=z_curb,
                            fov_deg=FOV_DEG,
                            pitch_deg=PITCH_DEG,
                            cam_height=CAM_HEIGHT_M,
                            point_id=pid,
                            side=side
                        )
                        debug_imgs[f"{side}_width_debug_{pid}.png"] = debug_plot_bytes
        except Exception:
            pass

    avg_w = float(np.mean(successful_widths)) if successful_widths else 2.0
    return avg_w, debug_imgs


def build_side_tiles(ctx: PipelineContext, side: str, ordered_points: list[str],
                     overrides: dict[str, dict[str, float]], avg_sidewalk_width_m: float) -> list[TileResult]:
    """Build every TileResult for one side, bottom-to-top, honoring line overrides."""
    results: list[TileResult] = []
    for spec in strip_sequence(side, ordered_points):
        results.append(build_tile(
            ctx, spec.side_strip, spec.point_id, spec.direction, spec.selected_side, spec.method,
            edge_override=overrides.get(spec.tile_id), flip_180=spec.flip_180,
            avg_sidewalk_width_m=avg_sidewalk_width_m
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
        avg_w, debug_imgs = _calculate_average_sidewalk_width(ctx, side, ordered)
        outputs.update(debug_imgs)
        tiles = build_side_tiles(ctx, side, ordered, overrides, avg_w)

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

            # --- Walkability metrics ---
            from .walkability_metrics import compute_walkability_metrics, generate_metrics_debug_strip
            H, W = clean.shape[:2]
            sidewalk_mask, _ = collect_footprint_boxes(segments, (H, W))
            px_to_m = avg_w / ctx.cfg.target_sidewalk_width_px
            metrics = compute_walkability_metrics(sidewalk_mask, boxes, px_to_m, avg_w)
            metrics_debug = generate_metrics_debug_strip(rendered, metrics, px_to_m)
            outputs[f"{side}_sidewalk_strip_WALKABILITY_debug.png"] = png_bytes(metrics_debug)

            # Separate METRICS.json
            import json
            outputs[f"{side}_sidewalk_strip_METRICS.json"] = json.dumps(
                metrics.to_dict(), indent=2
            ).encode("utf-8")

            H, W = clean.shape[:2]
            metadata = {
                "width": int(W),
                "height": int(H),
                "boxes": boxes,
                "target_sidewalk_width_px": ctx.cfg.target_sidewalk_width_px,
                "avg_sidewalk_width_m": avg_w,
            }
            outputs[f"{side}_sidewalk_strip_METADATA.json"] = json.dumps(metadata, indent=2).encode("utf-8")

    return outputs
