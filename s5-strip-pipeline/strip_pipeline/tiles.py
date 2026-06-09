"""Tile building and labeling. Ported from cell 8 of the notebook.

Notebook globals (``manifest``, ``gcs``, ``MASKS_ROOT``, ``RobustHorizonRectifier``,
``robust_rectifiers_by_side``, ``TARGET_SIDEWALK_WIDTH_PX``) are replaced by a
:class:`PipelineContext`. An optional ``edge_override`` (hand-edited left/right
boundary lines, in the frame :func:`find_row_edges` operates on) is threaded
through every rectification branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    DEPTH_RATIO,
    HFOV_DEG,
    LABEL_BAR_HEIGHT,
    OBSTACLE_IS_TREE,
    ROBUST_BOUNDARY_THICKNESS_PX,
    ROBUST_CAMERA_HEIGHT_M,
    ROBUST_MAX_OUTPUT_WIDTH,
    ROBUST_MIN_MASK_AREA,
    ROBUST_MIN_ROW_COVERAGE,
    ROBUST_PIXELS_PER_METER,
    ROBUST_Z_MAX_M,
    RECTIFY_INTERPOLATION,
    USE_ROBUST_RECTIFIER,
    WARNING_TILE_HEIGHT,
    FOOTPRINT_ASPECT_RATIO,
    FOOTPRINT_BASE_SCAN_RATIO,
    FOOTPRINT_MAX_HEIGHT,
    TREE_TRUNK_SCAN_RATIO,
    StripConfig,
)
from .manifest_gcs import (
    GCSClient,
    bytes_to_image,
    load_individual_sidewalk_masks,
    load_obstacle_masks,
    normalize_point_id,
    resolve_masks_prefix,
)
from .rectify import (
    _compute_rectify_params,
    find_row_edges,
    rectify_side_datadriven_fan,
    rectify_sidewalk,
    rotate_code_for_side,
)
from .robust_rectifier import RobustHorizonRectifier


@dataclass
class PipelineContext:
    """Bundles per-run state previously held in notebook globals."""

    gcs: GCSClient
    manifest: dict[str, dict[str, dict[str, Any]]]
    cfg: StripConfig
    tile_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    rectifiers_by_side: dict[str, Any] = field(default_factory=dict)


@dataclass
class TileResult:
    image: np.ndarray
    status: str
    side_strip: str
    point_id: str
    direction: str
    selected_side: str
    method: str
    image_blob: str = ""
    mask_blob: str = ""
    message: str = ""
    shape: tuple[int, ...] | None = None
    clean_image: Any = None
    clean_mask: Any = None
    footprints: Any = None
    meta: dict[str, Any] | None = None


def make_edge_model(override: dict[str, float]) -> dict[str, Any]:
    """Wrap hand-edited line coefficients into the dict rectify functions expect."""
    return {
        "a_L": float(override["a_L"]),
        "b_L": float(override["b_L"]),
        "a_R": float(override["a_R"]),
        "b_R": float(override["b_R"]),
    }


def line_boundary_mask(a: float, b: float, shape: tuple[int, int],
                       thickness: int = ROBUST_BOUNDARY_THICKNESS_PX) -> np.ndarray:
    """Rasterize a boundary line ``x = a*r + b`` into a thick boolean mask."""
    H, W = shape
    out = np.zeros((H, W), dtype=bool)
    half = max(0, int(thickness) // 2)
    rows = np.arange(H)
    cols = np.round(a * rows + b).astype(int)
    for r, c in zip(rows, cols):
        if 0 <= c < W:
            out[r, max(0, c - half):min(W, c + half + 1)] = True
    return out


def select_mask(masks: list[dict[str, Any]], selected_side: str, direction: str) -> dict[str, Any] | None:
    if direction in {"left", "right"} and selected_side == "largest":
        return masks[0] if masks else None
    side_matches = [m for m in masks if m["side"] == selected_side]
    return side_matches[0] if side_matches else None


def sidewalk_boundary_masks(mask: np.ndarray, thickness: int = ROBUST_BOUNDARY_THICKNESS_PX) -> tuple[np.ndarray, np.ndarray]:
    H, W = mask.shape[:2]
    left = np.zeros((H, W), dtype=bool)
    right = np.zeros((H, W), dtype=bool)
    half = max(0, int(thickness) // 2)
    for row in range(H):
        cols = np.where(mask[row])[0]
        if len(cols) == 0:
            continue
        c_left = int(cols[0])
        c_right = int(cols[-1])
        left[row, max(0, c_left - half):min(W, c_left + half + 1)] = True
        right[row, max(0, c_right - half):min(W, c_right + half + 1)] = True
    return left, right


def robust_road_edge_masks(selected_side: str, left_boundary: np.ndarray, right_boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if selected_side == "left":
        return right_boundary, left_boundary
    if selected_side == "right":
        return left_boundary, right_boundary
    raise ValueError(f"Robust road-facing rectification needs selected_side left/right, got {selected_side!r}")


def robust_mask_quality(mask: np.ndarray) -> dict[str, Any]:
    row_coverage = float(np.mean(mask.any(axis=1))) if mask.size else 0.0
    return {"mask_area": int(mask.sum()), "mask_row_coverage": row_coverage}


def remap_with_robust_warp(image_or_mask: np.ndarray, warp: dict[str, Any], is_mask: bool = False) -> np.ndarray:
    flags = cv2.INTER_NEAREST if is_mask else RECTIFY_INTERPOLATION
    inp = image_or_mask.astype(np.uint8) if is_mask else image_or_mask
    out = cv2.remap(inp, warp["map_x"], warp["map_y"], flags, borderMode=cv2.BORDER_CONSTANT)
    return out.astype(bool) if is_mask else out


def rectify_road_robust(img: np.ndarray, mask: np.ndarray, selected_side: str,
                        rectifier: Any, target_width: int,
                        edge_override: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if rectifier is None:
        raise RuntimeError("Robust rectifier is enabled but no rectifier instance was provided")

    quality = robust_mask_quality(mask)
    meta: dict[str, Any] = {"rectifier": "robust-horizon", **quality}
    if quality["mask_area"] < ROBUST_MIN_MASK_AREA or quality["mask_row_coverage"] < ROBUST_MIN_ROW_COVERAGE:
        raise ValueError(
            f"weak sidewalk mask for robust rectification: area={quality['mask_area']} "
            f"row_coverage={quality['mask_row_coverage']:.3f}"
        )

    if edge_override is not None:
        left_boundary = line_boundary_mask(edge_override["a_L"], edge_override["b_L"], mask.shape[:2])
        right_boundary = line_boundary_mask(edge_override["a_R"], edge_override["b_R"], mask.shape[:2])
    else:
        left_boundary, right_boundary = sidewalk_boundary_masks(mask)
    good_mask, bad_mask = robust_road_edge_masks(selected_side, left_boundary, right_boundary)
    f_px = img.shape[1] / (2.0 * np.tan(np.radians(HFOV_DEG / 2.0)))
    warp = rectifier.build_warp(
        good_mask=good_mask,
        bad_mask=bad_mask,
        f_px=f_px,
        camera_height_m=ROBUST_CAMERA_HEIGHT_M,
        pixels_per_meter=ROBUST_PIXELS_PER_METER,
        z_max_m=ROBUST_Z_MAX_M,
        max_output_width=ROBUST_MAX_OUTPUT_WIDTH,
        output_order="far_to_near",
        output_width=target_width,
    )
    meta.update({
        "final_vy": warp.get("final_vy"),
        "local_vy": warp.get("local_vy"),
        "robust_output_shape": warp.get("output_shape"),
        "robust_reason": warp.get("reason"),
        "robust_target_width": warp.get("target_width"),
        "robust_natural_target_width": warp.get("natural_target_width"),
    })
    if not warp.get("ok"):
        raise ValueError(f"robust rectifier skipped tile: {warp.get('reason', 'unknown reason')}")

    rect_img = remap_with_robust_warp(img, warp, is_mask=False)
    rect_mask = remap_with_robust_warp(mask, warp, is_mask=True)
    meta["robust_warp"] = warp
    meta["robust_output_shape"] = rect_img.shape[:2]
    return rect_img, rect_mask, meta


def rectify_tile(img: np.ndarray, mask: np.ndarray, direction: str, method: str,
                 target_width: int,
                 selected_side: str | None = None,
                 rectifier: Any = None,
                 edge_override: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if method == "side-view-fan":
        rect_img, w, pad = rectify_side_datadriven_fan(
            img, mask, direction=direction, target_width=target_width, depth_ratio=DEPTH_RATIO,
            edge_override=edge_override,
        )
        rect_mask, _, _ = rectify_side_datadriven_fan(
            mask, mask, direction=direction, target_width=w, is_mask=True, depth_ratio=DEPTH_RATIO,
            edge_override=edge_override,
        )
        return rect_img, rect_mask, {"rectifier": "side-view-fan"}

    if method == "geometry":
        if USE_ROBUST_RECTIFIER and direction in {"forward", "backward"}:
            if selected_side is None:
                raise ValueError("selected_side is required for robust road-facing rectification")
            return rectify_road_robust(img, mask, selected_side, rectifier, target_width, edge_override=edge_override)

        left, right, valid, extrap, model = find_row_edges(mask)
        if edge_override is not None:
            model = make_edge_model(edge_override)
        f_px, cos_corr = _compute_rectify_params(img, model)
        rect_img, w, pad = rectify_sidewalk(
            img, left, right, valid, edge_model=model, target_width=target_width,
            f_px=f_px, cos_correction=cos_corr
        )
        rect_mask, _, _ = rectify_sidewalk(
            mask, left, right, valid, edge_model=model, target_width=w, is_mask=True,
            f_px=f_px, cos_correction=cos_corr
        )
        return rect_img, rect_mask, {"rectifier": "legacy-geometry"}

    raise ValueError(f"Unknown rectification method: {method}")


def normalize_canvas_width(img: np.ndarray, canvas_width: int) -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    h, w = img.shape[:2]
    if w == canvas_width:
        return img
    if w > canvas_width:
        x0 = (w - canvas_width) // 2
        return img[:, x0:x0 + canvas_width]
    pad_left = (canvas_width - w) // 2
    pad_right = canvas_width - w - pad_left
    return np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)


def normalize_mask_width(mask: np.ndarray, canvas_width: int) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    h, w = mask.shape[:2]
    if w == canvas_width:
        return mask.astype(bool)
    if w > canvas_width:
        x0 = (w - canvas_width) // 2
        return mask[:, x0:x0 + canvas_width].astype(bool)
    pad_left = (canvas_width - w) // 2
    pad_right = canvas_width - w - pad_left
    return np.pad(mask.astype(bool), ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=False)


def _mask_col_range(mask: np.ndarray) -> tuple[int, int] | None:
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1])


def assign_obstacles_to_selected_sidewalk(obstacles: list[dict[str, Any]],
                                          sidewalk_masks: list[dict[str, Any]],
                                          selected_mask_item: dict[str, Any]) -> list[dict[str, Any]]:
    ranges: list[tuple[str, int, int]] = []
    for item in sidewalk_masks:
        col_range = _mask_col_range(item["mask"])
        if col_range is None:
            continue
        ranges.append((item["blob_name"], col_range[0], col_range[1]))
    if not ranges:
        return []

    selected_blob = selected_mask_item["blob_name"]
    assigned: list[dict[str, Any]] = []
    for obs in obstacles:
        col_range = _mask_col_range(obs["mask"])
        if col_range is None:
            continue
        obs_center = (col_range[0] + col_range[1]) / 2.0

        def distance_to_range(item: tuple[str, int, int]) -> float:
            _, c_lo, c_hi = item
            if c_lo <= obs_center <= c_hi:
                return 0.0
            return min(abs(obs_center - c_lo), abs(obs_center - c_hi))

        best_blob, _, _ = sorted(ranges, key=lambda item: (distance_to_range(item), item[0]))[0]
        if best_blob == selected_blob:
            assigned.append(obs)
    return assigned


def rectify_obstacle_mask(mask: np.ndarray, sidewalk_mask: np.ndarray, direction: str, method: str,
                          target_width: int,
                          robust_warp: dict[str, Any] | None = None,
                          edge_override: dict[str, float] | None = None) -> np.ndarray:
    if method == "side-view-fan":
        rect, _, _ = rectify_side_datadriven_fan(
            mask, sidewalk_mask, direction=direction, target_width=target_width,
            is_mask=True, depth_ratio=DEPTH_RATIO, edge_override=edge_override,
        )
        return rect.astype(bool)

    if method == "geometry":
        if USE_ROBUST_RECTIFIER and direction in {"forward", "backward"} and robust_warp is not None:
            return remap_with_robust_warp(mask, robust_warp, is_mask=True).astype(bool)

        left, right, valid, extrap, model = find_row_edges(sidewalk_mask)
        if edge_override is not None:
            model = make_edge_model(edge_override)
        f_px, cos_corr = _compute_rectify_params(mask.astype(np.uint8), model)
        rect, _, _ = rectify_sidewalk(
            mask, left, right, valid, edge_model=model, target_width=target_width,
            is_mask=True, f_px=f_px, cos_correction=cos_corr
        )
        return rect.astype(bool)

    raise ValueError(f"Unknown rectification method: {method}")


def connected_components(mask: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for component_id in range(1, n):
        x, y, w, h, area = stats[component_id]
        if area <= 0 or w <= 0 or h <= 0:
            continue
        yield component_id, labels == component_id, (int(y), int(x), int(y + h), int(x + w))


def component_ground_anchor(component: np.ndarray, direction: str) -> tuple[float, float]:
    rows, cols = np.where(component)
    if len(rows) == 0:
        raise ValueError("Cannot estimate anchor for an empty component")

    if direction == "forward":
        max_row = int(rows.max())
        edge_cols = cols[rows == max_row]
        return float(np.median(edge_cols)), float(max_row)
    if direction == "backward":
        min_row = int(rows.min())
        edge_cols = cols[rows == min_row]
        return float(np.median(edge_cols)), float(min_row)
    if direction == "right":
        min_col = int(cols.min())
        edge_rows = rows[cols == min_col]
        return float(min_col), float(np.median(edge_rows))
    if direction == "left":
        max_col = int(cols.max())
        edge_rows = rows[cols == max_col]
        return float(max_col), float(np.median(edge_rows))

    raise ValueError(f"Unknown direction for footprint anchor: {direction}")


def estimate_width_footprint(mask: np.ndarray, direction: str, is_tree: bool = False,
                             base_scan_ratio: float = FOOTPRINT_BASE_SCAN_RATIO,
                             trunk_scan_ratio: float = TREE_TRUNK_SCAN_RATIO,
                             aspect_ratio: float = FOOTPRINT_ASPECT_RATIO,
                             max_height: int = FOOTPRINT_MAX_HEIGHT) -> tuple[np.ndarray, list[tuple[float, float]]]:
    footprint = np.zeros_like(mask, dtype=bool)
    anchors: list[tuple[float, float]] = []

    for component_id, component, bbox in connected_components(mask):
        min_row, min_col, max_row, max_col = bbox
        height = max_row - min_row
        width = max_col - min_col
        if height <= 0 or width <= 0:
            continue

        scan_ratio = trunk_scan_ratio if is_tree else base_scan_ratio

        if direction in {"forward", "backward"}:
            scan_rows = max(1 if is_tree else 3, int(height * scan_ratio))
            scan_rows = min(scan_rows, height)
            if direction == "forward":
                scan_range = range(max_row - scan_rows, max_row)
            else:
                scan_range = range(min_row, min_row + scan_rows)

            row_widths = []
            row_centers = []
            for r in scan_range:
                cols = np.where(component[r])[0]
                if len(cols) == 0:
                    continue
                row_widths.append(cols[-1] - cols[0] + 1)
                row_centers.append((cols[0] + cols[-1]) / 2.0)
            if not row_widths:
                continue

            fp_width = max(int(np.median(row_widths)), 1)
            fp_height = min(max(1, int(fp_width * aspect_ratio)), height, max_height)

            median_center = float(np.median(row_centers))
            fp_top = max(0, max_row - fp_height) if direction == "forward" else min_row
            fp_bottom = max_row if direction == "forward" else min(mask.shape[0], min_row + fp_height)
            fp_left = max(0, int(median_center - fp_width / 2.0))
            fp_right = min(mask.shape[1], fp_left + fp_width)
            footprint[fp_top:fp_bottom, fp_left:fp_right] = True

        elif direction in {"right", "left"}:
            scan_cols = max(1 if is_tree else 3, int(width * scan_ratio))
            scan_cols = min(scan_cols, width)
            if direction == "right":
                scan_range = range(min_col, min_col + scan_cols)
            else:
                scan_range = range(max_col - scan_cols, max_col)

            col_heights = []
            col_centers = []
            for c in scan_range:
                rows = np.where(component[:, c])[0]
                if len(rows) == 0:
                    continue
                col_heights.append(rows[-1] - rows[0] + 1)
                col_centers.append((rows[0] + rows[-1]) / 2.0)
            if not col_heights:
                continue

            fp_height = max(int(np.median(col_heights)), 1)
            fp_width = min(max(1, int(fp_height * aspect_ratio)), width, max_height)

            median_center = float(np.median(col_centers))
            fp_top = max(0, int(median_center - fp_height / 2.0))
            fp_bottom = min(mask.shape[0], fp_top + fp_height)
            fp_left = min_col if direction == "right" else max(0, max_col - fp_width)
            fp_right = min(mask.shape[1], min_col + fp_width) if direction == "right" else max_col
            footprint[fp_top:fp_bottom, fp_left:fp_right] = True

        else:
            raise ValueError(f"Unknown direction for footprint mask: {direction}")

        anchors.append(component_ground_anchor(component, direction))

    return footprint, anchors


def build_tile_footprints(obstacles: list[dict[str, Any]], sidewalk_mask: np.ndarray,
                          direction: str, method: str, canvas_width: int, target_width: int,
                          robust_warp: dict[str, Any] | None = None,
                          edge_override: dict[str, float] | None = None,
                          flip_180: bool = False) -> list[dict[str, Any]]:
    footprints: list[dict[str, Any]] = []
    for obs in obstacles:
        rect_full = rectify_obstacle_mask(obs["mask"], sidewalk_mask, direction, method, target_width,
                                          robust_warp=robust_warp, edge_override=edge_override)
        if flip_180:
            rect_full = cv2.flip(rect_full.astype(np.uint8), -1).astype(bool)
        rect_full = normalize_mask_width(rect_full, canvas_width)
        is_tree = any(t in obs["class_name"].lower() for t in OBSTACLE_IS_TREE)
        footprint, anchors = estimate_width_footprint(rect_full, direction, is_tree=is_tree)
        if not footprint.any():
            continue
        footprints.append({
            "class_name": obs["class_name"],
            "blob_name": obs["blob_name"],
            "method": "trunk" if is_tree else "base",
            "mask": footprint,
            "anchors": anchors,
            "full_mask": rect_full,
        })
    return footprints


def image_id_from_blob(blob_name: str) -> str:
    stem = Path(blob_name).stem
    m = re.match(r"^(.+?)_(forward|backward|left|right)_", stem)
    return m.group(1) if m else stem


def add_image_badge(img: np.ndarray, image_id: str, direction: str, transform_note: str = "") -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(1.0, w / 720.0))
    thickness = max(1, int(round(font_scale * 2)))
    lines = [f"ID: {image_id}", f"DIR: {direction.upper()}"]
    if transform_note:
        lines.append(transform_note.upper())
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_w = min(w - 16, max(size[0] for size in sizes) + 24)
    box_h = 26 * len(lines) + 18
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (15, 22, 30), -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    y = 34
    for line in lines:
        cv2.putText(out, line, (20, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 26
    return out


def add_label_bar(img: np.ndarray, label: str, status: str = "ok") -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    h, w = img.shape[:2]
    bar = np.zeros((LABEL_BAR_HEIGHT, w, 3), dtype=np.uint8)
    bar[:] = (32, 42, 52) if status == "ok" else (104, 43, 43)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [label]
    max_chars = max(30, int(w / 8.0))
    if len(label) > max_chars:
        lines = [label[:max_chars], label[max_chars:max_chars * 2]]
    y = 20
    for line in lines[:2]:
        cv2.putText(bar, line, (8, y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return np.vstack([bar, img])


def warning_tile(canvas_width: int, label: str) -> np.ndarray:
    tile = np.zeros((WARNING_TILE_HEIGHT, canvas_width, 3), dtype=np.uint8)
    tile[:] = (58, 45, 45)
    cv2.putText(tile, "MISSING / SKIPPED", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(tile, label[:max(20, canvas_width // 8)], (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def load_tile_assets(ctx: PipelineContext, point_id: str, direction: str) -> dict[str, Any]:
    """Download (or cache) image + masks for a point/direction. Returns the cache item."""
    point_id = normalize_point_id(point_id)
    key = (point_id, direction)
    if key not in ctx.tile_cache:
        row = ctx.manifest.get(point_id, {}).get(direction)
        if row is None:
            raise KeyError(f"no manifest row for point={point_id} direction={direction}")
        blob = row["blob_name"]
        parsed = row["parsed"]
        img = bytes_to_image(ctx.gcs.download_as_bytes(blob))
        masks_prefix, coord = resolve_masks_prefix(ctx.gcs, ctx.cfg.masks_root, parsed)
        masks = load_individual_sidewalk_masks(ctx.gcs, masks_prefix, img.shape[:2])
        obstacles = load_obstacle_masks(ctx.gcs, masks_prefix, img.shape[:2])
        ctx.tile_cache[key] = {"row": row, "image": img, "masks": masks, "obstacles": obstacles,
                               "masks_prefix": masks_prefix, "coord": coord}
    return ctx.tile_cache[key]


def edge_frame(img: np.ndarray, mask: np.ndarray, direction: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (image, mask) in the frame ``find_row_edges`` operates on.

    Side-view tiles are rotated 90deg first (matching the fan rectifier); road
    tiles stay upright. Used to render the editable preview.
    """
    if method == "side-view-fan":
        rotate_code = rotate_code_for_side(direction)
        safe = img.astype(np.uint8) if img.dtype == bool else img
        return cv2.rotate(safe, rotate_code), cv2.rotate(mask.astype(np.uint8), rotate_code).astype(bool)
    return img, mask


def build_tile(ctx: PipelineContext, side_strip: str, point_id: str, direction: str,
               selected_side: str, method: str,
               edge_override: dict[str, float] | None = None,
               flip_180: bool | None = None) -> TileResult:
    point_id = normalize_point_id(point_id)
    if flip_180 is None:
        flip_180 = direction == "backward"
    canvas_width = ctx.cfg.canvas_width
    target_width = ctx.cfg.target_sidewalk_width_px
    label_prefix = f"strip={side_strip} | point={point_id} | view={direction} | side={selected_side} | {method}"

    row = ctx.manifest.get(point_id, {}).get(direction)
    if row is None:
        msg = f"{label_prefix} | no manifest row"
        return TileResult(warning_tile(canvas_width, msg), "missing", side_strip, point_id, direction, selected_side, method, message=msg)

    try:
        item = load_tile_assets(ctx, point_id, direction)
        img = item["image"]
        masks = item["masks"]
        obstacles = item.get("obstacles", [])
        mask_item = select_mask(masks, selected_side, direction)
        if mask_item is None:
            msg = f'{label_prefix} | no mask found | prefix={item["masks_prefix"]}/sidewalk/'
            return TileResult(warning_tile(canvas_width, msg), "missing", side_strip, point_id, direction, selected_side, method,
                              image_blob=row["blob_name"], message=msg)

        rectifier = None
        if USE_ROBUST_RECTIFIER and method == "geometry" and direction in {"forward", "backward"}:
            rectifier = ctx.rectifiers_by_side.get(side_strip)
            if rectifier is None or getattr(rectifier, "H", img.shape[0]) != img.shape[0]:
                rectifier = RobustHorizonRectifier(image_height=img.shape[0])
                ctx.rectifiers_by_side[side_strip] = rectifier

        rect_img, rect_mask, rect_meta = rectify_tile(
            img, mask_item["mask"], direction, method, target_width,
            selected_side=selected_side, rectifier=rectifier, edge_override=edge_override,
        )
        transform_note = ""
        if flip_180:
            rect_img = cv2.flip(rect_img, -1)
            rect_mask = cv2.flip(rect_mask.astype(np.uint8), -1).astype(bool)
            transform_note = "flip_xy"

        rect_img = normalize_canvas_width(rect_img, canvas_width)
        rect_mask = normalize_mask_width(rect_mask, canvas_width)
        assigned_obstacles = assign_obstacles_to_selected_sidewalk(obstacles, masks, mask_item)
        footprints = build_tile_footprints(
            assigned_obstacles, mask_item["mask"], direction, method, canvas_width, target_width,
            robust_warp=rect_meta.get("robust_warp") if isinstance(rect_meta, dict) else None,
            edge_override=edge_override, flip_180=flip_180
        )
        clean_img = rect_img.copy()
        clean_mask = rect_mask.copy()
        shape = rect_img.shape
        image_id = image_id_from_blob(row["blob_name"])
        meta = {
            "side_strip": side_strip,
            "point_id": point_id,
            "direction": direction,
            "selected_side": selected_side,
            "method": method,
            "image_id": image_id,
            "image_blob": row["blob_name"],
            "mask_blob": mask_item["blob_name"],
            "obstacle_count": len(assigned_obstacles),
            "footprint_count": len(footprints),
            "transform": transform_note or "none",
            "shape": shape,
        }
        if isinstance(rect_meta, dict):
            meta.update({k: v for k, v in rect_meta.items() if k != "robust_warp"})
        rect_img = add_image_badge(rect_img, image_id, direction, transform_note=transform_note)
        rectifier_note = meta.get("rectifier", method)
        vy_note = f" | vy={meta['final_vy']:.1f}" if isinstance(meta.get("final_vy"), (int, float)) else ""
        label = (
            f'image_id={image_id} | direction={direction} | transform={transform_note or "none"} | {label_prefix} | '
            f'rectifier={rectifier_note}{vy_note} | img={Path(row["blob_name"]).name} | '
            f'mask={Path(mask_item["blob_name"]).name} | shape={shape[1]}x{shape[0]}'
        )
        labeled = add_label_bar(rect_img, label, status="ok")
        return TileResult(labeled, "ok", side_strip, point_id, direction, selected_side, method,
                          image_blob=row["blob_name"], mask_blob=mask_item["blob_name"], shape=shape,
                          clean_image=clean_img, clean_mask=clean_mask, footprints=footprints, meta=meta)
    except Exception as exc:
        msg = f"{label_prefix} | ERROR: {type(exc).__name__}: {exc}"
        image_blob = row.get("blob_name", "") if row else ""
        return TileResult(warning_tile(canvas_width, msg), "error", side_strip, point_id, direction, selected_side, method,
                          image_blob=image_blob, message=msg)
