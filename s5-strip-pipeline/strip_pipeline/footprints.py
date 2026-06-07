"""Continuous footprint-box rendering. Ported from cell 21 of the notebook."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .merge_loftr import tile_label
from .tiles import connected_components, normalize_mask_width


def _class_color_map(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    names = sorted(set(class_names))
    try:
        from matplotlib import colormaps
        cmap = colormaps["tab20"].resampled(max(len(names), 1))
    except Exception:  # pragma: no cover - older matplotlib
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("tab20", max(len(names), 1))
    colors: dict[str, tuple[int, int, int]] = {}
    for idx, name in enumerate(names):
        rgb = cmap(idx)[:3]
        colors[name] = tuple(int(round(c * 255)) for c in rgb)
    return colors


def _segment_mask_slice(mask: np.ndarray, start: int, end: int, canvas_width: int) -> np.ndarray:
    mask = normalize_mask_width(mask, canvas_width)
    start = max(0, min(int(start), mask.shape[0]))
    end = max(0, min(int(end), mask.shape[0]))
    return mask[start:end, :].astype(bool)


def collect_footprint_boxes(segments: list[dict[str, Any]], output_shape: tuple[int, int]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    H, W = output_shape
    sidewalk = np.zeros((H, W), dtype=bool)
    boxes: list[dict[str, Any]] = []

    for segment in segments:
        tile = segment["tile"]
        y_out = int(segment["y_out"])
        start = int(segment["source_start"])
        end = int(segment["source_end"])
        seg_h = int(segment["image"].shape[0])
        if seg_h <= 0:
            continue

        sidewalk_slice = _segment_mask_slice(tile.clean_mask, start, end, W)
        h = min(seg_h, sidewalk_slice.shape[0], H - y_out)
        if h <= 0:
            continue
        sidewalk[y_out:y_out + h, :] |= sidewalk_slice[:h, :]

        for fp in tile.footprints or []:
            fp_slice = _segment_mask_slice(fp["mask"], start, end, W)
            h_fp = min(seg_h, fp_slice.shape[0], H - y_out)
            if h_fp <= 0 or not fp_slice[:h_fp, :].any():
                continue
            for component_id, component, bbox in connected_components(fp_slice[:h_fp, :]):
                min_row, min_col, max_row, max_col = bbox
                boxes.append({
                    "class_name": fp["class_name"],
                    "method": fp.get("method", ""),
                    "blob_name": fp.get("blob_name", ""),
                    "bbox": (int(y_out + min_row), int(min_col), int(y_out + max_row), int(max_col)),
                    "tile_index": segment["tile_index"],
                    "tile_label": tile_label(tile),
                })

    return sidewalk, boxes


def render_footprint_box_strip(clean_strip: np.ndarray, segments: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    H, W = clean_strip.shape[:2]
    sidewalk, boxes = collect_footprint_boxes(segments, (H, W))
    bg_color = np.array([38, 38, 38], dtype=np.uint8)
    sidewalk_color = np.array([190, 230, 255], dtype=np.uint8)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:] = bg_color
    sidewalk_rows = np.where(sidewalk.any(axis=1))[0]
    sidewalk_cols = np.where(sidewalk.any(axis=0))[0]
    if len(sidewalk_rows) > 0 and len(sidewalk_cols) > 0:
        out[sidewalk_rows[0]:sidewalk_rows[-1] + 1, sidewalk_cols[0]:sidewalk_cols[-1] + 1] = sidewalk_color

    colors = _class_color_map([box["class_name"] for box in boxes])
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        min_row, min_col, max_row, max_col = box["bbox"]
        if max_row <= min_row or max_col <= min_col:
            continue
        color = colors.get(box["class_name"], (255, 80, 80))
        overlay = out.copy()
        cv2.rectangle(overlay, (min_col, min_row), (max_col - 1, max_row - 1), color, -1)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        cv2.rectangle(out, (min_col, min_row), (max_col - 1, max_row - 1), color, 2)
        label_text = f"{box['class_name']} {max_col - min_col}x{max_row - min_row}px"
        text_y = min(max_row + 14, H - 4)
        cv2.putText(out, label_text, (min_col, text_y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return out, boxes


def make_footprint_debug_strip(rendered: np.ndarray, segments: list[dict[str, Any]], boxes: list[dict[str, Any]]) -> np.ndarray:
    debug = rendered.copy()
    H, W = debug.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    segment_colors = [(255, 80, 80), (80, 220, 120), (80, 160, 255), (255, 210, 80)]

    for idx, segment in enumerate(segments):
        y0 = int(segment["y_out"])
        y1 = min(H - 1, y0 + int(segment["image"].shape[0]) - 1)
        if y1 < 0 or y0 >= H:
            continue
        color = segment_colors[idx % len(segment_colors)]
        cv2.rectangle(debug, (0, max(0, y0)), (W - 1, y1), color, 2)
        tile = segment["tile"]
        label_text = f"{segment['tile_index']}: {tile_label(tile)} rows {segment['source_start']}:{segment['source_end']}"
        text_x = min(8, max(0, W - 2))
        text_y = min(H - 8, max(20, y0 + 20))
        cv2.putText(debug, label_text, (text_x, text_y), font, 0.5, color, 2, cv2.LINE_AA)

    for box in boxes:
        min_row, min_col, max_row, max_col = box["bbox"]
        label_text = f"{box['class_name']} | tile {box['tile_index']} | {box['tile_label']}"
        text_y = max(14, min_row - 5)
        if text_y < 18:
            text_y = min(max_row + 16, H - 4)
        text_x = min(max(2, min_col), max(0, W - 2))
        cv2.putText(debug, label_text, (text_x, text_y), font, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug, label_text, (text_x, text_y), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    return debug
