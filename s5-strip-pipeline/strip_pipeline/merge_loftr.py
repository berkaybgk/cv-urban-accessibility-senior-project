"""LoFTR-based seam merging and match diagnostics. Ported from cells 16, 17, 19.

Globals replaced: ``CANVAS_WIDTH`` is threaded as a parameter, the LoFTR matcher
is a lazily-loaded singleton, and per-pair stdout prints are dropped (logs are
returned instead).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import (
    MIN_CONFIDENCE,
    MIN_RANSAC_INLIERS,
    LOFTR_LONG_SIDE,
    LOFTR_WEIGHTS,
    RESTRICT_TO_SIDEWALK,
    ROAD_ROAD_PAIR_KEEP_RATIO,
    ROAD_SIDE_PAIR_KEEP_RATIO,
    ROAD_VIEW_MATCH_KEEP_RATIO,
    ROBUST_PIXELS_PER_METER,
    SIDE_VIEW_PAIR_KEEP_RATIO,
)
from .tiles import TileResult, normalize_canvas_width, warning_tile

ROAD_VIEW_MATCH_KEEP_PERCENT = int(round(ROAD_VIEW_MATCH_KEEP_RATIO * 100))
SIDE_VIEW_PAIR_KEEP_PERCENT = int(round(SIDE_VIEW_PAIR_KEEP_RATIO * 100))
ROAD_SIDE_PAIR_KEEP_PERCENT = int(round(ROAD_SIDE_PAIR_KEEP_RATIO * 100))
ROAD_ROAD_PAIR_KEEP_PERCENT = int(round(ROAD_ROAD_PAIR_KEEP_RATIO * 100))
FALLBACK_GAP_PX = ROBUST_PIXELS_PER_METER / 2
MIN_SEAM_SLICE_HEIGHT_PX = -40

_matcher = None
_device = None


def get_matcher():
    """Lazily load LoFTR + pick the best available torch device."""
    global _matcher, _device
    if _matcher is None:
        import torch
        import kornia.feature as KF

        _device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _matcher = KF.LoFTR(pretrained=LOFTR_WEIGHTS).eval().to(_device)
    return _matcher, _device


def to_loftr_tensor(img_rgb: np.ndarray, long_side: int):
    import torch

    h, w = img_rgb.shape[:2]
    s = long_side / max(h, w)
    new_w, new_h = int(round(w * s)), int(round(h * s))
    new_w -= new_w % 8
    new_h -= new_h % 8
    new_w = max(8, new_w)
    new_h = max(8, new_h)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    tensor = torch.from_numpy(gray).float()[None, None] / 255.0
    return tensor, w / new_w, h / new_h


def points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    xs = np.clip(points[:, 0].round().astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(points[:, 1].round().astype(int), 0, mask.shape[0] - 1)
    return mask[ys, xs]


def tile_label(tile: TileResult) -> str:
    meta = tile.meta or {}
    return f"{meta.get('point_id', tile.point_id)} {meta.get('direction', tile.direction)} {meta.get('selected_side', tile.selected_side)}"


def road_view_crop_bounds(tile: TileResult) -> tuple[int, int, str]:
    h = int(tile.clean_image.shape[0])
    if tile.direction == "forward":
        y0 = int(h * (1.0 - ROAD_VIEW_MATCH_KEEP_RATIO))
        return y0, h, f"forward-bottom{ROAD_VIEW_MATCH_KEEP_PERCENT}"
    if tile.direction == "backward":
        y1 = int(h * ROAD_VIEW_MATCH_KEEP_RATIO)
        return 0, y1, f"backward-top{ROAD_VIEW_MATCH_KEEP_PERCENT}"
    return 0, h, "full"


def _is_side_view(tile: TileResult) -> bool:
    return tile.direction in {"left", "right"}


def _crop_bounds_portion(start: int, end: int, portion: str, ratio: float) -> tuple[int, int]:
    span = max(0, int(end) - int(start))
    keep = int(span * ratio)
    if portion == "top":
        return int(start), int(start) + keep
    if portion == "bottom":
        return int(end) - keep, int(end)
    return int(start), int(end)


def _crop_tile_window(tile: TileResult, portion: str, ratio: float | None, label: str):
    img = tile.clean_image
    mask = tile.clean_mask
    base_start, base_end, base_mode = road_view_crop_bounds(tile)
    if ratio is None:
        y0, y1 = base_start, base_end
        mode = base_mode
    else:
        y0, y1 = _crop_bounds_portion(base_start, base_end, portion, ratio)
        mode = label
    return img[y0:y1, :], mask[y0:y1, :], np.array([0.0, float(y0)]), mode


def crop_tile_for_matching(tile: TileResult):
    return _crop_tile_window(tile, "full", None, "")


def crop_pair_for_matching(a: TileResult, b: TileResult):
    if _is_side_view(a) and b.direction == "forward":
        img_a, mask_a, off_a, mode_a = _crop_tile_window(a, "top", SIDE_VIEW_PAIR_KEEP_RATIO, f"{a.direction}-top{SIDE_VIEW_PAIR_KEEP_PERCENT}")
        img_b, mask_b, off_b, mode_b = _crop_tile_window(b, "bottom", ROAD_SIDE_PAIR_KEEP_RATIO, f"forward-base{ROAD_VIEW_MATCH_KEEP_PERCENT}-bottom{ROAD_SIDE_PAIR_KEEP_PERCENT}")
    elif a.direction == "backward" and _is_side_view(b):
        img_a, mask_a, off_a, mode_a = _crop_tile_window(a, "top", ROAD_SIDE_PAIR_KEEP_RATIO, f"backward-base{ROAD_VIEW_MATCH_KEEP_PERCENT}-top{ROAD_SIDE_PAIR_KEEP_PERCENT}")
        img_b, mask_b, off_b, mode_b = _crop_tile_window(b, "bottom", SIDE_VIEW_PAIR_KEEP_RATIO, f"{b.direction}-bottom{SIDE_VIEW_PAIR_KEEP_PERCENT}")
    elif a.direction == "forward" and b.direction == "backward":
        img_a, mask_a, off_a, mode_a = _crop_tile_window(a, "top", ROAD_ROAD_PAIR_KEEP_RATIO, f"forward-base{ROAD_VIEW_MATCH_KEEP_PERCENT}-top{ROAD_ROAD_PAIR_KEEP_PERCENT}")
        img_b, mask_b, off_b, mode_b = _crop_tile_window(b, "bottom", ROAD_ROAD_PAIR_KEEP_RATIO, f"backward-base{ROAD_VIEW_MATCH_KEEP_PERCENT}-bottom{ROAD_ROAD_PAIR_KEEP_PERCENT}")
    else:
        img_a, mask_a, off_a, mode_a = crop_tile_for_matching(a)
        img_b, mask_b, off_b, mode_b = crop_tile_for_matching(b)
    mode = f"{mode_a}__{mode_b}"
    return img_a, mask_a, off_a, img_b, mask_b, off_b, mode


def match_adjacent_tiles(a: TileResult, b: TileResult) -> dict[str, Any]:
    import torch

    base = {
        "a": tile_label(a),
        "b": tile_label(b),
        "status": "fallback",
        "mode": "unmatched",
        "confidence": 0.0,
        "inliers": 0,
        "offset_x": 0.0,
        "offset_y": -float(b.clean_image.shape[0] + FALLBACK_GAP_PX) if b.clean_image is not None else 0.0,
        "pA": None,
        "pB": None,
        "all_pA": [],
        "all_pB": [],
        "all_conf": [],
        "message": "",
    }

    if a.clean_image is None or b.clean_image is None or a.clean_mask is None or b.clean_mask is None:
        base["message"] = "missing clean tile data"
        return base

    crop_a, mask_a, off_a, crop_b, mask_b, off_b, mode = crop_pair_for_matching(a, b)
    base["mode"] = mode
    if crop_a.shape[0] < 8 or crop_b.shape[0] < 8 or crop_a.shape[1] < 8 or crop_b.shape[1] < 8:
        base["message"] = "crop too small"
        return base

    try:
        matcher, device = get_matcher()
        t_a, sx_a, sy_a = to_loftr_tensor(crop_a, LOFTR_LONG_SIDE)
        t_b, sx_b, sy_b = to_loftr_tensor(crop_b, LOFTR_LONG_SIDE)
        with torch.inference_mode():
            out = matcher({"image0": t_a.to(device), "image1": t_b.to(device)})
        k_a = out["keypoints0"].cpu().numpy() * np.array([sx_a, sy_a])
        k_b = out["keypoints1"].cpu().numpy() * np.array([sx_b, sy_b])
        conf = out["confidence"].cpu().numpy()
    except Exception as exc:
        base["message"] = f"LoFTR error: {type(exc).__name__}: {exc}"
        return base

    keep = conf >= MIN_CONFIDENCE
    if RESTRICT_TO_SIDEWALK:
        keep &= points_in_mask(k_a, mask_a) & points_in_mask(k_b, mask_b)

    p_a = k_a[keep]
    p_b = k_b[keep]
    p_c = conf[keep]

    if len(p_a) >= 4:
        try:
            _, inliers = cv2.findHomography(p_a, p_b, cv2.RANSAC, 3.0)
            if inliers is not None:
                inliers = inliers.ravel().astype(bool)
                p_a, p_b, p_c = p_a[inliers], p_b[inliers], p_c[inliers]
        except cv2.error:
            p_a, p_b, p_c = np.array([]), np.array([]), np.array([])

    if len(p_a) < 1:
        base["message"] = "no valid inlier matches"
        return base

    # Convert crops to full image coordinates
    all_full_a = p_a + off_a
    all_full_b = p_b + off_b

    # Compute median consensus translation vector
    deltas = all_full_a - all_full_b  # shape (N, 2)
    median_delta = np.median(deltas, axis=0)

    # Select the keypoint pair closest to the median consensus delta
    distances = np.linalg.norm(deltas - median_delta, axis=1)
    best_idx = int(np.argmin(distances))

    full_a = all_full_a[best_idx]
    full_b = all_full_b[best_idx]
    delta = full_a - full_b

    base.update({
        "status": "matched",
        "confidence": float(p_c[best_idx]),
        "inliers": int(len(p_a)),
        "offset_x": float(delta[0]),
        "offset_y": float(delta[1]),
        "pA": full_a.astype(float),
        "pB": full_b.astype(float),
        "all_pA": all_full_a.astype(float).tolist(),
        "all_pB": all_full_b.astype(float).tolist(),
        "all_conf": p_c.astype(float).tolist(),
        "message": "ok",
    })
    return base


def _row_from_point(point: np.ndarray | None, height: int) -> int | None:
    if point is None:
        return None
    y = int(round(float(point[1])))
    if y < 0 or y > height:
        return None
    return y


def _fallback_pair_offset(next_tile: TileResult) -> float:
    start, end, _ = road_view_crop_bounds(next_tile)
    return -float((end - start) + FALLBACK_GAP_PX)


def build_seam_cuts(tiles: list[TileResult], logs: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    visible_bounds = [road_view_crop_bounds(t) for t in tiles]
    starts = [start for start, end, mode in visible_bounds]
    ends = [end for start, end, mode in visible_bounds]

    def apply_fallback(idx, log, below, above):
        if (below.direction == "forward" and above.direction == "backward") or (below.direction == "backward" and above.direction == "forward"):
            below_visible_start, below_visible_end, _ = visible_bounds[idx]
            above_visible_start, above_visible_end, _ = visible_bounds[idx + 1]

            portion_below = "top" if below.direction == "forward" else "bottom"
            portion_above = "bottom" if above.direction == "forward" else "top"

            y0_below, y1_below = _crop_bounds_portion(below_visible_start, below_visible_end, portion_below, ROAD_ROAD_PAIR_KEEP_RATIO)
            y0_above, y1_above = _crop_bounds_portion(above_visible_start, above_visible_end, portion_above, ROAD_ROAD_PAIR_KEEP_RATIO)

            starts[idx] = y0_below
            ends[idx] = y1_below
            starts[idx + 1] = y0_above
            ends[idx + 1] = y1_above

            log["offset_y"] = -float((y1_above - y0_above) + FALLBACK_GAP_PX)
        else:
            log["offset_y"] = _fallback_pair_offset(above)

    for idx, log in enumerate(logs):
        below = tiles[idx]
        above = tiles[idx + 1]
        below_h = int(below.clean_image.shape[0])
        above_h = int(above.clean_image.shape[0])
        log["seam_applied"] = False
        log["seam_y_below"] = ""
        log["seam_y_above"] = ""

        if log["status"] != "matched":
            apply_fallback(idx, log, below, above)
            continue

        y_below = _row_from_point(log.get("pA"), below_h)
        y_above = _row_from_point(log.get("pB"), above_h)
        log["seam_y_below"] = y_below if y_below is not None else ""
        log["seam_y_above"] = y_above if y_above is not None else ""

        reasons = []
        if y_below is None:
            reasons.append("y_below is None")
        if y_above is None:
            reasons.append("y_above is None")

        if y_below is not None and y_above is not None:
            below_visible_start, below_visible_end, _ = visible_bounds[idx]
            above_visible_start, above_visible_end, _ = visible_bounds[idx + 1]

            if not (below_visible_start <= y_below <= below_visible_end):
                reasons.append(f"y_below={y_below} not in visible bounds [{below_visible_start}, {below_visible_end}]")
            if not (above_visible_start <= y_above <= above_visible_end):
                reasons.append(f"y_above={y_above} not in visible bounds [{above_visible_start}, {above_visible_end}]")

            if not reasons:
                below_slice_h = ends[idx] - y_below
                above_slice_h = y_above - starts[idx + 1]
                if below_slice_h < MIN_SEAM_SLICE_HEIGHT_PX:
                    reasons.append(f"below_slice_h={below_slice_h} < MIN={MIN_SEAM_SLICE_HEIGHT_PX}")
                if above_slice_h < MIN_SEAM_SLICE_HEIGHT_PX:
                    reasons.append(f"above_slice_h={above_slice_h} < MIN={MIN_SEAM_SLICE_HEIGHT_PX}")

        valid = len(reasons) == 0
        if not valid:
            log["status"] = "invalid-seam-fallback"
            log["offset_y"] = _fallback_pair_offset(above)
            log["message"] = f"invalid seam: {'; '.join(reasons)}"
            continue

        starts[idx] = y_below
        ends[idx + 1] = y_above
        log["seam_applied"] = True
        log["offset_y"] = float(y_below - y_above)

    return starts, ends


def build_strip_segments(tiles: list[TileResult], starts: list[int], ends: list[int], canvas_width: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    y_out = 0
    for tile_index in reversed(range(len(tiles))):
        tile = tiles[tile_index]
        img = normalize_canvas_width(tile.clean_image, canvas_width)
        start = max(0, min(int(starts[tile_index]), img.shape[0]))
        end = max(0, min(int(ends[tile_index]), img.shape[0]))
        if end <= start:
            continue
        segment = img[start:end, :].copy()
        segments.append({
            "tile_index": tile_index,
            "tile": tile,
            "image": segment,
            "source_start": start,
            "source_end": end,
            "y_out": y_out,
        })
        y_out += segment.shape[0]
    return segments


def compose_seam_strip(segments: list[dict[str, Any]], canvas_width: int) -> np.ndarray:
    if not segments:
        return warning_tile(canvas_width, "No valid seam segments")
    return np.vstack([segment["image"] for segment in segments])


def make_seam_debug_image(clean: np.ndarray, segments: list[dict[str, Any]], logs: list[dict[str, Any]]) -> np.ndarray:
    debug = clean.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = [(255, 80, 80), (80, 220, 120), (80, 160, 255), (255, 210, 80)]
    by_tile_index = {segment["tile_index"]: segment for segment in segments}

    for idx, segment in enumerate(segments):
        y0 = int(segment["y_out"])
        y1 = y0 + int(segment["image"].shape[0]) - 1
        color = colors[idx % len(colors)]
        cv2.rectangle(debug, (0, y0), (debug.shape[1] - 1, y1), color, 2)
        label = f"{segment['tile_index']}: {tile_label(segment['tile'])} rows {segment['source_start']}:{segment['source_end']}"
        cv2.putText(debug, label, (8, max(22, y0 + 22)), font, 0.55, color, 2, cv2.LINE_AA)

    for log in logs:
        pair_idx = log.get("pair_index")
        if pair_idx is None:
            continue
        below_segment = by_tile_index.get(pair_idx)
        above_segment = by_tile_index.get(pair_idx + 1)
        if below_segment is None or above_segment is None:
            continue
        boundary_y = int(below_segment["y_out"])
        if 0 <= boundary_y < debug.shape[0]:
            color = (0, 255, 255) if log.get("seam_applied") else (255, 150, 150)
            cv2.line(debug, (0, boundary_y), (debug.shape[1] - 1, boundary_y), color, 2)
            text = f"pair {pair_idx} {log['status']} conf={log['confidence']:.2f} inl={log['inliers']} dx ignored={log['offset_x']:.0f}"
            cv2.putText(debug, text, (8, min(debug.shape[0] - 8, boundary_y + 18)), font, 0.5, color, 1, cv2.LINE_AA)

    return debug


def merge_side_strip(side: str, tiles_bottom_to_top: list[TileResult], canvas_width: int, return_segments: bool = False):
    tiles = [t for t in tiles_bottom_to_top if t.status == "ok" and t.clean_image is not None]
    if not tiles:
        clean = warning_tile(canvas_width, f"No clean tiles for {side}")
        debug = warning_tile(canvas_width, f"No clean tiles for {side}")
        return (clean, debug, [], []) if return_segments else (clean, debug, [])

    logs: list[dict[str, Any]] = []
    for idx in range(len(tiles) - 1):
        match = match_adjacent_tiles(tiles[idx], tiles[idx + 1])
        match["pair_index"] = idx
        logs.append(match)

    starts, ends = build_seam_cuts(tiles, logs)
    segments = build_strip_segments(tiles, starts, ends, canvas_width)
    clean = compose_seam_strip(segments, canvas_width)
    debug = make_seam_debug_image(clean, segments, logs)

    return (clean, debug, logs, segments) if return_segments else (clean, debug, logs)


# --- Full-tile LoFTR match diagnostic strip (cell 19) ---

def _as_int_point(point: Any, tile_shape) -> tuple[int, int] | None:
    if point is None:
        return None
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    h, w = tile_shape[:2]
    x = int(round(float(arr[0])))
    y = int(round(float(arr[1])))
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    return x, y


def _draw_match_marker(img: np.ndarray, point: tuple[int, int], label: str, color: tuple[int, int, int]) -> None:
    x, y = point
    cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 11, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, label, (min(img.shape[1] - 80, x + 12), max(18, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


def _point_on_normalized_canvas(point: tuple[int, int], source_shape, canvas_width: int) -> tuple[int, int] | None:
    x, y = point
    h, w = source_shape[:2]
    if w == canvas_width:
        return x, y
    if w > canvas_width:
        x0 = (w - canvas_width) // 2
        x = x - x0
        if x < 0 or x >= canvas_width:
            return None
        return x, y
    pad_left = (canvas_width - w) // 2
    return x + pad_left, y


def make_full_tile_loftr_debug_strip(side: str, tiles_bottom_to_top: list[TileResult],
                                     logs: list[dict[str, Any]], canvas_width: int) -> np.ndarray:
    tiles = [t for t in tiles_bottom_to_top if t.status == "ok" and t.clean_image is not None]
    if not tiles:
        return warning_tile(canvas_width, f"No clean tiles for {side}")

    rendered: list[dict[str, Any]] = []
    y_out = 0
    for tile_index in reversed(range(len(tiles))):
        tile = tiles[tile_index]
        img = normalize_canvas_width(tile.clean_image, canvas_width).copy()
        label = f"{tile_index}: {tile_label(tile)} | full rectified"
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        rendered.append({"tile_index": tile_index, "tile": tile, "image": img, "y_out": y_out})
        y_out += img.shape[0]

    strip = np.vstack([item["image"] for item in rendered])
    by_tile_index = {item["tile_index"]: item for item in rendered}
    red = (255, 0, 0)
    muted = (180, 180, 180)

    for log in logs:
        pair_idx = log.get("pair_index")
        if pair_idx is None:
            continue
        below_item = by_tile_index.get(pair_idx)
        above_item = by_tile_index.get(pair_idx + 1)
        if below_item is None or above_item is None:
            continue

        boundary_y = below_item["y_out"]
        cv2.line(strip, (0, boundary_y), (strip.shape[1] - 1, boundary_y), muted, 1, cv2.LINE_AA)

        if log.get("status") != "matched":
            text = f"pair {pair_idx}: {log.get('status')} | {log.get('message', '')}"
            cv2.putText(strip, text, (8, min(strip.shape[0] - 8, boundary_y + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1, cv2.LINE_AA)
            continue

        below_tile = below_item["tile"]
        above_tile = above_item["tile"]

        all_pA = log.get("all_pA", [])
        all_pB = log.get("all_pB", [])
        best_pA = log.get("pA")
        best_pB = log.get("pB")

        cyan = (0, 255, 255)  # Cyan in RGB

        # Draw all candidate (inlier) matches first in cyan
        for i in range(len(all_pA)):
            pA_i = np.array(all_pA[i])
            pB_i = np.array(all_pB[i])

            # Skip drawing candidate if it is the best match
            if best_pA is not None and best_pB is not None:
                if np.allclose(pA_i, best_pA) and np.allclose(pB_i, best_pB):
                    continue

            p_below_raw_i = _as_int_point(pA_i, below_tile.clean_image.shape)
            p_above_raw_i = _as_int_point(pB_i, above_tile.clean_image.shape)
            if p_below_raw_i is None or p_above_raw_i is None:
                continue
            p_below_i = _point_on_normalized_canvas(p_below_raw_i, below_tile.clean_image.shape, canvas_width)
            p_above_i = _point_on_normalized_canvas(p_above_raw_i, above_tile.clean_image.shape, canvas_width)
            if p_below_i is None or p_above_i is None:
                continue

            x1_i, y1_i = p_above_i[0], above_item["y_out"] + p_above_i[1]
            x2_i, y2_i = p_below_i[0], below_item["y_out"] + p_below_i[1]
            cv2.line(strip, (x1_i, y1_i), (x2_i, y2_i), cyan, 1, cv2.LINE_AA)
            cv2.circle(strip, (x1_i, y1_i), 3, cyan, -1, cv2.LINE_AA)
            cv2.circle(strip, (x2_i, y2_i), 3, cyan, -1, cv2.LINE_AA)

        # Draw the chosen best match in bold red
        p_below_raw = _as_int_point(best_pA, below_tile.clean_image.shape)
        p_above_raw = _as_int_point(best_pB, above_tile.clean_image.shape)
        if p_below_raw is None or p_above_raw is None:
            continue
        p_below = _point_on_normalized_canvas(p_below_raw, below_tile.clean_image.shape, canvas_width)
        p_above = _point_on_normalized_canvas(p_above_raw, above_tile.clean_image.shape, canvas_width)
        if p_below is None or p_above is None:
            continue

        x1, y1 = p_above[0], above_item["y_out"] + p_above[1]
        x2, y2 = p_below[0], below_item["y_out"] + p_below[1]
        cv2.line(strip, (x1, y1), (x2, y2), red, 3, cv2.LINE_AA)
        _draw_match_marker(strip, (x1, y1), f"{pair_idx}B", red)
        _draw_match_marker(strip, (x2, y2), f"{pair_idx}A", red)

        text = f"pair {pair_idx}: conf={log.get('confidence', 0.0):.2f} inl={log.get('inliers', 0)} dx={log.get('offset_x', 0.0):.0f} dy={log.get('offset_y', 0.0):.0f}"
        mid_y = int((y1 + y2) / 2)
        cv2.putText(strip, text, (8, int(np.clip(mid_y, 18, strip.shape[0] - 8))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2, cv2.LINE_AA)

    return strip
