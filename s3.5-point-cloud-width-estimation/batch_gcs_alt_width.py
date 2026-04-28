#!/usr/bin/env python3
"""
Batch alternative sidewalk width estimation from GCS streetview images.

This runner:
  1) Lists images under an image prefix,
  2) Filters by point_id range and direction (typically left/right),
  3) Runs VGGT + SegFormer width estimation,
  4) Uploads per-image artifacts to GCS.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from google.cloud import storage
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from evaluate_sidewalk import (
    autocast_context,
    check_quality,
    load_vggt,
    maybe_empty_cache,
    measure_width_columnwise,
    resolve_device,
    save_geometry_plots,
    visualize_measurement,
)
from vggt.utils.load_fn import load_and_preprocess_images


_NEW_NAME_RE = re.compile(
    r"^(\d+)-(\d+)_(forward|backward|left|right)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)
_OLD_NAME_RE = re.compile(
    r"^(\d+)_(forward|backward|left|right)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)


@dataclass
class ParsedImage:
    point_id: str
    direction: str
    lat: str
    lon: str
    heading: str
    coordinate_folder: str


class GCSClient:
    def __init__(self, project_id: str, bucket_name: str):
        self._client = storage.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def list_blobs(self, prefix: str) -> list[str]:
        return [b.name for b in self._bucket.list_blobs(prefix=prefix)]

    def download_as_bytes(self, blob_name: str) -> bytes:
        return self._bucket.blob(blob_name).download_as_bytes()

    def upload_bytes(self, data: bytes, blob_name: str, content_type: str) -> str:
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{blob_name}"


def _parse_image_name(blob_name: str) -> ParsedImage | None:
    name = Path(blob_name).name
    m_new = _NEW_NAME_RE.match(name)
    if m_new:
        _, point_id, direction, lat, lon, heading = m_new.groups()
        return ParsedImage(
            point_id=point_id,
            direction=direction,
            lat=lat,
            lon=lon,
            heading=heading,
            coordinate_folder=f"{point_id}_{lat}_{lon}",
        )
    m_old = _OLD_NAME_RE.match(name)
    if m_old:
        point_id, direction, lat, lon, heading = m_old.groups()
        return ParsedImage(
            point_id=point_id,
            direction=direction,
            lat=lat,
            lon=lon,
            heading=heading,
            coordinate_folder=f"{point_id}_{lat}_{lon}",
        )
    return None


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("gcs", {})
    cfg.setdefault("batch", {})
    cfg.setdefault("model", {})
    cfg.setdefault("measurement", {})

    cfg["batch"].setdefault("point_id_min", 0)
    cfg["batch"].setdefault("point_id_max", 9999)
    cfg["batch"].setdefault("directions", ["left", "right"])

    cfg["model"].setdefault("vggt_weights", None)
    cfg["model"].setdefault("device", "auto")
    cfg["model"].setdefault("save_overlay", True)
    cfg["model"].setdefault("save_plot3d", True)
    cfg["model"].setdefault("verbose", False)

    cfg["measurement"].setdefault("cam_height", 2.5)
    cfg["measurement"].setdefault("band_frac", 0.20)
    cfg["measurement"].setdefault("min_run_px", 15)
    cfg["measurement"].setdefault("min_valid_cols", 10)
    cfg["measurement"].setdefault("min_sw_frac", 0.02)
    cfg["measurement"].setdefault("max_phys_width", 5.0)
    cfg["measurement"].setdefault("max_width_cv", 0.40)
    cfg["measurement"].setdefault("min_width_m", 0.50)
    cfg["measurement"].setdefault("max_width_m", 4.00)
    return cfg


def _to_png_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def run(cfg: dict[str, Any]) -> None:
    gcs_cfg = cfg["gcs"]
    batch_cfg = cfg["batch"]
    model_cfg = cfg["model"]
    msr_cfg = cfg["measurement"]

    gcs = GCSClient(
        project_id=gcs_cfg["project_id"],
        bucket_name=gcs_cfg["bucket_name"],
    )

    image_prefix = gcs_cfg["image_prefix"].rstrip("/") + "/"
    output_prefix = gcs_cfg["output_prefix"].rstrip("/")
    point_min = int(batch_cfg["point_id_min"])
    point_max = int(batch_cfg["point_id_max"])
    allowed_dirs = set(batch_cfg["directions"])

    print(f"[INFO] image_prefix={image_prefix}")
    print(f"[INFO] output_prefix={output_prefix}")
    print(f"[INFO] point_id range={point_min}-{point_max}, dirs={sorted(allowed_dirs)}")

    all_blobs = gcs.list_blobs(image_prefix)
    image_blobs = [b for b in all_blobs if b.lower().endswith((".jpg", ".jpeg", ".png"))]
    selected: list[tuple[int, str, ParsedImage]] = []
    for blob in image_blobs:
        parsed = _parse_image_name(blob)
        if parsed is None:
            continue
        if parsed.direction not in allowed_dirs:
            continue
        pid = int(parsed.point_id)
        if point_min <= pid <= point_max:
            selected.append((pid, blob, parsed))
    selected.sort(key=lambda t: (t[0], t[1]))

    if not selected:
        print("[WARN] No matching images found.")
        return
    print(f"[INFO] selected images: {len(selected)}")

    device = resolve_device(str(model_cfg["device"]))
    amp_ctx = autocast_context(device)
    use_non_blocking = device == "cuda"
    print(f"[INFO] Using device: {device}")

    vggt_weights = (model_cfg["vggt_weights"] or os.environ.get("VGGT_WEIGHTS") or "").strip()
    vggt_model = load_vggt(vggt_weights or None, device)
    vggt_model.eval()

    seg_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    processor = AutoImageProcessor.from_pretrained(seg_id)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        seg_id, revision="refs/pr/3", use_safetensors=True
    ).to(device)
    seg_model.eval()
    id2label = {int(k): v for k, v in seg_model.config.id2label.items()}
    label2id = {v.lower(): k for k, v in id2label.items()}
    id_sidewalk = label2id["sidewalk"]
    id_road = label2id["road"]

    t0 = time.time()
    ok_count = 0
    skip_reasons: Counter[str] = Counter()
    per_image: list[dict[str, Any]] = []

    for idx, (_, blob_name, parsed) in enumerate(selected, start=1):
        print(f"\n[{idx}/{len(selected)}] {blob_name}")
        out_folder = f"{output_prefix}/{parsed.coordinate_folder}/{parsed.direction}"
        meta: dict[str, Any] = {
            "source_image_blob": blob_name,
            "point_id": parsed.point_id,
            "direction": parsed.direction,
            "coordinate_folder": parsed.coordinate_folder,
            "lat": parsed.lat,
            "lon": parsed.lon,
            "heading": parsed.heading,
            "status": "skipped",
            "reason": "",
            "width_m": None,
            "n_cols": 0,
            "processed_utc": datetime.now(timezone.utc).isoformat(),
        }

        try:
            img_bytes = gcs.download_as_bytes(blob_name)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("imdecode_fail")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
                tf.write(img_bytes)
                tf.flush()
                img_tensor = load_and_preprocess_images([tf.name]).to(
                    device, non_blocking=use_non_blocking
                )
                with torch.inference_mode(), amp_ctx:
                    pred = vggt_model(img_tensor)

            wp = pred["world_points"]
            if wp.dim() == 5:
                wp = wp[0]
            wp = wp[0].detach().cpu().numpy()
            pose_enc = None
            if "pose_enc" in pred and pred["pose_enc"] is not None:
                pe = pred["pose_enc"]
                if pe.dim() == 3:
                    pe = pe[0]
                pose_enc = pe[0].detach().cpu().numpy()
            del pred
            maybe_empty_cache(device)
            if pose_enc is None:
                raise RuntimeError("no_pose_enc")

            with torch.inference_mode():
                inputs = processor(images=[img_rgb], return_tensors="pt").to(device)
                with amp_ctx:
                    outputs = seg_model(**inputs)
                seg_map = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[img_rgb.shape[:2]]
                )[0].cpu().numpy().astype(np.int32)
                del inputs, outputs
                maybe_empty_cache(device)

            ok, reason = check_quality(
                seg_map,
                id_sidewalk,
                id_road,
                min_sw_frac=float(msr_cfg["min_sw_frac"]),
                band_frac=float(msr_cfg["band_frac"]),
            )
            if not ok:
                skip_reasons[reason] += 1
                meta["reason"] = reason
            else:
                width, n_cols, info, fail_reason = measure_width_columnwise(
                    wp,
                    seg_map,
                    pose_enc,
                    float(msr_cfg["cam_height"]),
                    id_sidewalk,
                    id_road,
                    img_gray=img_gray,
                    band_frac=float(msr_cfg["band_frac"]),
                    min_run_px=int(msr_cfg["min_run_px"]),
                    min_valid_cols=int(msr_cfg["min_valid_cols"]),
                    width_range=(
                        float(msr_cfg["min_width_m"]),
                        float(msr_cfg["max_width_m"]),
                    ),
                    max_phys_width=float(msr_cfg["max_phys_width"]),
                    max_width_cv=float(msr_cfg["max_width_cv"]),
                    collect_geometry=bool(model_cfg["save_plot3d"]),
                )
                if width is None:
                    skip_reasons[fail_reason] += 1
                    meta["reason"] = fail_reason
                else:
                    meta["status"] = "ok"
                    meta["width_m"] = float(width)
                    meta["n_cols"] = int(n_cols)
                    meta["reason"] = ""
                    meta["measurement_info"] = {
                        k: v for k, v in info.items() if not k.startswith("geom_")
                    }
                    ok_count += 1

                    if bool(model_cfg["save_overlay"]):
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as ovf:
                            visualize_measurement(img_bgr, info, width, ovf.name)
                            gcs.upload_bytes(
                                _to_png_bytes(ovf.name),
                                f"{out_folder}/alt_width_overlay.png",
                                "image/png",
                            )

                    if bool(model_cfg["save_plot3d"]):
                        geom_keys = [k for k in list(info.keys()) if k.startswith("geom_")]
                        if geom_keys:
                            geom = {k: info[k] for k in geom_keys}
                            with tempfile.TemporaryDirectory() as td:
                                prefix = os.path.join(td, "geom")
                                save_geometry_plots(geom, prefix)
                                scene_path = f"{prefix}_scene3d.png"
                                width_path = f"{prefix}_width3d.png"
                                if os.path.isfile(scene_path):
                                    gcs.upload_bytes(
                                        _to_png_bytes(scene_path),
                                        f"{out_folder}/alt_width_scene3d.png",
                                        "image/png",
                                    )
                                if os.path.isfile(width_path):
                                    gcs.upload_bytes(
                                        _to_png_bytes(width_path),
                                        f"{out_folder}/alt_width_width3d.png",
                                        "image/png",
                                    )

        except Exception as exc:
            reason = f"E:exception({exc})"
            skip_reasons[reason] += 1
            meta["reason"] = reason

        gcs.upload_bytes(
            json.dumps(meta, indent=2).encode("utf-8"),
            f"{out_folder}/alt_width_metadata.json",
            "application/json",
        )
        per_image.append(meta)
        if bool(model_cfg["verbose"]):
            print(f"  status={meta['status']} reason={meta['reason']} width={meta['width_m']}")

    elapsed = time.time() - t0
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_prefix": image_prefix,
        "output_prefix": output_prefix,
        "point_id_range": [point_min, point_max],
        "directions": sorted(allowed_dirs),
        "processed": len(per_image),
        "ok": ok_count,
        "failed_or_skipped": len(per_image) - ok_count,
        "elapsed_s": round(elapsed, 1),
        "skip_reason_counts": dict(skip_reasons),
        "per_image": per_image,
    }
    summary_blob = f"{output_prefix}/_alt_width_batch_summary.json"
    gcs.upload_bytes(
        json.dumps(summary, indent=2).encode("utf-8"),
        summary_blob,
        "application/json",
    )
    print("\n[INFO] Alt-width batch complete")
    print(f"[INFO] processed={len(per_image)} ok={ok_count} elapsed={elapsed:.1f}s")
    print(f"[INFO] summary=gs://{gcs.bucket_name}/{summary_blob}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch alternative sidewalk width estimation from GCS."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "batch_gcs_alt_width.yaml"),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
