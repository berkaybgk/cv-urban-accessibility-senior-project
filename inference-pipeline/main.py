"""
SAM3 Inference Pipeline
───────────────────────
Loads SAM3 on a CUDA GPU, optionally fetches images from GCS, runs
text-prompted segmentation, and saves annotated results.

Usage:
    # Run all preflight checks without loading the model
    python main.py --preflight

    # Segment a local image
    python main.py --image /path/to/image.jpg --prompt "sidewalk"

    # Segment all images under a GCS prefix and save results locally
    python main.py --gcs-prefix streetview/istanbul/ --prompt "sidewalk"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Resolve project paths relative to *this file*, not cwd ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # inference-pipeline/
PROJECT_ROOT = SCRIPT_DIR.parent                       # cv-urban-accessibility-senior-project/
SAM3_REPO_DIR = PROJECT_ROOT / "sam3-trials" / "local-gpu-setup" / "sam3"
BPE_PATH = SAM3_REPO_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
API_TRIALS_DIR = PROJECT_ROOT / "api-trials"
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = SCRIPT_DIR / "output"


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 0 — Helpers
# ═════════════════════════════════════════════════════════════════════════════

class _PhaseError(RuntimeError):
    """Raised when a preflight phase fails."""


def _banner(msg: str) -> None:
    width = max(len(msg) + 4, 50)
    print("\n" + "─" * width)
    print(f"  {msg}")
    print("─" * width)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Environment Variables
# ═════════════════════════════════════════════════════════════════════════════

def check_environment() -> None:
    _banner("Phase 1: Environment Variables")

    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
    _info(f".env path: {ENV_PATH}  (exists={ENV_PATH.exists()})")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        _fail("HF_TOKEN not set.  Add it to the .env at the project root.")
        raise _PhaseError("Missing HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    _ok(f"HF_TOKEN loaded ({hf_token[:8]}...)")

    for var in ("GCS_BUCKET_NAME", "GCP_PROJECT_ID"):
        val = os.getenv(var, "")
        if val:
            _ok(f"{var} = {val}")
        else:
            _info(f"{var} not set (only needed for GCS operations)")


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 2 — GPU / CUDA
# ═════════════════════════════════════════════════════════════════════════════

def check_gpu():
    """Returns the torch module after verifying CUDA availability."""
    _banner("Phase 2: GPU & CUDA")

    try:
        import torch
        import torchvision
    except ImportError as exc:
        _fail(f"Cannot import torch/torchvision: {exc}")
        _info("Install with:  pip install torch torchvision")
        raise _PhaseError("torch not installed") from exc

    _ok(f"PyTorch  {torch.__version__}")
    _ok(f"Torchvision  {torchvision.__version__}")

    if not torch.cuda.is_available():
        _fail("CUDA is NOT available.")
        _info("Make sure you are on the GPU VM and using the right venv.")
        raise _PhaseError("No CUDA")

    _ok(f"CUDA {torch.version.cuda}")
    _ok(f"GPU: {torch.cuda.get_device_name(0)}")

    cap = torch.cuda.get_device_properties(0)
    _ok(f"Compute capability: {cap.major}.{cap.minor}")

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if cap.major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _ok("TF32 enabled (Ampere+)")
    else:
        _info("TF32 not available (compute capability < 8)")

    return torch


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 3 — SAM3 Repository & Package
# ═════════════════════════════════════════════════════════════════════════════

def check_sam3():
    _banner("Phase 3: SAM3 Repository & Package")

    _info(f"Expected SAM3 repo at: {SAM3_REPO_DIR}")
    if not SAM3_REPO_DIR.exists():
        _fail("SAM3 repo directory not found.")
        _info("Clone it first from the notebook, or run:")
        _info(f"  cd {SAM3_REPO_DIR.parent}")
        _info("  git clone https://github.com/facebookresearch/sam3.git")
        _info("  cd sam3 && pip install -e '.[notebooks]'")
        raise _PhaseError("SAM3 repo missing")
    _ok("SAM3 repo directory exists")

    if not BPE_PATH.exists():
        _fail(f"BPE vocab file not found at: {BPE_PATH}")
        raise _PhaseError("BPE file missing")
    _ok(f"BPE vocab found: {BPE_PATH.name}")

    try:
        from sam3.model_builder import build_sam3_image_model  # noqa: F401
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
    except ImportError as exc:
        _fail(f"Cannot import sam3 package: {exc}")
        _info("Did you install it?  cd sam3-trials/local-gpu-setup/sam3 && pip install -e '.[notebooks]'")
        raise _PhaseError("sam3 not importable") from exc

    _ok("sam3 package importable")


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Load Model
# ═════════════════════════════════════════════════════════════════════════════

def load_model(confidence_threshold: float = 0.3):
    """Build and return (model, processor).  Call only after phases 1-3 pass."""
    _banner("Phase 4: Loading SAM3 Model")
    _info("This may download weights on first run (~3.5 GB) …")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    t0 = time.time()
    model = build_sam3_image_model(bpe_path=str(BPE_PATH))
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    _ok(f"Model loaded in {time.time() - t0:.1f}s")

    return model, processor


# ═════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Supervision (visualization)
# ═════════════════════════════════════════════════════════════════════════════

def _get_supervision():
    """Import supervision lazily and return (sv module, COLOR palette)."""
    try:
        import supervision as sv
    except ImportError as exc:
        _fail(f"Cannot import supervision: {exc}")
        _info("Install with:  pip install supervision")
        raise _PhaseError("supervision not installed") from exc

    COLOR = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
        "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
    ])
    return sv, COLOR


def from_sam(sam_result: dict):
    """Convert SAM3 output dict → sv.Detections."""
    import torch
    sv, _ = _get_supervision()

    xyxy = sam_result["boxes"].to(torch.float32).cpu().numpy()
    confidence = sam_result["scores"].to(torch.float32).cpu().numpy()

    mask = sam_result["masks"].to(torch.bool)
    mask = mask.reshape(mask.shape[0], mask.shape[2], mask.shape[3]).cpu().numpy()

    return sv.Detections(xyxy=xyxy, confidence=confidence, mask=mask)


def annotate(image, detections, label: Optional[str] = None):
    """Draw masks, boxes, and optional labels on a PIL image."""
    sv, COLOR = _get_supervision()

    mask_annotator = sv.MaskAnnotator(
        color=COLOR, color_lookup=sv.ColorLookup.INDEX, opacity=0.6,
    )
    box_annotator = sv.BoxAnnotator(
        color=COLOR, color_lookup=sv.ColorLookup.INDEX, thickness=1,
    )
    label_annotator = sv.LabelAnnotator(
        color=COLOR, color_lookup=sv.ColorLookup.INDEX,
        text_scale=0.4, text_padding=5,
        text_color=sv.Color.BLACK, text_thickness=1,
    )

    annotated = image.copy()
    annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)

    if label:
        labels = [f"{label} {c:.2f}" for c in detections.confidence]
        annotated = label_annotator.annotate(annotated, detections, labels)

    return annotated


# ═════════════════════════════════════════════════════════════════════════════
#  Metadata extraction
# ═════════════════════════════════════════════════════════════════════════════

def build_metadata(
    detections,
    image_width: int,
    image_height: int,
    prompt: str,
    confidence_min: float,
    source_name: str,
    inference_time_s: float | None = None,
) -> dict:
    """Build a structured metadata dict from detection results.

    Includes per-detection stats (confidence, bbox, mask area) and
    image-level aggregates (total segmented ratio, mean confidence, …).
    """
    import numpy as np

    total_pixels = image_width * image_height
    per_detection: list[dict] = []

    # Union mask to compute total *unique* segmented area
    union_mask = np.zeros((image_height, image_width), dtype=bool)

    for i in range(len(detections)):
        conf = float(detections.confidence[i])
        x1, y1, x2, y2 = detections.xyxy[i].tolist()

        mask_pixels = 0
        mask_ratio = 0.0
        if detections.mask is not None and i < len(detections.mask):
            m = detections.mask[i]
            mask_pixels = int(m.sum())
            mask_ratio = mask_pixels / total_pixels
            union_mask |= m

        per_detection.append({
            "index": i,
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)],
            "bbox_width": round(x2 - x1, 1),
            "bbox_height": round(y2 - y1, 1),
            "mask_pixels": mask_pixels,
            "mask_area_ratio": round(mask_ratio, 6),
        })

    total_segmented_pixels = int(union_mask.sum())
    confidences = [d["confidence"] for d in per_detection]

    meta = {
        "source": source_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_width": image_width,
        "image_height": image_height,
        "total_pixels": total_pixels,
        "prompt": prompt,
        "confidence_threshold": confidence_min,
        "detection_count": len(per_detection),
        "detections": per_detection,
        "summary": {
            "total_segmented_pixels": total_segmented_pixels,
            "total_segmented_ratio": round(total_segmented_pixels / total_pixels, 6),
            "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            "min_confidence": round(min(confidences), 4) if confidences else None,
            "max_confidence": round(max(confidences), 4) if confidences else None,
        },
    }
    if inference_time_s is not None:
        meta["inference_time_s"] = round(inference_time_s, 3)

    return meta


def _save_metadata(meta: dict, out_path: Path) -> None:
    """Write metadata dict as a JSON sidecar file."""
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    _ok(f"Metadata → {json_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Inference helpers
# ═════════════════════════════════════════════════════════════════════════════

def segment_image(processor, image, prompt: str, confidence_min: float = 0.5,
                  source_name: str = "unknown"):
    """Run SAM3 text-prompted segmentation on a single PIL image.

    Returns (detections, annotated_image, metadata_dict).
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    t0 = time.time()
    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
    inference_time = time.time() - t0

    detections = from_sam(inference_state)
    detections = detections[detections.confidence > confidence_min]

    annotated = annotate(image, detections, label=prompt)

    meta = build_metadata(
        detections,
        image_width=image.size[0],
        image_height=image.size[1],
        prompt=prompt,
        confidence_min=confidence_min,
        source_name=source_name,
        inference_time_s=inference_time,
    )

    return detections, annotated, meta


def segment_local_image(processor, image_path: str, prompt: str,
                        confidence_min: float = 0.5,
                        save_dir: Path | None = None) -> dict | None:
    """Segment a single local file, save annotated image + JSON metadata."""
    from PIL import Image

    image_path = Path(image_path)
    if not image_path.exists():
        _fail(f"Image not found: {image_path}")
        return None

    _info(f"Processing: {image_path.name}")
    image = Image.open(image_path).convert("RGB")
    detections, annotated, meta = segment_image(
        processor, image, prompt,
        confidence_min=confidence_min,
        source_name=str(image_path),
    )

    count = meta["detection_count"]
    ratio = meta["summary"]["total_segmented_ratio"]
    print(f"  → {count} '{prompt}' detection(s), segmented area = {ratio:.2%} of image")

    out_dir = save_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_seg{image_path.suffix}"
    annotated.save(str(out_path))
    _ok(f"Image  → {out_path}")
    _save_metadata(meta, out_path)

    return meta


# ═════════════════════════════════════════════════════════════════════════════
#  GCS helpers
# ═════════════════════════════════════════════════════════════════════════════

def _import_gcs_utils():
    """Import gcs_utils from the api-trials directory."""
    if str(API_TRIALS_DIR) not in sys.path:
        sys.path.insert(0, str(API_TRIALS_DIR))
    try:
        import gcs_utils
        return gcs_utils
    except ImportError as exc:
        _fail(f"Cannot import gcs_utils: {exc}")
        _info(f"Expected at: {API_TRIALS_DIR / 'gcs_utils.py'}")
        _info("Make sure google-cloud-storage is installed: pip install google-cloud-storage")
        raise _PhaseError("gcs_utils not importable") from exc


def segment_gcs_prefix(processor, prefix: str, prompt: str,
                       confidence_min: float = 0.5,
                       save_dir: Path | None = None) -> list[dict]:
    """Download images from a GCS prefix, segment each, save results.

    Returns list of per-image metadata dicts.  Also writes a
    ``_batch_summary.json`` with aggregate stats across all images.
    """
    from PIL import Image

    gcs_utils = _import_gcs_utils()

    blobs = gcs_utils.list_blobs(prefix=prefix)
    image_blobs = [b for b in blobs if b.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_blobs:
        _info(f"No images found under gs://…/{prefix}")
        return []

    _info(f"Found {len(image_blobs)} images under prefix '{prefix}'")

    out_dir = save_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta: list[dict] = []

    for blob_name in image_blobs:
        _info(f"Downloading {blob_name} …")
        local_path = gcs_utils.download_file(blob_name)
        image = Image.open(local_path).convert("RGB")

        detections, annotated, meta = segment_image(
            processor, image, prompt,
            confidence_min=confidence_min,
            source_name=f"gs://…/{blob_name}",
        )

        count = meta["detection_count"]
        ratio = meta["summary"]["total_segmented_ratio"]
        print(f"  → {count} '{prompt}' detection(s), segmented area = {ratio:.2%} of image")

        stem = Path(blob_name).stem
        out_path = out_dir / f"{stem}_seg.jpg"
        annotated.save(str(out_path))
        _ok(f"Image  → {out_path}")
        _save_metadata(meta, out_path)

        all_meta.append(meta)
        os.remove(local_path)

    # ── Batch summary ────────────────────────────────────────────────────
    if all_meta:
        total_dets = sum(m["detection_count"] for m in all_meta)
        ratios = [m["summary"]["total_segmented_ratio"] for m in all_meta]
        batch_summary = {
            "batch_timestamp": datetime.now(timezone.utc).isoformat(),
            "gcs_prefix": prefix,
            "prompt": prompt,
            "confidence_threshold": confidence_min,
            "images_processed": len(all_meta),
            "total_detections": total_dets,
            "avg_detections_per_image": round(total_dets / len(all_meta), 2),
            "avg_segmented_ratio": round(sum(ratios) / len(ratios), 6),
            "min_segmented_ratio": round(min(ratios), 6),
            "max_segmented_ratio": round(max(ratios), 6),
        }
        summary_path = out_dir / "_batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2)
        _ok(f"Batch summary → {summary_path}")

    return all_meta


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SAM3 text-prompted segmentation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preflight", action="store_true",
                   help="Run environment / GPU / SAM3 checks only (no model load)")
    p.add_argument("--image", type=str,
                   help="Path to a local image to segment")
    p.add_argument("--gcs-prefix", type=str,
                   help="GCS blob prefix to fetch images from")
    p.add_argument("--prompt", type=str, default="sidewalk",
                   help="Text prompt for segmentation (default: 'sidewalk')")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="Minimum detection confidence (default: 0.5)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to save annotated images (default: inference-pipeline/output/)")
    return p


def main():
    args = build_parser().parse_args()

    _banner("SAM3 Inference Pipeline")
    _info(f"Project root : {PROJECT_ROOT}")
    _info(f"Script dir   : {SCRIPT_DIR}")
    _info(f"Python       : {sys.executable}")

    # ── Preflight phases ─────────────────────────────────────────────────
    try:
        check_environment()
        torch = check_gpu()
        check_sam3()
    except _PhaseError as exc:
        print(f"\n  ** Preflight FAILED at: {exc} **")
        print("  Fix the issue above and re-run.\n")
        sys.exit(1)

    _banner("All preflight checks passed")

    if args.preflight:
        print("\n  --preflight flag set; exiting without loading the model.\n")
        return

    # ── Load model ───────────────────────────────────────────────────────
    try:
        _, processor = load_model()
    except Exception as exc:
        _fail(f"Model loading failed: {exc}")
        sys.exit(1)

    # ── Run inference ────────────────────────────────────────────────────
    save_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    if args.image:
        segment_local_image(processor, args.image, args.prompt,
                            confidence_min=args.confidence, save_dir=save_dir)
    elif args.gcs_prefix:
        segment_gcs_prefix(processor, args.gcs_prefix, args.prompt,
                           confidence_min=args.confidence, save_dir=save_dir)
    else:
        _info("No --image or --gcs-prefix given. Nothing to segment.")
        _info("Run with --help to see usage.")


if __name__ == "__main__":
    main()
