"""
SAM3 Inference Pipeline
───────────────────────
Loads SAM3 on a CUDA GPU, optionally fetches images from GCS, runs
text-prompted segmentation, and saves annotated results back to GCS.

Usage:
    # Run from a YAML config file (recommended)
    python main.py --config config.yaml

    # Run all preflight checks without loading the model
    python main.py --preflight

    # Segment a local image
    python main.py --image /path/to/image.jpg --prompt "sidewalk"

    # Segment all images under a GCS prefix and save results locally
    python main.py --gcs-prefix streetview/istanbul/ --prompt "sidewalk"
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Resolve project paths relative to *this file*, not cwd ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # inference-pipeline/
PROJECT_ROOT = SCRIPT_DIR.parent                       # cv-urban-accessibility-senior-project/
SAM3_REPO_DIR = PROJECT_ROOT / "sam3-trials" / "local-gpu-setup" / "sam3"
BPE_PATH = SAM3_REPO_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
API_TRIALS_DIR = PROJECT_ROOT / "api-trials"
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = SCRIPT_DIR / "output"


DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


# ═════════════════════════════════════════════════════════════════════════════
#  YAML Configuration
# ═════════════════════════════════════════════════════════════════════════════

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration file."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    defaults: dict[str, Any] = {
        "gcs": {
            "bucket_name": "",
            "project_id": "",
            "input_image": "",
            "output_prefix": "segmentation-results/",
        },
        "batch": {
            "enabled": False,
            "input_prefix": "",
            "prefix_min": 0,
            "prefix_max": 9999,
        },
        "model": {"confidence_threshold": 0.3},
        "inference": {"prompt": "sidewalk", "confidence_min": 0.5},
        "local_output_dir": None,
    }

    for section, section_defaults in defaults.items():
        if isinstance(section_defaults, dict):
            cfg.setdefault(section, {})
            for k, v in section_defaults.items():
                cfg[section].setdefault(k, v)
        else:
            cfg.setdefault(section, section_defaults)

    return cfg


# ═════════════════════════════════════════════════════════════════════════════
#  GCS Client (standalone – no dependency on api-trials/config.py)
# ═════════════════════════════════════════════════════════════════════════════

class GCSClient:
    """Thin wrapper around google-cloud-storage driven by explicit config."""

    def __init__(self, project_id: str, bucket_name: str):
        from google.cloud import storage
        self._client = storage.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def download_to_tempfile(self, blob_name: str, suffix: str = ".jpg") -> str:
        """Download a blob into a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self._bucket.blob(blob_name).download_to_filename(tmp.name)
        tmp.close()
        return tmp.name

    def upload_bytes(self, data: bytes, blob_name: str,
                     content_type: str = "image/jpeg",
                     metadata: dict[str, str] | None = None) -> str:
        blob = self._bucket.blob(blob_name)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{blob_name}"

    def upload_file(self, local_path: str, blob_name: str,
                    metadata: dict[str, str] | None = None) -> str:
        blob = self._bucket.blob(blob_name)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_filename(local_path)
        return f"gs://{self.bucket_name}/{blob_name}"

    def list_blobs(self, prefix: str) -> list[str]:
        """List blob names under *prefix*, returning only image files."""
        blobs = self._bucket.list_blobs(prefix=prefix)
        return [
            b.name for b in blobs
            if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]


def _extract_numeric_prefix(filename: str) -> int | None:
    """Return the leading integer from a filename, or None.

    '0598_forward_40.97_29.05_123.0.jpg'  →  598
    """
    m = re.match(r"(\d+)", Path(filename).name)
    return int(m.group(1)) if m else None


_FILENAME_RE = re.compile(
    r"^(\d+)_(forward|backward|left|right)_([-\d.]+)_([-\d.]+)_([\d.]+)\.\w+$"
)


def _parse_image_filename(filename: str) -> dict[str, str] | None:
    """Parse a streetview image filename into its components.

    '0598_forward_40.9715456_29.0555065_123.0.jpg'
    → { index: '0598', direction: 'forward',
        lat: '40.9715456', lon: '29.0555065', heading: '123.0',
        coordinate_folder: '0598_40.9715456_29.0555065' }
    """
    m = _FILENAME_RE.match(Path(filename).name)
    if not m:
        return None
    return {
        "index": m.group(1),
        "direction": m.group(2),
        "lat": m.group(3),
        "lon": m.group(4),
        "heading": m.group(5),
        "coordinate_folder": f"{m.group(1)}_{m.group(3)}_{m.group(4)}",
    }


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
        labels = [f"{c:.2f}" for c in detections.confidence]
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
#  GCS single-image workflow (config-driven)
# ═════════════════════════════════════════════════════════════════════════════

def segment_gcs_image(
    processor,
    gcs_client: GCSClient,
    input_blob: str,
    output_prefix: str,
    prompt: str,
    confidence_min: float = 0.5,
    local_output_dir: Path | None = None,
) -> dict:
    """Fetch one image from GCS, segment it, and upload results back.

    Uploads two artefacts to ``output_prefix`` inside the same bucket:
        <stem>_seg.jpg   – annotated image
        <stem>_seg.json  – detection metadata

    Returns the metadata dict.
    """
    from PIL import Image

    stem = Path(input_blob).stem
    suffix = Path(input_blob).suffix or ".jpg"

    _info(f"Downloading gs://{gcs_client.bucket_name}/{input_blob} …")
    tmp_path = gcs_client.download_to_tempfile(input_blob, suffix=suffix)

    try:
        image = Image.open(tmp_path).convert("RGB")
        detections, annotated, meta = segment_image(
            processor, image, prompt,
            confidence_min=confidence_min,
            source_name=f"gs://{gcs_client.bucket_name}/{input_blob}",
        )

        count = meta["detection_count"]
        ratio = meta["summary"]["total_segmented_ratio"]
        print(f"  → {count} '{prompt}' detection(s), segmented area = {ratio:.2%} of image")

        # ── Serialize annotated image to bytes ────────────────────────
        img_buf = io.BytesIO()
        annotated.save(img_buf, format="JPEG", quality=95)
        img_bytes = img_buf.getvalue()

        meta_bytes = json.dumps(meta, indent=2).encode()

        # ── Upload to GCS ─────────────────────────────────────────────
        out_prefix = output_prefix.rstrip("/")
        img_blob = f"{out_prefix}/{stem}_seg.jpg"
        json_blob = f"{out_prefix}/{stem}_seg.json"

        img_uri = gcs_client.upload_bytes(
            img_bytes, img_blob, content_type="image/jpeg",
        )
        _ok(f"Image    → {img_uri}")

        json_uri = gcs_client.upload_bytes(
            meta_bytes, json_blob, content_type="application/json",
        )
        _ok(f"Metadata → {json_uri}")

        # ── Optional local save ───────────────────────────────────────
        if local_output_dir:
            local_dir = Path(local_output_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            local_img = local_dir / f"{stem}_seg.jpg"
            local_json = local_dir / f"{stem}_seg.json"
            annotated.save(str(local_img))
            with open(local_json, "w") as f:
                json.dump(meta, f, indent=2)
            _ok(f"Local copy → {local_dir}")

    finally:
        os.remove(tmp_path)

    return meta


# ═════════════════════════════════════════════════════════════════════════════
#  GCS batch workflow with per-object masks
# ═════════════════════════════════════════════════════════════════════════════

def _mask_to_png_bytes(mask_array) -> bytes:
    """Convert a boolean numpy mask to PNG bytes (white=object, black=bg)."""
    from PIL import Image
    import numpy as np
    img = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def segment_gcs_batch_masks(
    processor,
    gcs_client: GCSClient,
    input_prefix: str,
    output_prefix: str,
    prompt: str,
    confidence_min: float = 0.5,
    prefix_min: int = 0,
    prefix_max: int = 9999,
) -> list[dict]:
    """Segment a filtered range of images from GCS and upload per-object masks.

    Images are filtered by the leading numeric prefix in their filename:
    only those with ``prefix_min <= prefix <= prefix_max`` are processed.

    Uploads per image to GCS::

        <output_prefix>/<index>_<lat>_<lon>/<direction>/<prompt>/
            annotated.jpg
            metadata.json
            mask_000.png
            mask_001.png
            …

    Returns a list of per-image metadata dicts.
    """
    from PIL import Image

    _banner("Batch Mask Extraction")
    _info(f"GCS prefix   : {input_prefix}")
    _info(f"Prefix range : {prefix_min} – {prefix_max}")
    _info(f"Prompt       : {prompt}")
    _info(f"Conf. min    : {confidence_min}")

    all_blobs = gcs_client.list_blobs(input_prefix)
    if not all_blobs:
        _info(f"No images found under gs://{gcs_client.bucket_name}/{input_prefix}")
        return []

    filtered: list[tuple[int, str]] = []
    for blob_name in all_blobs:
        num = _extract_numeric_prefix(blob_name)
        if num is not None and prefix_min <= num <= prefix_max:
            filtered.append((num, blob_name))
    filtered.sort(key=lambda t: t[0])

    if not filtered:
        _info(f"No images matched prefix range {prefix_min}–{prefix_max}")
        return []

    _info(f"Matched {len(filtered)} / {len(all_blobs)} images in range")

    out_root = output_prefix.rstrip("/")
    safe_prompt = re.sub(r"[^\w\-]+", "_", prompt).strip("_")
    all_meta: list[dict] = []

    skipped = 0
    for idx, (num, blob_name) in enumerate(filtered, 1):
        suffix = Path(blob_name).suffix or ".jpg"
        parsed = _parse_image_filename(blob_name)

        if parsed is None:
            _info(f"[{idx}/{len(filtered)}] Skipping (unrecognised name): {Path(blob_name).name}")
            skipped += 1
            continue

        coord_folder = parsed["coordinate_folder"]
        direction = parsed["direction"]
        img_folder = f"{out_root}/{coord_folder}/{direction}/{safe_prompt}"

        _info(f"[{idx}/{len(filtered)}] {Path(blob_name).name}  →  {coord_folder}/{direction}/{safe_prompt}/")
        tmp_path = gcs_client.download_to_tempfile(blob_name, suffix=suffix)

        try:
            image = Image.open(tmp_path).convert("RGB")
            detections, annotated, meta = segment_image(
                processor, image, prompt,
                confidence_min=confidence_min,
                source_name=f"gs://{gcs_client.bucket_name}/{blob_name}",
            )

            count = meta["detection_count"]
            ratio = meta["summary"]["total_segmented_ratio"]
            print(f"     → {count} detection(s), segmented area = {ratio:.2%}")

            # ── Upload annotated image ────────────────────────────────
            img_buf = io.BytesIO()
            annotated.save(img_buf, format="JPEG", quality=95)
            gcs_client.upload_bytes(
                img_buf.getvalue(),
                f"{img_folder}/annotated.jpg",
                content_type="image/jpeg",
            )

            # ── Upload metadata ───────────────────────────────────────
            gcs_client.upload_bytes(
                json.dumps(meta, indent=2).encode(),
                f"{img_folder}/metadata.json",
                content_type="application/json",
            )

            # ── Upload individual masks ───────────────────────────────
            if detections.mask is not None:
                for i in range(len(detections)):
                    mask_bytes = _mask_to_png_bytes(detections.mask[i])
                    gcs_client.upload_bytes(
                        mask_bytes,
                        f"{img_folder}/mask_{i:03d}.png",
                        content_type="image/png",
                    )

            _ok(f"Uploaded {count} masks → gs://…/{img_folder}/")
            all_meta.append(meta)
        finally:
            os.remove(tmp_path)

    # ── Batch summary ────────────────────────────────────────────────────
    _banner("Batch Complete")
    if skipped:
        _info(f"Skipped {skipped} file(s) with unrecognised names")
    total_dets = sum(m["detection_count"] for m in all_meta)
    ratios = [m["summary"]["total_segmented_ratio"] for m in all_meta]
    summary_blob = f"{out_root}/_batch_summary_{safe_prompt}.json"
    batch_summary = {
        "batch_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_prefix": input_prefix,
        "prefix_range": [prefix_min, prefix_max],
        "prompt": prompt,
        "confidence_threshold": confidence_min,
        "images_processed": len(all_meta),
        "images_skipped": skipped,
        "total_detections": total_dets,
        "avg_detections_per_image": round(total_dets / len(all_meta), 2) if all_meta else 0,
        "avg_segmented_ratio": round(sum(ratios) / len(ratios), 6) if ratios else 0,
    }
    gcs_client.upload_bytes(
        json.dumps(batch_summary, indent=2).encode(),
        summary_blob,
        content_type="application/json",
    )

    _info(f"Images processed : {len(all_meta)}")
    _info(f"Total detections : {total_dets}")
    _info(f"Avg segmented    : {batch_summary['avg_segmented_ratio']:.2%}")
    _ok(f"Summary → gs://…/{summary_blob}")

    return all_meta


# ═════════════════════════════════════════════════════════════════════════════
#  GCS batch helpers (legacy CLI path)
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
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file (default: inference-pipeline/config.yaml)")
    p.add_argument("--preflight", action="store_true",
                   help="Run environment / GPU / SAM3 checks only (no model load)")
    p.add_argument("--image", type=str,
                   help="Path to a local image to segment")
    p.add_argument("--gcs-prefix", type=str,
                   help="GCS blob prefix to fetch images from (batch mode)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt for segmentation (overrides config)")
    p.add_argument("--confidence", type=float, default=None,
                   help="Minimum detection confidence (overrides config)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to save annotated images locally (overrides config)")
    return p


def _run_config_mode(processor, cfg: dict[str, Any]) -> None:
    """Execute the config-driven GCS single-image workflow."""
    gcs_cfg = cfg["gcs"]
    inf_cfg = cfg["inference"]

    input_image = gcs_cfg["input_image"]
    if not input_image:
        _fail("gcs.input_image is empty in config – nothing to segment.")
        return

    gcs_client = GCSClient(
        project_id=gcs_cfg["project_id"],
        bucket_name=gcs_cfg["bucket_name"],
    )

    local_out = Path(cfg["local_output_dir"]) if cfg.get("local_output_dir") else None

    _banner("Config-driven GCS Segmentation")
    _info(f"Input blob   : {input_image}")
    _info(f"Output prefix: {gcs_cfg['output_prefix']}")
    _info(f"Prompt       : {inf_cfg['prompt']}")
    _info(f"Conf. min    : {inf_cfg['confidence_min']}")

    meta = segment_gcs_image(
        processor=processor,
        gcs_client=gcs_client,
        input_blob=input_image,
        output_prefix=gcs_cfg["output_prefix"],
        prompt=inf_cfg["prompt"],
        confidence_min=inf_cfg["confidence_min"],
        local_output_dir=local_out,
    )

    _banner("Done")
    _info(f"Detections: {meta['detection_count']}")
    _info(f"Segmented area: {meta['summary']['total_segmented_ratio']:.2%}")
    if meta.get("inference_time_s"):
        _info(f"Inference time: {meta['inference_time_s']:.2f}s")


def _run_batch_mode(processor, cfg: dict[str, Any]) -> None:
    """Execute the config-driven batch mask-extraction workflow."""
    gcs_cfg = cfg["gcs"]
    batch_cfg = cfg["batch"]
    inf_cfg = cfg["inference"]

    gcs_client = GCSClient(
        project_id=gcs_cfg["project_id"],
        bucket_name=gcs_cfg["bucket_name"],
    )

    segment_gcs_batch_masks(
        processor=processor,
        gcs_client=gcs_client,
        input_prefix=batch_cfg["input_prefix"],
        output_prefix=gcs_cfg["output_prefix"],
        prompt=inf_cfg["prompt"],
        confidence_min=inf_cfg["confidence_min"],
        prefix_min=batch_cfg["prefix_min"],
        prefix_max=batch_cfg["prefix_max"],
    )


def main():
    args = build_parser().parse_args()

    # ── Load YAML config (always, for defaults) ──────────────────────────
    cfg = load_config(args.config)

    # CLI overrides take precedence over the YAML values
    if args.prompt is not None:
        cfg["inference"]["prompt"] = args.prompt
    if args.confidence is not None:
        cfg["inference"]["confidence_min"] = args.confidence
    if args.output_dir is not None:
        cfg["local_output_dir"] = args.output_dir

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
    model_threshold = cfg["model"]["confidence_threshold"]
    try:
        _, processor = load_model(confidence_threshold=model_threshold)
    except Exception as exc:
        _fail(f"Model loading failed: {exc}")
        sys.exit(1)

    # ── Dispatch: CLI flags → batch mode → single-image config ─────────
    prompt = cfg["inference"]["prompt"]
    conf_min = cfg["inference"]["confidence_min"]
    save_dir = Path(cfg["local_output_dir"]) if cfg.get("local_output_dir") else OUTPUT_DIR

    if args.image:
        segment_local_image(processor, args.image, prompt,
                            confidence_min=conf_min, save_dir=save_dir)
    elif args.gcs_prefix:
        segment_gcs_prefix(processor, args.gcs_prefix, prompt,
                           confidence_min=conf_min, save_dir=save_dir)
    elif cfg["batch"].get("enabled"):
        _run_batch_mode(processor, cfg)
    elif cfg["gcs"].get("input_image"):
        _run_config_mode(processor, cfg)
    else:
        _info("No --image, --gcs-prefix, batch.enabled, or gcs.input_image in config.")
        _info("Run with --help to see usage.")


if __name__ == "__main__":
    main()
