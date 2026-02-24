# SAM3 Local Inference

Run [Segment Anything 3](https://github.com/facebookresearch/sam3) image segmentation locally on macOS (CPU/MPS) or Linux (CUDA).

## Prerequisites

- Python 3.10 or 3.11
- A [HuggingFace](https://huggingface.co/) account with access to the SAM3 model weights
- ~4 GB disk space for model weights (downloaded on first run)

## Setup

1. **Create and activate a virtual environment** from the project root:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set up SAM3** by running the setup script, which clones the repo and applies the macOS compatibility patches:

   ```bash
   cd sam3-trials
   ./setup_sam3.sh
   ```

   This will:
   - Clone the SAM3 repository into `sam3-trials/sam3/`
   - Apply `sam3-macos-compat.patch` (CPU/MPS compatibility fixes)
   - Install SAM3 as an editable package along with extra dependencies

3. **Configure your HuggingFace token** by creating a `.env` file in the project root:

   ```
   HF_TOKEN=hf_your_token_here
   ```

   The model weights are downloaded from HuggingFace on first run. You need approved access to the [SAM3 model](https://github.com/facebookresearch/sam3).

## Running the notebook

Open the notebook in Jupyter or VS Code:

```bash
cd sam3-trials
jupyter notebook how_to_segment_images_with_segment_anything_3.ipynb
```

Run cells from top to bottom. The notebook will:
1. Detect available hardware (CUDA GPU, Apple MPS, or CPU)
2. Download sample images on first run
3. Load the SAM3 model (downloads weights on first use, ~4 GB)
4. Run text-prompted and interactive segmentation examples

## What the patch fixes

The upstream SAM3 code assumes CUDA is available. The patch (`sam3-macos-compat.patch`) makes these changes:

| File | Fix |
|---|---|
| `edt.py` | Adds scipy CPU fallback for the Triton-based Euclidean Distance Transform |
| `position_encoding.py`, `decoder.py` | Replaces hardcoded `device="cuda"` with auto-detection |
| `sam3_image_processor.py`, `vl_combiner.py` | Makes `device` parameter default to auto-detection |
| `geometry_encoders.py`, `sam3_video_inference.py`, `sam3_tracker_base.py` | Removes `pin_memory()` calls that fail on MPS |
| `model_builder.py` | Adds fallback for BPE vocabulary path resolution |
| `sam3_image_dataset.py` | Makes `decord` import optional (unavailable on macOS via pip) |
| `pyproject.toml` | Relaxes `numpy<2` constraint |

## Performance notes

- **CUDA GPU**: Full speed, as designed by Meta
- **Apple MPS** (M1/M2/M3): Works but slower; first inference takes a few minutes as MPS compiles kernels
- **CPU**: Functional but slow; expect several minutes per image
