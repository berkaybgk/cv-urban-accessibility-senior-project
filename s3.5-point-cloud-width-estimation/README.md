# Point-cloud sidewalk width estimation

This folder estimates **sidewalk width in meters** from street-view style images using **VGGT** (dense 3D / camera from pixels) plus **SegFormer** (Cityscapes sidewalk vs road). The main entry point is `evaluate_sidewalk.py`.

Upstream VGGT: [facebookresearch/vggt](https://github.com/facebookresearch/vggt).

## What you need

- **Python 3.10+** (matches the `vggt` package).
- **Git** (so `pip` can install `vggt` from GitHub; see fallback below if that fails).
- Enough **disk and RAM** for VGGT-1B; on Apple Silicon, **MPS** is used automatically when available (`--device auto`).

## 1. Virtual environment and Python deps

From this directory:

```bash
cd s3.5-point-cloud-width-estimation
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

**NVIDIA GPU:** install PyTorch with the CUDA build from [pytorch.org](https://pytorch.org) *before* or *instead of* relying on the CPU wheels pulled by `requirements.txt`, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install everything else:

```bash
pip install -r requirements.txt
```

If `pip` cannot clone GitHub (proxy, no git), clone VGGT next to this folder and install it in editable mode, then remove the `vggt @ git+...` line from `requirements.txt` and run `pip install -r requirements.txt` again:

```bash
git clone https://github.com/facebookresearch/vggt.git ../vggt
pip install -e ../vggt
```

## 2. VGGT weights (manual download)

The script can load **`model.pt`** from disk so you do not need to fetch VGGT from Hugging Face at runtime.

1. Download **`model.pt`** for [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B/tree/main) (or the file you were approved for, if you use another checkpoint).
2. Place it in this repo under `weights/`, e.g. `weights/model.pt`. That path is **gitignored** so large files are not committed.
3. Point the script at the file in either way:

```bash
export VGGT_WEIGHTS="$PWD/weights/model.pt"
```

or per run:

```bash
python evaluate_sidewalk.py --vggt_weights weights/model.pt ...
```

If neither `VGGT_WEIGHTS` nor `--vggt_weights` is set, the script calls `VGGT.from_pretrained("facebook/VGGT-1B")`, which downloads via Hugging Face. Check the [VGGT license / checkpoint terms](https://github.com/facebookresearch/vggt) for your use case.

## 3. SegFormer weights

`nvidia/segformer-b5-finetuned-cityscapes-1024-1024` is loaded from **Hugging Face** the first time (cached under `~/.cache/huggingface` by default). No extra manual step unless you are offline after the first cache.

## 4. Running

**Single image** (filename does not need to be numeric):

```bash
python evaluate_sidewalk.py \
  --image side_image.jpg \
  --vggt_weights weights/model.pt \
  --out_dir eval_one \
  --visualize \
  --plot3d \
  --verbose
```

**Batch** (expects `img_dir` with `.jpg` / `.png`; basenames that are **integers** are used as IDs, e.g. `42.jpg`):

```bash
python evaluate_sidewalk.py \
  --img_dir streetview_out \
  --gt_csv submission_result.csv \
  --vggt_weights weights/model.pt \
  --out_dir eval_out \
  --visualize
```

Optional **`--gt_csv`**: if the file is missing or empty, evaluation metrics are skipped; predictions are still written to `predicted_widths.csv`.

Useful flags:

| Flag | Role |
|------|------|
| `--device auto\|cpu\|mps\|cuda` | `auto`: CUDA, else MPS, else CPU |
| `--vggt_weights PATH` | Local VGGT `model.pt` |
| `--image PATH` | One image; ID defaults to `0` if the stem is not an integer (`--image_id` overrides) |
| `--visualize` | 2D overlay: midline band and inner/outer edges |
| `--plot3d` | Two 3D PNGs: scene + plane + camera, and per-column width chords |
| `--cam_height` | Camera height in meters for scale (default `2.5`, typical for GSV-style imagery) |

### GCS batch mode for alternative (left/right) estimation

This repo now includes `batch_gcs_alt_width.py` to run the stronger point-cloud method
on GCS images directly and upload outputs for the web app.

1) Edit `batch_gcs_alt_width.yaml`:
- `gcs.image_prefix`: source run folder (for example `streetview/polygon_4v/20260404T133859Z/`)
- `batch.point_id_min` / `batch.point_id_max`
- `batch.directions`: usually `["left", "right"]`
- `gcs.output_prefix`: where artifacts are uploaded (default `v3/alternative-width-results/`)

2) Run:

```bash
python batch_gcs_alt_width.py --config batch_gcs_alt_width.yaml
```

Per point+direction uploads:
- `alt_width_metadata.json` (always written, includes status + width when successful)
- `alt_width_overlay.png` (if enabled)
- `alt_width_scene3d.png` and `alt_width_width3d.png` (if enabled)

Batch summary:
- `<output_prefix>/_alt_width_batch_summary.json`

## 5. Outputs (under `--out_dir`)

- `predicted_widths.csv` — per-ID predicted width (and GT columns when available)
- `skip_report.csv` — reasons for skipped frames
- `vis_<stem>.png` — if `--visualize`
- `geom3d_<stem>_scene3d.png`, `geom3d_<stem>_width3d.png` — if `--plot3d` and the frame was measured successfully

## 6. Troubleshooting

- **Out of memory on Apple Silicon:** try `--device cpu` (slower, sometimes more stable) or close other GPU-heavy apps.
- **SegFormer / Hub errors:** run once online; or set `HF_HOME` to a shared cache; use `huggingface-cli login` only if a model requires it.
- **`pip install` git clone errors:** use the editable `../vggt` install path above.
