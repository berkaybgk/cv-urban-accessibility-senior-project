# SAM3 GPU Server Notebook

This notebook is optimized for running SAM3 on a CUDA-enabled GPU server.

## Key Differences from Other Versions

### vs. Online Colab Notebook
- Uses `.env` file for HuggingFace token instead of Colab secrets
- Uses local file paths instead of `/content/` paths
- Creates a `data/` directory for sample images
- Checks if SAM3 is already cloned before cloning again

### vs. MacBook MPS Version  
- **Does NOT apply any patches** - uses the official SAM3 repo as-is
- Optimized for CUDA GPUs (no MPS/CPU compatibility needed)
- Includes TF32 optimizations for Ampere+ GPUs (compute capability ≥ 8)
- Full hardware acceleration without performance compromises

## Prerequisites

- Python 3.10 or 3.11
- CUDA-enabled GPU with drivers installed
- HuggingFace account with access to SAM3 model weights
- ~4 GB disk space for model weights

## Setup

1. **Create a `.env` file** in the project root with your HuggingFace token:

   ```
   HF_TOKEN=hf_your_token_here
   ```

2. **Run the notebook** in Jupyter or JupyterLab:

   ```bash
   cd sam3-trials/run_sam3_on_gpu
   jupyter notebook sam3_local_gpu.ipynb
   ```

3. **First run**: Execute cells 1-11 to install dependencies and download sample images

4. **Restart the kernel** after installation (as prompted in the notebook)

5. **Continue**: Run the remaining cells to load the model and perform segmentation

## What It Does

The notebook demonstrates:

- **Text-based segmentation**: Describe objects in natural language (e.g., "taxi", "person", "building")
- **Box-based segmentation**: Draw bounding boxes to segment specific objects
- **Point-based segmentation**: Click points to segment objects interactively
- **Interactive mode**: Real-time refinement with multiple prompts

## Performance

This notebook is optimized for maximum performance on CUDA GPUs:

- **Ampere GPUs (RTX 30xx/40xx, A100, etc.)**: Full speed with TF32 acceleration
- **Turing/Volta GPUs**: Fast performance without TF32
- **Pascal and older**: Functional but slower

First inference takes a few seconds as the model loads into GPU memory. Subsequent inferences are much faster.

## File Structure

```
run_sam3_on_gpu/
├── sam3_local_gpu.ipynb    # Main notebook
├── README.md               # This file
├── sam3/                   # Cloned SAM3 repo (created on first run)
└── data/                   # Sample images (created on first run)
```

## Troubleshooting

**CUDA not available**: Make sure you have:
- NVIDIA GPU drivers installed
- PyTorch installed with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

**Out of memory**: Try reducing batch size or using a model with fewer parameters

**HuggingFace token error**: Make sure you've:
1. Requested access to the SAM3 model on HuggingFace
2. Created a `.env` file in the project root with your token
