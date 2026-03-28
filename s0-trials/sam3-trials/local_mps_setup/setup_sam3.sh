#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAM3_DIR="$SCRIPT_DIR/sam3"

if [ -d "$SAM3_DIR/.git" ]; then
    echo "sam3 repo already exists at $SAM3_DIR"
else
    echo "Cloning sam3..."
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
fi

echo "Applying macOS CPU/MPS compatibility patch..."
cd "$SAM3_DIR"
git checkout main
git checkout -b macos-cpu-mps-compat 2>/dev/null || git checkout macos-cpu-mps-compat
git apply "$SCRIPT_DIR/sam3-macos-compat.patch"
git add -A
git commit -m "Add macOS CPU/MPS compatibility patches"

echo "Installing sam3 as editable package..."
pip install -e "$SAM3_DIR"
pip install scipy pycocotools supervision jupyter_bbox_widget

echo "Done! sam3 is ready for local use."
