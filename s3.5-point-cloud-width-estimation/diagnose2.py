#!/usr/bin/env python3
"""Quick check: what does the expanded mask look like and why did width drop?"""

import numpy as np
import cv2
import torch
import contextlib

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from evaluate_sidewalk import (
    find_contiguous_runs, batch_bilinear_sample,
    fit_ground_plane, project_to_plane, gather_ground_points,
)

IMAGE = "side_image.jpg"
CAM_HEIGHT = 2.5
MIN_RUN_PX = 15
BAND_FRAC = 0.20

device = "mps" if torch.backends.mps.is_available() else "cpu"

# VGGT
print("Loading VGGT...")
vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device); vggt.eval()
img_tensor = load_and_preprocess_images([IMAGE]).to(device)
with torch.inference_mode():
    try: amp = torch.amp.autocast("mps", dtype=torch.float16)
    except: amp = contextlib.nullcontext()
    with amp: pred = vggt(img_tensor)
wp = pred["world_points"]
if wp.dim() == 5: wp = wp[0]
wp = wp[0].detach().cpu().numpy()
pe = pred["pose_enc"]
if pe.dim() == 3: pe = pe[0]
pose_enc = pe[0].detach().cpu().numpy()
del pred, img_tensor

# SegFormer
print("Loading SegFormer...")
seg_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
proc = AutoImageProcessor.from_pretrained(seg_id)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    seg_id, revision="refs/pr/3", use_safetensors=True).to(device)
seg_model.eval()
id2label = {int(k): v for k, v in seg_model.config.id2label.items()}
label2id = {v.lower(): k for k, v in id2label.items()}
ID_SW, ID_RD = label2id["sidewalk"], label2id["road"]

img_bgr = cv2.imread(IMAGE)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H_img, W_img = img_bgr.shape[:2]

with torch.inference_mode():
    inputs = proc(images=[img_rgb], return_tensors="pt").to(device)
    with amp: out = seg_model(**inputs)
    seg_list = proc.post_process_semantic_segmentation(out, target_sizes=[img_rgb.shape[:2]])
    seg_map = seg_list[0].cpu().numpy().astype(np.int32)

# Original mask
sw_mask_orig = (seg_map == ID_SW).astype(np.uint8)
min_area = max(50, int(0.001 * H_img * W_img))
num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(sw_mask_orig, connectivity=8)
for k in range(1, num_cc):
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        sw_mask_orig[labels == k] = 0
sw_mask_orig = cv2.morphologyEx(sw_mask_orig, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

# Merged mask
sw_mask = sw_mask_orig.copy()
merge_ids = [label2id["vegetation"], label2id["terrain"]]
merge_mask = np.zeros_like(sw_mask)
for mid in merge_ids:
    merge_mask |= (seg_map == mid).astype(np.uint8)
for _ in range(20):
    candidate = cv2.dilate(sw_mask, np.ones((3,3), np.uint8)) & merge_mask
    new_pixels = candidate & (~sw_mask.astype(bool)).astype(np.uint8)
    if new_pixels.sum() == 0: break
    sw_mask = sw_mask | new_pixels
sw_mask = cv2.morphologyEx(sw_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

# Visualize both masks
vis = img_bgr.copy()
# Original in green
vis[sw_mask_orig > 0] = (vis[sw_mask_orig > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
# Added pixels in yellow
added = (sw_mask > 0) & (sw_mask_orig == 0)
vis[added] = (vis[added] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)
cv2.imwrite("eval_boun/diag_merged_mask.png", vis)
print("Saved merged mask to eval_boun/diag_merged_mask.png")

# Column analysis with merged mask
x_mid = W_img // 2
half_band = max(10, int(W_img * BAND_FRAC / 2))
x_start, x_end = max(0, x_mid - half_band), min(W_img, x_mid + half_band)

y_centers = []
for x in range(x_start, x_end):
    col = sw_mask[:, x]
    if col.any():
        ys = np.where(col > 0)[0]
        y_centers.append(float(np.median(ys)))
y_cg = np.median(y_centers)

edge_xs, edge_y_tops, edge_y_bots = [], [], []
n_border = 0
for x in range(x_start, x_end):
    col = sw_mask[:, x].astype(bool)
    runs = find_contiguous_runs(col)
    if not runs: continue
    best_run, best_score = None, -1.0
    for rs, re in runs:
        length = re - rs
        if length < MIN_RUN_PX: continue
        mid_y = (rs + re) / 2.0
        dist = abs(mid_y - y_cg)
        score = length / (1.0 + dist / 50.0)
        if score > best_score:
            best_score = score
            best_run = (rs, re)
    if best_run is None: continue
    y_top, y_bot = best_run[0], best_run[1] - 1
    if y_top <= 1 or y_bot >= H_img - 2:
        n_border += 1
        continue
    edge_xs.append(x)
    edge_y_tops.append(y_top)
    edge_y_bots.append(y_bot)

print(f"\nWith merged mask:")
print(f"  Valid columns: {len(edge_xs)}, border-skipped: {n_border}")
if edge_xs:
    edge_y_tops = np.array(edge_y_tops)
    edge_y_bots = np.array(edge_y_bots)
    print(f"  Median y_top: {np.median(edge_y_tops):.1f}")
    print(f"  Median y_bot: {np.median(edge_y_bots):.1f}")
    print(f"  Median pixel gap: {np.median(edge_y_bots - edge_y_tops):.1f}")
    
# Now do same with original mask
edge_xs2, edge_y_tops2, edge_y_bots2 = [], [], []
n_border2 = 0
y_centers2 = []
for x in range(x_start, x_end):
    col = sw_mask_orig[:, x]
    if col.any():
        ys = np.where(col > 0)[0]
        y_centers2.append(float(np.median(ys)))
y_cg2 = np.median(y_centers2) if y_centers2 else 0

for x in range(x_start, x_end):
    col = sw_mask_orig[:, x].astype(bool)
    runs = find_contiguous_runs(col)
    if not runs: continue
    best_run, best_score = None, -1.0
    for rs, re in runs:
        length = re - rs
        if length < MIN_RUN_PX: continue
        mid_y = (rs + re) / 2.0
        dist = abs(mid_y - y_cg2)
        score = length / (1.0 + dist / 50.0)
        if score > best_score:
            best_score = score
            best_run = (rs, re)
    if best_run is None: continue
    y_top, y_bot = best_run[0], best_run[1] - 1
    if y_top <= 1 or y_bot >= H_img - 2:
        n_border2 += 1
        continue
    edge_xs2.append(x)
    edge_y_tops2.append(y_top)
    edge_y_bots2.append(y_bot)

print(f"\nWith original mask:")
print(f"  Valid columns: {len(edge_xs2)}, border-skipped: {n_border2}")
if edge_xs2:
    edge_y_tops2 = np.array(edge_y_tops2)
    edge_y_bots2 = np.array(edge_y_bots2)
    print(f"  Median y_top: {np.median(edge_y_tops2):.1f}")
    print(f"  Median y_bot: {np.median(edge_y_bots2):.1f}")
    print(f"  Median pixel gap: {np.median(edge_y_bots2 - edge_y_tops2):.1f}")

# Show what's happening at the borders — check a few merged columns
print(f"\n--- Sample merged-mask columns ---")
for x in [256, 288, 320, 352, 384]:
    if x >= W_img: continue
    col = sw_mask[:, x].astype(bool)
    runs = find_contiguous_runs(col)
    print(f"  Col {x}: runs={runs}, touches_top={col[0] or col[1]}, touches_bot={col[-1] or col[-2]}")
