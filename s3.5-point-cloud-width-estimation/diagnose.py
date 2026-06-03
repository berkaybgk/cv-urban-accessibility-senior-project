#!/usr/bin/env python3
"""Diagnostic script: dump every intermediate value to find under-prediction cause."""

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
CAM_HEIGHT = 2.5   # assumed
BAND_FRAC = 0.20
MIN_RUN_PX = 15

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- VGGT ---
print("Loading VGGT...")
vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
vggt.eval()

img_tensor = load_and_preprocess_images([IMAGE]).to(device)
with torch.inference_mode():
    try:
        amp = torch.amp.autocast("mps", dtype=torch.float16)
    except Exception:
        amp = contextlib.nullcontext()
    with amp:
        pred = vggt(img_tensor)

wp = pred["world_points"]
if wp.dim() == 5: wp = wp[0]
wp = wp[0].detach().cpu().numpy()   # [H_d, W_d, 3]

pe = pred["pose_enc"]
if pe.dim() == 3: pe = pe[0]
pose_enc = pe[0].detach().cpu().numpy()

del pred, img_tensor
print(f"VGGT world_points shape: {wp.shape}")
print(f"pose_enc: {pose_enc}")
print(f"  cam_center  = {pose_enc[:3]}")
print(f"  quaternion  = {pose_enc[3:7]}")
print(f"  fov_h, fov_w = {pose_enc[7]:.4f}, {pose_enc[8]:.4f} rad "
      f"= {np.degrees(pose_enc[7]):.1f}, {np.degrees(pose_enc[8]):.1f} deg")

# --- SegFormer ---
print("\nLoading SegFormer...")
seg_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
proc = AutoImageProcessor.from_pretrained(seg_id)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    seg_id, revision="refs/pr/3", use_safetensors=True).to(device)
seg_model.eval()

id2label = {int(k): v for k, v in seg_model.config.id2label.items()}
label2id = {v.lower(): k for k, v in id2label.items()}
ID_SW = label2id["sidewalk"]
ID_RD = label2id["road"]

img_bgr = cv2.imread(IMAGE)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H_img, W_img = img_bgr.shape[:2]

with torch.inference_mode():
    inputs = proc(images=[img_rgb], return_tensors="pt").to(device)
    with amp:
        out = seg_model(**inputs)
    seg_list = proc.post_process_semantic_segmentation(out, target_sizes=[img_rgb.shape[:2]])
    seg_map = seg_list[0].cpu().numpy().astype(np.int32)

sw_mask = (seg_map == ID_SW).astype(np.uint8)
rd_mask = (seg_map == ID_RD).astype(np.uint8)
sw_pix = int(sw_mask.sum())
rd_pix = int(rd_mask.sum())
total = H_img * W_img
print(f"\nImage size: {W_img}x{H_img} = {total} pixels")
print(f"Sidewalk pixels: {sw_pix} ({100*sw_pix/total:.1f}%)")
print(f"Road pixels:     {rd_pix} ({100*rd_pix/total:.1f}%)")

# Save segmentation overlay
seg_vis = img_bgr.copy()
seg_vis[sw_mask > 0] = (seg_vis[sw_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
seg_vis[rd_mask > 0] = (seg_vis[rd_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
cv2.imwrite("eval_boun/diag_segmentation.png", seg_vis)
print("Saved segmentation overlay to eval_boun/diag_segmentation.png")

# --- Clean mask (same as pipeline) ---
min_area = max(50, int(0.001 * H_img * W_img))
num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(sw_mask, connectivity=8)
for k in range(1, num_cc):
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        sw_mask[labels == k] = 0
sw_mask = cv2.morphologyEx(sw_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

# --- Ground plane ---
H_d, W_d = wp.shape[:2]
sx = (W_d - 1) / max(W_img - 1, 1)
sy = (H_d - 1) / max(H_img - 1, 1)

ground_pts = gather_ground_points(wp, seg_map, ID_SW, ID_RD)
print(f"\nGround points for plane fit: {len(ground_pts)}")
plane_n, plane_d = fit_ground_plane(ground_pts)
print(f"Ground plane normal: {plane_n}")
print(f"Ground plane d:      {plane_d}")

cam_center = pose_enc[:3].astype(np.float64)
h_pred = abs(float(np.dot(plane_n, cam_center) + plane_d))
scale = CAM_HEIGHT / h_pred
print(f"\nh_pred (camera-to-plane in VGGT units): {h_pred:.6f}")
print(f"scale = cam_height / h_pred = {CAM_HEIGHT} / {h_pred:.6f} = {scale:.4f}")

# --- Column-wise edge detection ---
x_mid = W_img // 2
half_band = max(10, int(W_img * BAND_FRAC / 2))
x_start = max(0, x_mid - half_band)
x_end = min(W_img, x_mid + half_band)
print(f"\nMidline band: x=[{x_start}, {x_end}] (band_frac={BAND_FRAC})")

y_centers = []
for x in range(x_start, x_end):
    col = sw_mask[:, x]
    if col.any():
        ys = np.where(col > 0)[0]
        y_centers.append(float(np.median(ys)))
y_center_global = np.median(y_centers) if y_centers else None
print(f"y_center_global (median sidewalk center): {y_center_global}")

edge_xs, edge_y_tops, edge_y_bots = [], [], []
n_border_skip = 0
for x in range(x_start, x_end):
    col = sw_mask[:, x].astype(bool)
    runs = find_contiguous_runs(col)
    if not runs: continue
    best_run, best_score = None, -1.0
    for rs, re in runs:
        length = re - rs
        if length < MIN_RUN_PX: continue
        mid_y = (rs + re) / 2.0
        dist = abs(mid_y - y_center_global)
        score = length / (1.0 + dist / 50.0)
        if score > best_score:
            best_score = score
            best_run = (rs, re)
    if best_run is None: continue
    y_top, y_bot = best_run[0], best_run[1] - 1
    if y_top <= 1 or y_bot >= H_img - 2:
        n_border_skip += 1
        continue
    edge_xs.append(x)
    edge_y_tops.append(y_top)
    edge_y_bots.append(y_bot)

edge_xs = np.array(edge_xs, dtype=np.float64)
edge_y_tops = np.array(edge_y_tops, dtype=np.float64)
edge_y_bots = np.array(edge_y_bots, dtype=np.float64)

print(f"\nValid columns: {len(edge_xs)}, border-skipped: {n_border_skip}")
print(f"Median y_top (inner edge): {np.median(edge_y_tops):.1f}")
print(f"Median y_bot (outer edge): {np.median(edge_y_bots):.1f}")
med_gap = np.median(edge_y_bots - edge_y_tops)
print(f"Median pixel gap (y_bot - y_top): {med_gap:.1f} px")

# Draw edges on image
edge_vis = img_bgr.copy()
for i in range(len(edge_xs)):
    x = int(edge_xs[i])
    cv2.circle(edge_vis, (x, int(edge_y_tops[i])), 2, (0, 255, 255), -1)
    cv2.circle(edge_vis, (x, int(edge_y_bots[i])), 2, (0, 0, 255), -1)
cv2.imwrite("eval_boun/diag_edges.png", edge_vis)
print("Saved edge visualization to eval_boun/diag_edges.png")

# --- 3D sampling ---
xs_d = edge_xs * sx
ys_d_top = edge_y_tops * sy
ys_d_bot = edge_y_bots * sy

pts_top = batch_bilinear_sample(wp, xs_d, ys_d_top)
pts_bot = batch_bilinear_sample(wp, xs_d, ys_d_bot)

valid_3d = (np.isfinite(pts_top).all(1) & np.isfinite(pts_bot).all(1) &
            (np.linalg.norm(pts_top, axis=1) < 1e5) &
            (np.linalg.norm(pts_bot, axis=1) < 1e5))

pts_top = pts_top[valid_3d]
pts_bot = pts_bot[valid_3d]

print(f"\n3D-valid columns: {len(pts_top)}")

# Raw 3D distances (before scale)
raw_dists = np.linalg.norm(pts_top - pts_bot, axis=1)
print(f"Raw 3D edge-to-edge distances (VGGT units):")
print(f"  mean={raw_dists.mean():.6f}, median={np.median(raw_dists):.6f}, "
      f"std={raw_dists.std():.6f}")
print(f"  After scale ({scale:.2f}x): mean={raw_dists.mean()*scale:.4f}m, "
      f"median={np.median(raw_dists)*scale:.4f}m")

# --- Ground-plane projection ---
pts_top_proj = project_to_plane(pts_top.astype(np.float64), plane_n, plane_d) * scale
pts_bot_proj = project_to_plane(pts_bot.astype(np.float64), plane_n, plane_d) * scale

diffs = pts_top_proj - pts_bot_proj
widths_raw = np.linalg.norm(diffs, axis=1)

# PCA
all_proj = np.vstack([pts_top_proj, pts_bot_proj])
mean_p = all_proj.mean(0)
centered = all_proj - mean_p
centered = centered - (centered @ plane_n[:, None]) * plane_n[None, :]
_, _, vh = np.linalg.svd(centered, full_matrices=False)
along_dir = vh[0]; along_dir /= np.linalg.norm(along_dir) + 1e-12
across_dir = np.cross(plane_n, along_dir)
across_dir /= np.linalg.norm(across_dir) + 1e-12

widths_pca = np.abs(diffs @ across_dir)
ratio = np.median(widths_pca) / (np.median(widths_raw) + 1e-9)

print(f"\n--- Width computation ---")
print(f"PCA along_dir: {along_dir}")
print(f"PCA across_dir: {across_dir}")
print(f"widths_raw: median={np.median(widths_raw):.4f}m, mean={widths_raw.mean():.4f}m")
print(f"widths_pca: median={np.median(widths_pca):.4f}m, mean={widths_pca.mean():.4f}m")
print(f"PCA/raw ratio: {ratio:.4f}")
print(f"Using: {'PCA' if 0.5 < ratio < 1.05 else 'raw'}")

widths = widths_pca if 0.5 < ratio < 1.05 else widths_raw
reasonable = (widths > 0.05) & (widths < 15.0) & np.isfinite(widths)
widths = widths[reasonable]
med_w = np.median(widths)
mad_w = np.median(np.abs(widths - med_w)) * 1.4826 + 1e-6
inliers = np.abs(widths - med_w) < 2.5 * mad_w
filtered = widths[inliers]
final_width = float(np.median(filtered))

print(f"\nFinal width: {final_width:.4f}m")
print(f"  (after MAD filtering: {len(filtered)} of {len(widths)} columns)")

# --- What-if analysis ---
print("\n" + "="*60)
print("WHAT-IF ANALYSIS")
print("="*60)
# If we directly measure 3D Euclidean distance without plane projection
direct_3d = np.linalg.norm(pts_top - pts_bot, axis=1) * scale
print(f"\nDirect 3D distance (no plane projection): median={np.median(direct_3d):.4f}m")

# Check if segmentation is cutting the sidewalk short
print(f"\n--- Segmentation boundary check ---")
# For a few sample columns, print the full column profile
sample_cols = [int(edge_xs[len(edge_xs)//4]), int(edge_xs[len(edge_xs)//2]), 
               int(edge_xs[3*len(edge_xs)//4])]
for x in sample_cols:
    col = seg_map[:, x]
    sw_rows = np.where(col == ID_SW)[0]
    rd_rows = np.where(col == ID_RD)[0]
    print(f"  Col {x}: sidewalk rows [{sw_rows.min()}-{sw_rows.max()}] "
          f"({len(sw_rows)}px), road rows [{rd_rows.min() if len(rd_rows) else 'N/A'}-"
          f"{rd_rows.max() if len(rd_rows) else 'N/A'}] ({len(rd_rows)}px)")
    # What class is just above the sidewalk?
    top_edge = sw_rows.min()
    if top_edge > 0:
        above_class = int(col[top_edge - 1])
        print(f"    Class above sidewalk (row {top_edge-1}): {above_class} = {id2label.get(above_class, '?')}")
    # What class is just below the sidewalk?
    bot_edge = sw_rows.max()
    if bot_edge < H_img - 1:
        below_class = int(col[bot_edge + 1])
        print(f"    Class below sidewalk (row {bot_edge+1}): {below_class} = {id2label.get(below_class, '?')}")

# Check depth consistency along a sample column  
print(f"\n--- Depth profile along center column ---")
center_x = W_img // 2
center_x_d = int(center_x * sx)
med_top_y = int(np.median(edge_y_tops))
med_bot_y = int(np.median(edge_y_bots))
sample_ys = np.linspace(med_top_y, med_bot_y, 10)
for y_px in sample_ys:
    y_d = y_px * sy
    pt = batch_bilinear_sample(wp, np.array([float(center_x_d)]), np.array([y_d]))[0]
    pt_scaled = pt * scale
    print(f"  y={y_px:.0f}: raw_3d={pt}, scaled={pt_scaled}")

print(f"\n--- Scale sensitivity ---")
for test_h in [1.5, 2.0, 2.5, 3.0, 3.5]:
    s = test_h / h_pred
    w = np.median(raw_dists) * s
    print(f"  cam_height={test_h}m -> scale={s:.2f} -> width={w:.4f}m")
