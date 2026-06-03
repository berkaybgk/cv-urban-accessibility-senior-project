import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

img_tensor = load_and_preprocess_images(["side_image.jpg"]).to(device)
with torch.inference_mode():
    with torch.amp.autocast("mps", dtype=torch.float16) if device == "mps" else torch.amp.autocast("cuda") if device == "cuda" else torch.autocast("cpu"):
        pred = model(img_tensor)

import math

pe = pred["pose_enc"][0].detach().cpu().numpy()
if pe.ndim == 2: pe = pe[0]
print("pose_enc:", pe)
qw, qx, qy, qz = pe[3:7]

# Roll (x-axis rotation)
sinr_cosp = 2 * (qw * qx + qy * qz)
cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
roll = math.atan2(sinr_cosp, cosr_cosp)

# Pitch (y-axis rotation)
sinp = 2 * (qw * qy - qz * qx)
if abs(sinp) >= 1:
    pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
else:
    pitch = math.asin(sinp)

# Yaw (z-axis rotation)
siny_cosp = 2 * (qw * qz + qx * qy)
cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
yaw = math.atan2(siny_cosp, cosy_cosp)

print(f"Euler angles (rad): roll={roll}, pitch={pitch}, yaw={yaw}")
print(f"Euler angles (deg): roll={math.degrees(roll)}, pitch={math.degrees(pitch)}, yaw={math.degrees(yaw)}")
