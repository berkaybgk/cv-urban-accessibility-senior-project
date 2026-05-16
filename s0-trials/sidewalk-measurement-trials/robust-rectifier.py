import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression

class RobustHorizonRectifier:
    def __init__(self, image_height: int, safe_zone_px: int = 30, alpha: float = 0.3, max_patience: int = 3, angle_bin_deg: float = 2):
        self.H = image_height
        self.global_vy = self.H / 2.0  # Fallback optical center
        self.is_initialized = False
        self.patience_counter = 0
        
        self.safe_zone = safe_zone_px
        self.alpha = alpha
        self.max_patience = max_patience
        self.angle_bin_size = np.radians(angle_bin_deg)
        self._base_model = LinearRegression()

    def _fit_2_point_ransac(self, mask: np.ndarray):
        """Standard 2-point RANSAC to extract slope (a) and intercept (b)."""
        rows, cols = np.where(mask)
        if len(rows) < 2:
            return None, None
        
        ransac = RANSACRegressor(self._base_model, min_samples=2, residual_threshold=2.0, random_state=42)
        # Fit rows vs cols (x = a*y + b) because lines are mostly vertical
        ransac.fit(rows.reshape(-1, 1), cols)
        a = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        return a, b

    def process_tile(self, image: np.ndarray, good_mask: np.ndarray, bad_mask: np.ndarray, 
                     f_px: float, camera_height_m: float = 2.5, 
                     pixels_per_meter: float = 40.0, z_max_m: float = 30.0):
        
        # --- Step 1: Find the "Good Line" ---
        a_good, b_good = self._fit_2_point_ransac(good_mask)
        if a_good is None:
            return image.copy(), None 

        # --- Step 2: Probe the Local Horizon ---
        a_probe, b_probe = self._fit_2_point_ransac(bad_mask)
        local_vy = None
        if a_probe is not None and abs(a_good - a_probe) > 1e-6:
            local_vy = (b_probe - b_good) / (a_good - a_probe)

        # --- Step 3: The Gated Tracking Filter ---
        if local_vy is not None:
            if not self.is_initialized:
                if 0.1 * self.H < local_vy < 0.9 * self.H:
                    self.global_vy = local_vy
                    self.is_initialized = True
            else:
                diff = abs(local_vy - self.global_vy)
                if diff <= self.safe_zone:
                    self.global_vy = (1 - self.alpha) * self.global_vy + self.alpha * local_vy
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter > self.max_patience:
                        self.global_vy = local_vy
                        self.patience_counter = 0

        # --- Step 4: Lock the Vanishing Point ---
        vx = a_good * self.global_vy + b_good

        # --- Step 5: 1-Point RANSAC (The Histogram Sweep) ---
        bad_rows, bad_cols = np.where(bad_mask)
        if len(bad_rows) > 0:
            dy = bad_rows - self.global_vy
            dx = bad_cols - vx
            
            valid_idx = dy > 1e-3
            dy = dy[valid_idx]
            dx = dx[valid_idx]
            
            if len(dy) > 0:
                angles = np.arctan2(dx, dy) 
                bins = np.arange(angles.min(), angles.max() + self.angle_bin_size, self.angle_bin_size)
                if len(bins) > 1:
                    hist, bin_edges = np.histogram(angles, bins=bins)
                    peak_idx = np.argmax(hist)
                    best_angle = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2.0
                else:
                    best_angle = angles[0]
                
                a_bad = np.tan(best_angle)
                b_bad = vx - (a_bad * self.global_vy)
            else:
                a_bad, b_bad = a_probe, b_probe
        else:
            a_bad, b_bad = a_probe, b_probe

        # --- Step 6: Pinhole Unrolling (The Render) ---
        bottom_y = self.H - 1
        dy_bottom = max(1.0, float(bottom_y - self.global_vy)) 
        
        z_min = (f_px * camera_height_m) / dy_bottom
        
        if z_min >= z_max_m:
            return image.copy(), self.global_vy
            
        physical_length_m = z_max_m - z_min
        out_height = int(np.ceil(physical_length_m * pixels_per_meter))
        out_height = min(out_height, 3000) 
        
        z_out = np.linspace(z_min, z_max_m, out_height)
        
        src_y = self.global_vy + (f_px * camera_height_m) / z_out
        src_y = np.clip(src_y, 0, self.H - 1).astype(np.float32)

        line1_x = a_good * src_y + b_good
        line2_x = a_bad * src_y + b_bad
        
        left_bound = np.minimum(line1_x, line2_x)
        right_bound = np.maximum(line1_x, line2_x)
        
        target_width = int(np.max(right_bound - left_bound))
        target_width = max(target_width, 100) 
        
        map_y = np.repeat(src_y[:, None], target_width, axis=1)
        out_cols = np.arange(target_width, dtype=np.float32)
        
        src_width = right_bound - left_bound
        src_width[src_width < 1] = 1 
        
        scale = src_width / target_width
        map_x = left_bound[:, None] + out_cols[None, :] * scale[:, None]
        map_x = map_x.astype(np.float32)
        
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        
        # --- Visualization overlay for the debug plot ---
        debug_img = image.copy()
        cv2.circle(debug_img, (int(vx), int(self.global_vy)), 8, (0, 255, 255), -1) # Draw Horizon VP
        # Draw the locked lines
        y1, y2 = int(self.global_vy), self.H
        cv2.line(debug_img, (int(a_good * y1 + b_good), y1), (int(a_good * y2 + b_good), y2), (0, 255, 0), 2)
        cv2.line(debug_img, (int(a_bad * y1 + b_bad), y1), (int(a_bad * y2 + b_bad), y2), (0, 0, 255), 2)

        return warped, debug_img, self.global_vy

# ==========================================
# TEST HARNESS
# ==========================================
def create_synthetic_test_data():
    """Generates a fake perspective sidewalk to prove the math works."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (40, 40, 40) 

    vx, vy = 400, 250
    lx, rx = 200, 600
    by = 599

    # Draw sidewalk trapezoid
    pts = np.array([[vx, vy], [lx, by], [rx, by]], np.int32)
    cv2.fillPoly(img, [pts], (150, 150, 150))

    # Add horizontal lines to show perspective depth
    for y in range(vy + 50, 600, 50):
        scale = (y - vy) / (600 - vy)
        w = (rx - vx) * scale
        l, r = int(vx - w), int(vx + w)
        cv2.line(img, (l, y), (r, y), (255, 255, 255), 2)

    left_mask = np.zeros((600, 800), dtype=np.uint8)
    right_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.line(left_mask, (lx, by), (vx, vy), 255, 4)
    cv2.line(right_mask, (rx, by), (vx, vy), 255, 4)

    # Let's say right is the "good" mask, left is the "bad" mask
    return img, right_mask > 127, left_mask > 127

if __name__ == "__main__":
    # 1. Load Data
    # NOTE: To test your real data, replace this function call with your cv2.imread() and mask loading logic
    test_image, good_mask, bad_mask = create_synthetic_test_data()

    H, W = test_image.shape[:2]
    fov_degrees = 90
    f_px = W / (2.0 * np.tan(np.radians(fov_degrees / 2.0)))

    # 2. Initialize the tracking class (Done once before processing a sequence of images)
    rectifier = RobustHorizonRectifier(image_height=H)

    # 3. Process the image
    unrolled_image, debug_image, final_vy = rectifier.process_tile(
        image=test_image,
        good_mask=good_mask,
        bad_mask=bad_mask,
        f_px=f_px,
        camera_height_m=2.5,
        pixels_per_meter=40.0,
        z_max_m=20.0
    )

    # 4. Display Results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Perspective (VP Locked at Y={final_vy:.1f})")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(unrolled_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Top-Down Unrolled Output")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()