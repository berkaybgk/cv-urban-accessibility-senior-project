import cv2
import math
import argparse

def calculate_width(y_wall, y_curb, image_height, fov_deg, pitch_deg, cam_height):
    """
    Calculate physical sidewalk width given pixel boundaries and camera parameters.
    """
    # Convert angles to radians
    fov_rad = math.radians(fov_deg)
    pitch_rad = math.radians(pitch_deg)
    
    # 1. Convert FOV to Focal Length
    f_y = image_height / (2 * math.tan(fov_rad / 2))
    
    # Center y
    c_y = image_height / 2.0
    
    # 2. Pixel angle function
    def get_gamma(y):
        return math.atan((y - c_y) / f_y)
        
    # 3. Projection function
    def get_z(y):
        gamma_y = get_gamma(y)
        # Use absolute pitch so that a negative pitch (looking down) adds positively to the downward angle
        downward_pitch = abs(pitch_rad) 
        angle = downward_pitch + gamma_y
        if angle <= 0:
            return float('inf') # At or above the horizon
        return cam_height / math.tan(angle)
        
    z_wall = get_z(y_wall)
    z_curb = get_z(y_curb)
    
    width = z_wall - z_curb
    return width, z_wall, z_curb

def manual_selection(image_path):
    """
    Interactively select the upper and lower boundaries of the sidewalk.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    clone = img.copy()
    y_wall = -1
    y_curb = -1
    state = 0 # 0: waiting for wall, 1: waiting for curb, 2: done
    
    def on_mouse(event, x, y, flags, param):
        nonlocal y_wall, y_curb, state, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            if state == 0:
                y_wall = y
                cv2.line(clone, (0, y), (img.shape[1], y), (0, 255, 0), 2)
                cv2.putText(clone, "Upper Boundary", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"Selected upper boundary (wall/fence) at y={y}")
                state = 1
                print("Click to select the LOWER boundary (curb/road).")
            elif state == 1:
                if y <= y_wall:
                    print("Lower boundary must be below the upper boundary! Please click again.")
                else:
                    y_curb = y
                    cv2.line(clone, (0, y), (img.shape[1], y), (0, 0, 255), 2)
                    cv2.putText(clone, "Lower Boundary", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print(f"Selected lower boundary (curb/road) at y={y}")
                    state = 2
            cv2.imshow("Select Boundaries", clone)
            
    cv2.namedWindow("Select Boundaries", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Boundaries", on_mouse)
    
    print("Click to select the UPPER boundary (wall/fence).")
    while True:
        cv2.imshow("Select Boundaries", clone)
        key = cv2.waitKey(10) & 0xFF
        if key == 27: # Esc
            break
        elif state == 2:
            # Wait briefly to show the final line, then exit loop
            cv2.waitKey(500)
            break
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return y_wall, y_curb, img.shape[0]

def main():
    parser = argparse.ArgumentParser(description="Calculate sidewalk width via manual border selection.")
    parser.add_argument("--image", required=True, help="Path to the sidewalk image.")
    parser.add_argument("--fov", type=float, default=90.0, help="Vertical FOV in degrees.")
    parser.add_argument("--pitch", type=float, default=20.0, help="Camera downward pitch in degrees.")
    parser.add_argument("--cam_height", type=float, default=2.5, help="Camera height in meters.")
    
    args = parser.parse_args()
    
    y_wall, y_curb, img_height = manual_selection(args.image)
    
    if y_wall == -1 or y_curb == -1:
        print("Selection cancelled or incomplete.")
        return
        
    width, z_wall, z_curb = calculate_width(y_wall, y_curb, img_height, args.fov, args.pitch, args.cam_height)
    
    print(f"\n--- Results ---")
    print(f"Image Height: {img_height} px")
    print(f"Camera Height: {args.cam_height} m | Pitch: {args.pitch} deg | FOV: {args.fov} deg")
    
    if math.isinf(z_wall) or z_wall < 0:
        print(f"Warning: Upper boundary at y={y_wall} is at or above the horizon line. Cannot project to ground plane.")
    else:
        print(f"Upper Boundary (Wall): y={y_wall} -> Z = {z_wall:.2f} m")
        
    print(f"Lower Boundary (Curb): y={y_curb} -> Z = {z_curb:.2f} m")
    
    if not math.isinf(z_wall) and z_wall > 0:
        print(f"Estimated Width: {width:.2f} m")

if __name__ == '__main__':
    main()
