import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import math
import scipy.interpolate as spi

def rotate_point(x, y, cx, cy, angle):
    """
    Rotate a single point around a center point by a given angle.
    
    Args:
        x, y: Coordinates of point to rotate
        cx, cy: Center of rotation coordinates
        angle: Rotation angle in degrees
    
    Returns:
        Rotated x and y coordinates
    """
    radians = math.radians(angle)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    x -= cx
    y -= cy
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    return x_rot + cx, y_rot + cy

def rotate_points(points, cx, cy, angle):
    """
    Rotate multiple points around a center point.
    
    Args:
        points: Array of points to rotate
        cx, cy: Center of rotation coordinates
        angle: Rotation angle in degrees
    
    Returns:
        Array of rotated points
    """
    return np.array([rotate_point(p[0], p[1], cx, cy, angle) for p in points])

def open_file_dialog(initial_dir, file_types=[("MP4 files", "*.mp4"), ("All files", "*.*")]):
    """
    Open a file dialog to select a video file.
    
    Args:
        initial_dir: Starting directory for file selection
        file_types: Allowed file types
    
    Returns:
        Selected file path
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = askopenfilename(initialdir=initial_dir, filetypes=file_types)
    return file_path

def white_thresholding(hsv_frame):
    """
    Create a binary mask for white regions in HSV color space.
    
    Args:
        hsv_frame: Input frame converted to HSV color space
    
    Returns:
        Binary mask of white regions
    """
    base_lower_white = np.array([0, 0, 200])
    base_upper_white = np.array([180, 30, 255])
    white_mask_lit = cv2.inRange(hsv_frame, base_lower_white, base_upper_white)
    return white_mask_lit

def shadow_thresholding(hsv_frame, adjustment_factor, shadow_min_v, shadow_max_v):
    """
    Create a binary mask for white regions under varying lighting/shadow conditions.
    
    Args:
        hsv_frame: Input frame in HSV color space
        adjustment_factor: Flexibility in color thresholding
        shadow_min_v: Minimum value for shadow detection
        shadow_max_v: Maximum value for shadow detection
    
    Returns:
        Binary mask of white regions including shadows
    """
    base_lower_white_shadow = np.array([0, 64, shadow_min_v])
    base_upper_white_shadow = np.array([120, 180, shadow_max_v])

    lower_white = np.array([
        base_lower_white_shadow[0],
        max(0, base_lower_white_shadow[1] - adjustment_factor),
        max(0, base_lower_white_shadow[2] - adjustment_factor)
    ])
    upper_white = np.array([
        base_upper_white_shadow[0],
        min(255, base_upper_white_shadow[1] + adjustment_factor),
        min(255, base_upper_white_shadow[2] + adjustment_factor)
    ])

    white_mask_shadow = cv2.inRange(hsv_frame, lower_white, upper_white)
    return white_mask_shadow

def nothing(x):
    """
    Placeholder function for trackbar callbacks."""
    pass

def process_videos(video_paths):
    """
    Main processing function for white line detection in video.
    
    Key features:
    - Real-time video processing
    - Interactive trackbars for parameter tuning
    - White line detection with shadow handling
    - Polynomial curve fitting for line tracking
    - Side-specific contour detection
    
    Args:
        video_paths: List of video file paths to process
    """
    # Video capture setup
    caps = [cv2.VideoCapture(path) for path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open one or more videos.")
        return

    # Create main window with adjustable parameters
    cv2.namedWindow("Multiple Videos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multiple Videos", 1920, 720)

    # Create trackbars for real-time parameter adjustment
    for i in range(1):
        cv2.createTrackbar(f"Adjust Factor {i + 1}", "Multiple Videos", 3, 50, nothing)
        cv2.createTrackbar(f"Shadow Min V {i + 1}", "Multiple Videos", 177, 255, nothing)
        cv2.createTrackbar(f"Shadow Max V {i + 1}", "Multiple Videos", 255, 255, nothing)
        cv2.createTrackbar("Rotation Angle", "Multiple Videos", 0, 360, nothing)
        cv2.createTrackbar("ROI","Multiple Videos", 51,100,nothing)

    paused = False
    rewind_forward_step = 30  # Frames to jump during rewind/forward

    # Determine side of interest based on video filename
    video_path = video_paths[0]
    video_name = os.path.basename(video_path).lower()
    side_of_interest = None
    if "left" in video_name:
        side_of_interest = "right"
    elif "right" in video_name:
        side_of_interest = "left"
    
    frame_num = 0
    while True:
        if not paused:
            frame_num = frame_num+1
            frames_with_poly = []
            for i, cap in enumerate(caps):
                # Frame processing pipeline
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                # Resize and prepare frame
                frame_resized = cv2.resize(frame, (640, 480))
                cv2.circle(frame_resized, (100, 0), 5, (255, 0, 255), -1)
                height, width = frame_resized.shape[:2]

                # Get current trackbar values for dynamic parameter adjustment
                adjustment_factor = cv2.getTrackbarPos(f"Adjust Factor {i + 1}", "Multiple Videos")
                shadow_min_v = cv2.getTrackbarPos(f"Shadow Min V {i + 1}", "Multiple Videos")
                shadow_max_v = cv2.getTrackbarPos(f"Shadow Max V {i + 1}", "Multiple Videos")
                rotation_angle = cv2.getTrackbarPos("Rotation Angle", "Multiple Videos")
                roi = cv2.getTrackbarPos("ROI","Multiple Videos")

                # Region of Interest (ROI) masking
                mask_start = int(height * (roi / 100.0))
                mask = np.zeros_like(frame_resized)
                mask[mask_start:, :] = 255
                masked_frame = cv2.bitwise_and(frame_resized, mask)

                # Convert to HSV for better color thresholding
                hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

                # Create white detection masks
                white_mask_lit = white_thresholding(hsv)
                shadow_mask = shadow_thresholding(hsv, adjustment_factor, shadow_min_v, shadow_max_v)

                # Combine and clean up masks
                combined_white_mask = cv2.bitwise_or(white_mask_lit, shadow_mask)
                kernel = np.ones((2, 2), np.uint8)
                combined_white_mask = cv2.dilate(combined_white_mask, kernel, iterations=2)
                combined_white_mask = cv2.erode(combined_white_mask, kernel, iterations=2)

                # Contour detection and filtering
                contours, _ = cv2.findContours(combined_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Remove small noise contours
                for contour in contours:
                    if cv2.contourArea(contour) < 500:
                        cv2.drawContours(combined_white_mask, [contour], -1, 0, -1)

                largest_contour = None
                if contours:
                    # Select largest contour with side-specific filtering
                    largest_contour = max(contours, key=cv2.contourArea)
                    if side_of_interest == "left" and np.mean(largest_contour[:, :, 0]) < width / 2:
                        cv2.drawContours(frame_resized, [largest_contour], -1, (255, 255, 0), 2)
                    elif side_of_interest == "right" and np.mean(largest_contour[:, :, 0]) >= width / 2:
                        cv2.drawContours(frame_resized, [largest_contour], -1, (255, 255, 0), 2)
                    else:
                        cv2.drawContours(frame_resized, [largest_contour], -1, (255, 255, 0), 2)

                # Polynomial curve fitting for line tracking
                frame_with_poly = frame_resized.copy()
                if largest_contour is not None:
                    contour_points = np.squeeze(largest_contour)
                    if len(contour_points.shape) == 2 and contour_points.shape[0] > 10:
                        # Rotate and fit polynomial to contour points
                        cx, cy = width // 2, height // 2
                        rotated_contour = rotate_points(contour_points, cx, cy, rotation_angle)

                        x_rotated = rotated_contour[:, 0]
                        y_rotated = rotated_contour[:, 1]
                        try:
                            # 3rd-degree polynomial fitting
                            polynomial_coeffs = np.polyfit(x_rotated, y_rotated, 3)
                            print(frame_num)
                            print(polynomial_coeffs)
                            polynomial_curve = np.poly1d(polynomial_coeffs)

                            # Generate and draw polynomial curve
                            x_start, x_end = min(x_rotated), max(x_rotated)
                            x_range_rotated = np.linspace(x_start, x_end, num=500)
                            y_range_rotated = polynomial_curve(x_range_rotated)

                            original_poly_points = rotate_points(
                                np.column_stack((x_range_rotated, y_range_rotated)), cx, cy, -rotation_angle
                            )

                            # Draw curve points
                            for (x_val, y_val) in original_poly_points:
                                if 0 <= int(y_val) < height and 0 <= int(x_val) < width:
                                    cv2.circle(frame_with_poly, (int(x_val), int(y_val)), 2, (0, 0, 255), -1)

                        except np.linalg.LinAlgError:
                            pass

                frames_with_poly.append(frame_with_poly)

                # Combine and display processed frames
                combined_frame = np.hstack([
                    cv2.cvtColor(white_mask_lit, cv2.COLOR_GRAY2BGR), 
                    cv2.cvtColor(combined_white_mask, cv2.COLOR_GRAY2BGR),
                    frame_with_poly
                ])

            # Show processing results
            cv2.imshow("Multiple Videos", combined_frame)

        # User interaction controls
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/Resume
            paused = not paused
        elif key == ord('o'):  # Open file selector
            print("Opening file selector...")
            new_file_path = open_file_dialog(os.path.dirname(video_path))
            if new_file_path:
                print(f"Switching to new file: {new_file_path}")
                caps[0].release()  # Release current video capture
                caps = [cv2.VideoCapture(new_file_path)]
                video_name = os.path.basename(new_file_path).lower()
                side_of_interest = "left" if "left" in video_name else "right" if "right" in video_name else None
        elif key == ord('r'):  # Rewind
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - rewind_forward_step))
        elif key == ord('f'):  # Fast forward
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES,
                        min(caps[0].get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + rewind_forward_step))

    # Cleanup
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

# Main execution
directory = "/home/derek/Downloads"
video_paths = []

# Select and process video
print("Select a video file")
video_path = open_file_dialog(directory)
if video_path:
    video_paths.append(video_path)
    process_videos(video_paths)
else:
    print("File selection canceled.")
