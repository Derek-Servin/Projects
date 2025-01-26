import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import math
import scipy.interpolate as spi



##----------- This section of rotation is my attempt at adjusting the polynomial for a better fit on the white line----
##------------ when doing a left turn / curves that move left.

# Function to rotate a point (x, y) around a center (cx, cy) by a specific angle
def rotate_point(x, y, cx, cy, angle):
    # Convert angle to radians
    radians = math.radians(angle)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    # Translate the point to origin (0, 0), then rotate and translate back
    x -= cx
    y -= cy
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    return x_rot + cx, y_rot + cy


# Function to rotate a set of points (contours) around a center (cx, cy) by a given angle
def rotate_points(points, cx, cy, angle):
    return np.array([rotate_point(p[0], p[1], cx, cy, angle) for p in points])

### ------------------------------------------------------------------------------------------------------------------

# Function to open a file dialog and select a video file
def open_file_dialog(initial_dir, file_types=[("MP4 files", "*.mp4"), ("All files", "*.*")]):
    root = Tk()  # Initialize Tkinter
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Make sure the dialog is on top
    file_path = askopenfilename(initialdir=initial_dir, filetypes=file_types)  # Open file dialog
    return file_path


# Function to apply white thresholding in the HSV color space to extract bright areas (white)
def white_thresholding(hsv_frame):
    base_lower_white = np.array([0, 0, 200])  # Define lower bounds for white color
    base_upper_white = np.array([180, 30, 255])  # Define upper bounds for white color
    white_mask_lit = cv2.inRange(hsv_frame, base_lower_white, base_upper_white)  # Apply thresholding
    return white_mask_lit


# Function to apply shadow thresholding based on saturation and value (brightness) in the HSV color space
def shadow_thresholding(hsv_frame, dynamic_threshold=80, dynamic_tolerance=5):
    # Split HSV frame into individual channels: Hue, Saturation, and Value
    _, s, v = cv2.split(hsv_frame)

    # Condition for detecting shadows: Low saturation and higher brightness
    condition = (s < dynamic_threshold) & (v > dynamic_threshold)

    # Convert boolean condition to a mask (255 for shadows, 0 for others)
    shadow_mask = np.where(condition, 255, 0).astype(np.uint8)
    return shadow_mask


# A dummy callback function for trackbars (no operation)
def nothing(x):
    pass


# Main video processing function
def process_videos(video_paths):
    # Open video captures for all selected video paths
    caps = [cv2.VideoCapture(path) for path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open one or more videos.")
        return

    cv2.namedWindow("Output Stages", cv2.WINDOW_NORMAL)  # Create a window for displaying videos
    cv2.resizeWindow("Output Stages", 1920, 720)  # Resize window

    # Create trackbars (sliders) for user-controlled settings
    for i in range(1):
        cv2.createTrackbar("Area", "Output Stages", 58, 5000, nothing)  # Area threshold for white line detection
        cv2.createTrackbar("Rotation Angle", "Output Stages", 0, 360, nothing)  # Rotation angle slider
        cv2.createTrackbar("ROI", "Output Stages", 51, 100, nothing)  # Region of interest slider
        cv2.createTrackbar("Dynamic Threshold", "Output Stages", 91, 255, nothing)  # Threshold for shadow detection
        cv2.createTrackbar("Dynamic Tolerance", "Output Stages", 5, 50, nothing)  # Tolerance for shadow detection

    paused = False  # Flag for pausing/unpausing the video
    rewind_forward_step = 30  # Number of frames to jump for rewind/fast forward

    # Extract side information from the video name
    video_path = video_paths[0]
    video_name = os.path.basename(video_path).lower()
    side_of_interest = None
    if "left" in video_name:
        side_of_interest = "right"
    elif "right" in video_name:
        side_of_interest = "left"
    frame_num = 0  # Frame counter

    # Main video processing loop
    while True:
        if not paused:
            frame_num += 1  # Increment the frame number
            frames_with_poly = []  # List to store frames with polynomial curves

            # Loop through each video capture and process each frame
            for i, cap in enumerate(caps):
                ret, frame = cap.read()  # Read the next frame from the video
                if not ret:  # If no frame is read (end of video), reset to the beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                frame_resized = cv2.resize(frame, (640, 480))  # Resize frame for display
                height, width = frame_resized.shape[:2]

                # Get trackbar positions for dynamic control
                area_thresh = cv2.getTrackbarPos("Area", "Output Stages")  # Area threshold
                rotation_angle = cv2.getTrackbarPos("Rotation Angle", "Output Stages")  # Rotation angle
                roi = cv2.getTrackbarPos("ROI", "Output Stages")  # Region of interest
                dynamic_threshold = cv2.getTrackbarPos("Dynamic Threshold", "Output Stages")  # Shadow threshold
                dynamic_tolerance = cv2.getTrackbarPos("Dynamic Tolerance", "Output Stages")  # Shadow tolerance

                # Create a mask for the region of interest (ROI) to focus processing on specific part of the frame
                mask_start = int(height * (roi / 100.0))
                mask = np.zeros_like(frame_resized)
                mask[mask_start:, :] = 255
                masked_frame = cv2.bitwise_and(frame_resized, mask)

                # Convert the masked frame to HSV color space
                hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

                # Apply thresholding for white regions and shadows
                white_mask_lit = white_thresholding(hsv)
                shadow_mask = shadow_thresholding(hsv, dynamic_threshold, dynamic_tolerance)

                # Combine white and shadow masks
                combined_white_mask = cv2.bitwise_or(white_mask_lit, shadow_mask)

                # Apply morphological operations to clean up the mask
                kernel = np.ones((2, 2), np.uint8)  # Structuring element for dilation/erosion
                combined_white_mask = cv2.dilate(combined_white_mask, kernel, iterations=2)
                combined_white_mask = cv2.erode(combined_white_mask, kernel, iterations=2)

                # Find contours in the combined mask
                contours, _ = cv2.findContours(combined_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Remove small contours (noise) by checking their area
                for contour in contours:
                    if cv2.contourArea(contour) < area_thresh:
                        cv2.drawContours(combined_white_mask, [contour], -1, 0, -1)

                largest_contour = None
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)  # Find the largest contour

                    # Determine the side of the largest contour based on video side information
                    if side_of_interest == "left":
                        if np.mean(largest_contour[:, :, 0]) < width / 2:
                            cv2.drawContours(frame_resized, [largest_contour], -1, (255, 255, 0), 2)
                    elif side_of_interest == "right":
                        if np.mean(largest_contour[:, :, 0]) >= width / 2:
                            cv2.drawContours(frame_resized, [largest_contour], -1, (255, 255, 0), 2)

                # Create a copy of the frame with polynomial curve (if any)
                frame_with_poly = frame_resized.copy()
                if largest_contour is not None:
                    contour_points = np.squeeze(largest_contour)
                    if len(contour_points.shape) == 2 and contour_points.shape[0] > 10:
                        # Rotate the contour points based on the slider value
                        cx, cy = width // 2, height // 2
                        rotated_contour = rotate_points(contour_points, cx, cy, rotation_angle)

                        # Fit a polynomial curve to the rotated points
                        x_rotated = rotated_contour[:, 0]
                        y_rotated = rotated_contour[:, 1]
                        try:
                            # Fit polynomial of degree 3
                            polynomial_coeffs = np.polyfit(x_rotated, y_rotated, 3)
                            polynomial_curve = np.poly1d(polynomial_coeffs)

                            # Generate points for the polynomial curve
                            x_range_rotated = np.linspace(min(x_rotated), max(x_rotated), num=500)
                            y_range_rotated = polynomial_curve(x_range_rotated)

                            # Rotate back polynomial points to original coordinate system
                            original_poly_points = rotate_points(
                                np.column_stack((x_range_rotated, y_range_rotated)), cx, cy, -rotation_angle
                            )

                            # Draw the polynomial curve on the frame
                            for (x_val, y_val) in original_poly_points:
                                if 0 <= int(y_val) < height and 0 <= int(x_val) < width:
                                    cv2.circle(frame_with_poly, (int(x_val), int(y_val)), 2, (0, 0, 255), -1)

                        except np.linalg.LinAlgError:
                            pass  # Handle any exceptions during polynomial fitting

                frames_with_poly.append(frame_with_poly)

                # Combine all relevant frames for display
                combined_frame = np.hstack([
                    cv2.cvtColor(white_mask_lit, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(combined_white_mask, cv2.COLOR_GRAY2BGR),
                    frame_with_poly,
                ])

            # Show the final result in the display window
            cv2.imshow("Output Stages", combined_frame)
            cv2.imshow("Output", frame_with_poly)

        # Handle keyboard input for controlling video playback
        key = cv2.waitKey(25)

        if key == ord('q'):  # Quit the application
            break
        elif key == ord(' '):  # Pause/Resume video playback
            paused = not paused
        elif key == ord('o'):  # Open a new video file
            print("Opening file selector...")
            new_file_path = open_file_dialog(os.path.dirname(video_path))
            if new_file_path:
                print(f"Switching to new file: {new_file_path}")
                caps[0].release()  # Release current video capture
                caps = [cv2.VideoCapture(new_file_path)]
                video_name = os.path.basename(new_file_path).lower()
                side_of_interest = "right" if "left" in video_name else "left" if "right" in video_name else None
        elif key == 81:  # Left arrow key (rewind)
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - rewind_forward_step))
        elif key == 83:  # Right arrow key (fast forward)
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES,
                        min(caps[0].get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + rewind_forward_step))
        elif key == ord('j'):  # Jump backward
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 100))
        elif key == ord('l'):  # Jump forward
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES,
                        min(caps[0].get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + 100))

    # Release video captures and close all windows at the end
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


# Default directory path for video files
directory = "/home/derek/Downloads"
video_paths = []

# Open file dialog to select a video file
print("Select a video file")
video_path = open_file_dialog(directory)
if video_path:
    video_paths.append(video_path)
    process_videos(video_paths)
else:
    print("No video selected.")
