
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
def white_thresholding(hsv_frame, dynamic_threshold):
    # Split HSV frame into individual channels: Hue, Saturation, and Value
    _, s, v = cv2.split(hsv_frame)

    # Condition for detecting shadows: Low saturation and higher brightness
    condition = (s < dynamic_threshold) & (v > dynamic_threshold)

    # Convert boolean condition to a mask (255 for shadows, 0 for others)
    white_mask_lit = np.where(condition, 255, 0).astype(np.uint8)

    return white_mask_lit



# Function to apply shadow thresholding based on saturation and value (brightness) in the HSV color space
# Function to apply shadow thresholding based on saturation and value (brightness) in the HSV color space
def shadow_thresholding(hsv_frame, sh_dyn_thresh):
    # Split HSV frame into individual channels: Hue, Saturation, and Value
    _, s, v = cv2.split(hsv_frame)

    # Condition for detecting shadows: Low saturation and higher brightness
    condition = (s < sh_dyn_thresh) & (v > sh_dyn_thresh)

    # Convert boolean condition to a mask (255 for shadows, 0 for others)
    shadow_mask = np.where(condition, 255, 0).astype(np.uint8)

    # Define the base lower and upper bounds for shadow color detection
    base_lower_gray = np.array([0, 0, 100])  # Define lower bounds for shadow color
    upper_gray = np.array([240, 30, sh_dyn_thresh])  # Define upper bounds for shadow color

    # Apply thresholding to detect shadow regions
    shadow_mask = cv2.inRange(hsv_frame, base_lower_gray, upper_gray)

    return shadow_mask



# A dummy callback function for trackbars (no operation)
def nothing(x):
    pass


# Function to analyze the histogram and adjust dynamic threshold if necessary
def analyze_and_adjust_histogram(white_mask, shadow_mask, dynamic_threshold, sh_dyn_thresh, adjust_threshold_up=True, white_threshold=0.02, black_threshold=0.98):
    # Calculate the histogram of the mask for white and shadow detection
    hist_white = cv2.calcHist([white_mask], [0], None, [256], [0, 256])
    hist_shadow = cv2.calcHist([shadow_mask], [0], None, [256], [0, 256])

    # Normalize the histograms
    hist_white /= hist_white.sum()
    hist_shadow /= hist_shadow.sum()

    # Calculate the percentage of white (255) and black (0) pixels for white mask
    white_percentage = hist_white[255][0]
    black_percentage = hist_white[0][0]

    # Calculate the percentage of shadow (255) and non-shadow (0) pixels for shadow mask
    shadow_percentage = hist_shadow[255][0]
    non_shadow_percentage = hist_shadow[0][0]

    print(f"White percentage: {white_percentage:.2f}, Black percentage: {black_percentage:.2f}")
    print(f"Shadow percentage: {shadow_percentage:.2f}, Non-shadow percentage: {non_shadow_percentage:.2f}")

    # Adjust dynamic threshold based on histogram analysis for white mask
    if white_percentage > white_threshold:
        print("Too much white, lowering dynamic threshold...")
        new_threshold = max(0, dynamic_threshold + 3)  # Decrease threshold
        cv2.setTrackbarPos("Dynamic Threshold", "Output Stages", new_threshold)

    elif black_percentage > black_threshold:
        print("Too much black, raising dynamic threshold...")
        new_threshold = min(255, dynamic_threshold - 3)  # Increase threshold
        cv2.setTrackbarPos("Dynamic Threshold", "Output Stages", new_threshold)

    # Adjust shadow dynamic threshold based on histogram analysis for shadow mask
    if shadow_percentage < 0.005:  # Example threshold, you can adjust this
        print("Too much shadow, lowering shadow dynamic threshold...")
        new_sh_dyn_thresh = max(10, sh_dyn_thresh + 3)  # Decrease shadow threshold
        cv2.setTrackbarPos("Shadow Dynamic Threshold", "Output Stages", new_sh_dyn_thresh)

    elif non_shadow_percentage < 0.995:  # Example threshold, adjust as needed
        print("Too much non-shadow, raising shadow dynamic threshold...")
        new_sh_dyn_thresh = min(200, sh_dyn_thresh - 3)  # Increase shadow threshold
        cv2.setTrackbarPos("Shadow Dynamic Threshold", "Output Stages", new_sh_dyn_thresh)



# Function to calculate the centroid of a contour
def calculate_centroid(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None  # To avoid division by zero
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)

# Function to calculate the distance from the center of the image
def calculate_distance_to_center(contour, image_center):
    centroid = calculate_centroid(contour)
    if centroid is None:
        return float('inf')  # If the contour has no valid centroid, return a large value
    cx, cy = centroid
    distance = math.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
    return distance

# Function to check if a contour's centroid is close to the center of the image
def is_centroid_close_to_center(contour, dynamic_threshold, area_thresh, threshold=40, center=(320, 260)):
    centroid = calculate_centroid(contour)
    if centroid:
        cx, cy = centroid
        distance = calculate_distance_to_center(contour, center)
        return distance < threshold
    return False

# Function to find the left and right closest contours based on the image center
def find_left_right_contours(contours, image_center):
    # Sort contours by their distance to the image center
    contours_sorted_by_distance = sorted(contours,
                                         key=lambda contour: calculate_distance_to_center(contour, image_center))

    left_contour = None
    right_contour = None

    for contour in contours_sorted_by_distance:
        centroid = calculate_centroid(contour)
        if centroid:
            cx, _ = centroid
            if cx < image_center[0] and left_contour is None:  # Left of the image center
                left_contour = contour
            elif cx > image_center[0] and right_contour is None:  # Right of the image center
                right_contour = contour

        # We now explicitly check if both left_contour and right_contour are not None
        if left_contour is not None and right_contour is not None:
            break  # We've found both left and right contours

    return left_contour, right_contour


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
        cv2.createTrackbar("Area", "Output Stages", 58, 2000, nothing)  # Area threshold for white line detection
        cv2.createTrackbar("Rotation Angle", "Output Stages", 90, 360, nothing)  # Rotation angle slider
        cv2.createTrackbar("ROI", "Output Stages", 51, 100, nothing)  # Region of interest slider
        cv2.createTrackbar("Dynamic Threshold", "Output Stages", 91, 255, nothing)  # Threshold for white detection
        cv2.createTrackbar("Shadow Dynamic Threshold", "Output Stages", 80, 255, nothing)  # New shadow threshold slider

    paused = False  # Flag for pausing/unpausing the video
    rewind_forward_step = 30  # Number of frames to jump for rewind/fast forward

    frame_num = 0  # Frame counter
    last_dynamic_threshold = 0  # Initialize with the default dynamic threshold value from trackbar
    frames_since_threshold_change = 0  # Counter for frames since the threshold last changed

    # Main video processing loop
    while True:
        if not paused:
            frame_num += 1  # Increment the frame number
            # Loop through each video capture and process each frame
            for i, cap in enumerate(caps):
                ret, frame = cap.read()  # Read the next frame from the video
                if not ret:  # If no frame is read (end of video), reset to the beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                frame_resized = cv2.resize(frame, (640, 480))  # Resize frame for display
                height, width = frame_resized.shape[:2]
                frame_with_poly = frame_resized.copy()

                # Get trackbar positions for dynamic control
                area_thresh = cv2.getTrackbarPos("Area", "Output Stages")  # Area threshold
                rotation_angle = cv2.getTrackbarPos("Rotation Angle", "Output Stages")  # Rotation angle
                roi = cv2.getTrackbarPos("ROI", "Output Stages")  # Region of interest
                dynamic_threshold = cv2.getTrackbarPos("Dynamic Threshold", "Output Stages")  # White threshold
                sh_dyn_thresh = cv2.getTrackbarPos("Shadow Dynamic Threshold", "Output Stages")  # Shadow threshold


                # If the dynamic threshold hasn't changed for 180 frames, divide it by 2
                #if dynamic_threshold == last_dynamic_threshold:
                #    frames_since_threshold_change += 1
                #else:
                #    frames_since_threshold_change = 0  # Reset counter if the threshold changes

                #if frames_since_threshold_change >= 9000:
                #    print("1 sec with no threshold change, dividing dynamic threshold by 2.")
                #    dynamic_threshold = max(0, dynamic_threshold // 2)
                #    cv2.setTrackbarPos("Dynamic Threshold", "Output Stages", dynamic_threshold)
                #    frames_since_threshold_change = 0  # Reset the counter after adjustment

                #last_dynamic_threshold = dynamic_threshold  # Update last threshold value for future comparisons

                # Create a mask for the region of interest (ROI) to focus processing on specific part of the frame
                mask_start = int(height * (roi / 100.0))
                mask = np.zeros_like(frame_resized)
                mask[mask_start:, :] = 255
                masked_frame = cv2.bitwise_and(frame_resized, mask)

                # Convert the masked frame to HSV color space
                hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)



                # Apply thresholding for white regions and shadows
                white_mask_lit = white_thresholding(hsv,dynamic_threshold)
                shadow_mask = shadow_thresholding(hsv, sh_dyn_thresh)

                # Apply morphological operations to clean up the mask
                sh_kernel = np.ones((3, 3), np.uint8)  # Structuring element for dilation/erosion

                shadow_mask = cv2.dilate(shadow_mask, sh_kernel, iterations=2)
                shadow_mask = cv2.erode(shadow_mask, sh_kernel, iterations=2)


                # Combine white and shadow masks
                combined_white_mask = cv2.bitwise_or(white_mask_lit, shadow_mask)

                kernel = np.ones((2, 2), np.uint8)  # Structuring element for dilation/erosion
                combined_white_mask = cv2.dilate(combined_white_mask, kernel, iterations=2)
                combined_white_mask = cv2.erode(combined_white_mask, kernel, iterations=2)

                # Find contours in the combined mask
                contours, _ = cv2.findContours(combined_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Remove small contours (noise) by checking their area
                for contour in contours:
                    if cv2.contourArea(contour) < area_thresh:
                        cv2.drawContours(combined_white_mask, [contour], -1, 0, -1)

                contours_wh, _ = cv2.findContours(white_mask_lit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if is_centroid_close_to_center(contour, area_thresh, dynamic_threshold):
                        print("White contour close to center, switching to black or ignoring.")
                        cv2.drawContours(white_mask_lit, [contour], -1, 0, -1)  # Ignore it or switch to black
                        # Combine white and shadow masks
                        combined_white_mask = cv2.bitwise_or(white_mask_lit, shadow_mask)

                        kernel = np.ones((2, 2), np.uint8)  # Structuring element for dilation/erosion
                        combined_white_mask = cv2.dilate(combined_white_mask, kernel, iterations=2)
                        combined_white_mask = cv2.erode(combined_white_mask, kernel, iterations=2)

                # Assuming contours_wh is already defined
                contours_combined = contours + contours_wh  # Combine both lists of contours

                # Sort the combined contours by area
                contours_sorted = sorted(contours_combined, key=cv2.contourArea, reverse=True)

                # Get the image center based on the kart (not the actual image center)
                image_center = (320, 360)  # Assuming this is the correct center for your use case

                # # Apply morphological operations to clean up the mask
                # kernel = np.ones((2, 2), np.uint8)  # Structuring element for dilation/erosion
                # combined_white_mask = cv2.dilate(combined_white_mask, kernel, iterations=2)
                # combined_white_mask = cv2.erode(combined_white_mask, kernel, iterations=2)

                # Analyze and adjust dynamic threshold based on histogram
                # Call the histogram analysis function to adjust both white and shadow thresholds
                analyze_and_adjust_histogram(white_mask_lit, shadow_mask, dynamic_threshold, sh_dyn_thresh)
                if dynamic_threshold == 0:
                    dynamic_threshold = 190
                elif dynamic_threshold > 225:
                    dynamic_threshold = 224
                analyze_and_adjust_histogram(white_mask_lit, shadow_mask, dynamic_threshold, sh_dyn_thresh)

                # Get the 4 largest contours
                top_4_contours = contours_sorted[:4]

                # Find the closest contour to the left and right of the image center
                left_contour, right_contour = find_left_right_contours(top_4_contours, image_center)

                # Now we can fit polynomials to these contours
                contours_to_process = []
                if left_contour is not None:
                    contours_to_process.append(left_contour)
                if right_contour is not None:
                    contours_to_process.append(right_contour)

                for contour in contours_to_process:
                    if is_centroid_close_to_center(contour,area_thresh, dynamic_threshold):
                        print("White contour close to center, switching to black or ignoring.")
                        cv2.drawContours(white_mask_lit, [contour], -1, 0, -1)  # Ignore it or switch to black
                    else:
                        contour_points = np.squeeze(contour)
                        if len(contour_points.shape) == 2 and contour_points.shape[0] > 10:
                            cx, cy = width // 2, height // 2
                            rotated_contour = rotate_points(contour_points, cx, cy, rotation_angle)

                            x_rotated = rotated_contour[:, 0]
                            y_rotated = rotated_contour[:, 1]
                            try:
                                polynomial_coeffs = np.polyfit(x_rotated, y_rotated, 3)
                                polynomial_curve = np.poly1d(polynomial_coeffs)

                                x_range_rotated = np.linspace(min(x_rotated), max(x_rotated), num=500)
                                y_range_rotated = polynomial_curve(x_range_rotated)

                                original_poly_points = rotate_points(
                                    np.column_stack((x_range_rotated, y_range_rotated)), cx, cy, -rotation_angle
                                )

                                for (x_val, y_val) in original_poly_points:
                                    if 0 <= int(y_val) < height and 0 <= int(x_val) < width:
                                        cv2.circle(frame_with_poly, (int(x_val), int(y_val)), 2, (0, 0, 255), -1)

                            except np.linalg.LinAlgError:
                                pass


                # Append the frame with polynomials to the list
                #frames_with_poly.append(frame_with_poly)

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
                video_paths = os.path.basename(new_file_path).lower()

        elif key == 81:  # Left arrow key (rewind)
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 25))
        elif key == 83:  # Right arrow key (fast forward)
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES,
                        min(caps[0].get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + 25))
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
