import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def open_file_dialog(initial_dir, file_types=[("MP4 files", "*.mp4"), ("All files", "*.*")]):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = askopenfilename(initialdir=initial_dir, filetypes=file_types)
    return file_path

def mask_green_to_black(hsv_frame, lower_green, upper_green):
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    result_frame = hsv_frame.copy()
    black_hsv = np.array([0, 0, 0])
    result_frame[green_mask == 255] = black_hsv
    return result_frame, green_mask

def fit_polynomial(points, order=3):
    if len(points) < order + 1:
        return None
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    coeffs = np.polyfit(x, y, order)  # Fit x as a function of y
    polynomial = np.poly1d(coeffs)
    return polynomial

def nothing(x):
    pass

def process_videos(video_paths):
    caps = [cv2.VideoCapture(path) for path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open one or more videos.")
        return

    cv2.namedWindow("Output Stages", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Output Stages", 1920, 720)

    cv2.createTrackbar("Lower H", "Output Stages", 20, 179, nothing)
    cv2.createTrackbar("Upper H", "Output Stages", 52, 179, nothing)
    cv2.createTrackbar("Lower S", "Output Stages", 24, 255, nothing)
    cv2.createTrackbar("Upper S", "Output Stages", 255, 255, nothing)
    cv2.createTrackbar("Lower V", "Output Stages", 45, 255, nothing)
    cv2.createTrackbar("Upper V", "Output Stages", 255, 255, nothing)

    paused = False

    while True:
        if not paused:
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                frame_resized = cv2.resize(frame, (640, 480))
                height, width = frame_resized.shape[:2]

                hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

                lower_h = cv2.getTrackbarPos("Lower H", "Output Stages")
                upper_h = cv2.getTrackbarPos("Upper H", "Output Stages")
                lower_s = cv2.getTrackbarPos("Lower S", "Output Stages")
                upper_s = cv2.getTrackbarPos("Upper S", "Output Stages")
                lower_v = cv2.getTrackbarPos("Lower V", "Output Stages")
                upper_v = cv2.getTrackbarPos("Upper V", "Output Stages")

                lower_green = np.array([lower_h, lower_s, lower_v])
                upper_green = np.array([upper_h, upper_s, upper_v])

                masked_frame, green_mask = mask_green_to_black(hsv_frame, lower_green, upper_green)

                result_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_HSV2BGR)

                mask_start = int(height * (48 / 100.0))
                mask = np.zeros_like(result_bgr)
                mask[mask_start:, :] = 255
                roi_frame = cv2.bitwise_and(result_bgr, mask)

                blurred_frame = cv2.GaussianBlur(roi_frame, (17, 17), 0)
                edges = cv2.Canny(blurred_frame, 50, 150)

                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Sort contours by arc length (longest first)
                contours_sorted = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

                # Select the two longest contours
                longest_contours = contours_sorted[:2]  # Get the top 2

                edges_largest_two = np.zeros_like(edges)  # Create a blank image

                for contour in longest_contours:
                    cv2.drawContours(edges_largest_two, [contour], -1, (255, 255, 255), 3)  # Draw longest two

                # ... (Rest of your polynomial fitting and drawing code, now using edges_largest_two)
                edge_points = np.argwhere(edges_largest_two > 0)
                edge_points = [(y, x) for y, x in edge_points]

                polynomial_frame = frame_resized.copy()

                left_points = []
                right_points = []

                # Separate points into left and right based on x-coordinate
                for y, x in edge_points:
                    if x < width // 2:  # Left side
                        left_points.append((y, x))
                    else:  # Right side
                        right_points.append((y, x))

                # Fit and draw left polynomial
                if left_points:
                    left_polynomial = fit_polynomial(left_points, order=3)
                    if left_polynomial:
                        min_y_left = min(p[0] for p in left_points)
                        max_y_left = max(p[0] for p in left_points)
                        y_vals_left = np.linspace(min_y_left, max_y_left, 500)
                        x_vals_left = left_polynomial(y_vals_left)

                        valid_indices_left = (x_vals_left >= 0) & (x_vals_left < frame_resized.shape[1]) & (
                                y_vals_left >= 0) & (y_vals_left < frame_resized.shape[0])
                        x_vals_left = x_vals_left[valid_indices_left]
                        y_vals_left = y_vals_left[valid_indices_left]

                        curve_points_left = np.array(list(zip(x_vals_left.astype(int), y_vals_left.astype(int))),
                                                     np.int32)
                        curve_points_left = curve_points_left.reshape((-1, 1, 2))

                        if curve_points_left.size > 0:
                            cv2.polylines(polynomial_frame, [curve_points_left], isClosed=False, color=(255, 255, 0),
                                          thickness=3)  # Cyan curve (left)

                # Fit and draw right polynomial
                if right_points:
                    right_polynomial = fit_polynomial(right_points, order=3)
                    if right_polynomial:
                        min_y_right = min(p[0] for p in right_points)
                        max_y_right = max(p[0] for p in right_points)
                        y_vals_right = np.linspace(min_y_right, max_y_right, 500)
                        x_vals_right = right_polynomial(y_vals_right)

                        valid_indices_right = (x_vals_right >= 0) & (x_vals_right < frame_resized.shape[1]) & (
                                y_vals_right >= 0) & (y_vals_right < frame_resized.shape[0])
                        x_vals_right = x_vals_right[valid_indices_right]
                        y_vals_right = y_vals_right[valid_indices_right]

                        curve_points_right = np.array(list(zip(x_vals_right.astype(int), y_vals_right.astype(int))),
                                                      np.int32)
                        curve_points_right = curve_points_right.reshape((-1, 1, 2))

                        if curve_points_right.size > 0:
                            cv2.polylines(polynomial_frame, [curve_points_right], isClosed=False, color=(255, 255, 0),
                                          thickness=3)  # Cyan curve (right)

                cv2.imshow("Original Frame", frame_resized)
                cv2.imshow("Masked Frame", roi_frame)
                cv2.imshow("Edges", edges)  # Original Edges
                cv2.imshow("Longest Two Edges", edges_largest_two)  # Longest Two Edges
                cv2.imshow("Polynomial Fit", polynomial_frame)
                cv2.imshow("Green Mask", green_mask)

        key = cv2.waitKey(25)

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('o'):
            print("Opening file selector...")
            new_file_path = open_file_dialog(os.path.dirname(video_path))
            if new_file_path:
                print(f"Switching to new file: {new_file_path}")
                caps[0].release()
                caps = [cv2.VideoCapture(new_file_path)]
                video_paths = os.path.basename(new_file_path).lower()

        elif key == 81:  # Left arrow key (rewind)
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 25))
        elif key == 83:  # Right arrow key (fast forward)
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, min(caps[0].get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + 25))
        elif key == ord('j'):  # Jump backward
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 100))
        elif key == ord('l'):  # Jump forward
            current_frame = caps[0].get(cv2.CAP_PROP_POS_FRAMES)
            caps[0].set(cv2.CAP_PROP_POS_FRAMES, min(caps[0].get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + 100))


    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

directory = "/home/derek/Downloads"  # Replace with your default directory
video_paths = []

print("Select a video file")
video_path = open_file_dialog(directory)
if video_path:
    video_paths.append(video_path)
    process_videos(video_paths)
else:
    print("No video selected.")
