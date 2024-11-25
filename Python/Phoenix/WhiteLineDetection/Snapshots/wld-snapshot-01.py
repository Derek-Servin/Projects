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


def white_thresholding(hsv_frame):
    base_lower_white = np.array([0, 0, 200])
    base_upper_white = np.array([180, 30, 255])
    white_mask_lit = cv2.inRange(hsv_frame, base_lower_white, base_upper_white)
    return white_mask_lit


def shadow_thresholding(hsv_frame, adjustment_factor, shadow_min_v, shadow_max_v):
    base_lower_white_shadow = np.array([0, 50, shadow_min_v])
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
    pass


def process_videos(video_paths):
    caps = [cv2.VideoCapture(path) for path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open one or more videos.")
        return

    cv2.namedWindow("Multiple Videos", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multiple Videos", 1920, 720)

    for i in range(4):
        cv2.createTrackbar(f"Adjust Factor {i + 1}", "Multiple Videos", 3, 50, nothing)
        cv2.createTrackbar(f"Shadow Min V {i + 1}", "Multiple Videos", 195, 255, nothing)
        cv2.createTrackbar(f"Shadow Max V {i + 1}", "Multiple Videos", 255, 255, nothing)

    paused = False
    rewind_forward_step = 30  # Number of frames to jump for rewind/forward

    while True:
        if not paused:
            frames_with_poly = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                frame_resized = cv2.resize(frame, (640, 480))
                height, width = frame_resized.shape[:2]

                adjustment_factor = cv2.getTrackbarPos(f"Adjust Factor {i + 1}", "Multiple Videos") # 3
                shadow_min_v = cv2.getTrackbarPos(f"Shadow Min V {i + 1}", "Multiple Videos") # 195
                shadow_max_v = cv2.getTrackbarPos(f"Shadow Max V {i + 1}", "Multiple Videos") # 255

                mask_start = int(height * (58 / 100.0))
                mask = np.zeros_like(frame_resized)
                mask[mask_start:, :] = 255
                masked_frame = cv2.bitwise_and(frame_resized, mask)

                hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

                white_mask_lit = white_thresholding(hsv)
                shadow_mask = shadow_thresholding(hsv, adjustment_factor, shadow_min_v, shadow_max_v)

                combined_white_mask = cv2.bitwise_or(white_mask_lit, shadow_mask)

                kernel = np.ones((5, 5), np.uint8)
                combined_white_mask = cv2.dilate(combined_white_mask, kernel, iterations=3)
                combined_white_mask = cv2.erode(combined_white_mask, kernel, iterations=2)

                contours, _ = cv2.findContours(combined_white_mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                largest_contour = None
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(frame_resized,
                                     [largest_contour], -1,
                                     (255, 255, 0), 2)

                frame_with_poly = frame_resized.copy()
                if largest_contour is not None:
                    contour_points = np.squeeze(largest_contour)
                    if len(contour_points.shape) == 2 and contour_points.shape[0] > 10:
                        x = contour_points[:, 0]
                        y = contour_points[:, 1]
                        try:
                            polynomial_coeffs = np.polyfit(x, y, 3)                 # Polynomimal degree
                            polynomial_curve = np.poly1d(polynomial_coeffs)

                            x_min, y_min, width_contour, height_contour = cv2.boundingRect(largest_contour)
                            x_max = x_min + width_contour
                            x_start = max(x_min, 0)
                            x_end = min(x_max, width)
                            x_range = np.linspace(x_start, x_end, num=500)
                            y_range = polynomial_curve(x_range)

                            for (x_val,
                                 y_val) in zip(x_range,
                                               y_range):
                                if (0 <= int(y_val) < height and
                                        0 <= int(x_val) < width):
                                    cv2.circle(frame_with_poly, (int(x_val),  int(y_val)),  2,  (0,    0,    255),   -1)
                        except np.linalg.LinAlgError:
                            pass

                frames_with_poly.append(frame_with_poly)

            combined_frame = np.hstack(frames_with_poly)

            cv2.imshow("Multiple Videos", combined_frame)

        # Handle keyboard input
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
                cap.release()
                cv2.destroyAllWindows()
                return new_file_path  # Switch to new video
        elif key == ord('r'):  # Press 'r' to rewind
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - rewind_forward_step))
        elif key == ord('f'):  # Press 'f' to fast forward
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    min(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + rewind_forward_step))

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


directory = "/home/derek/Downloads"
video_paths = []

for i in range(4):
    print(f"Select video file {i + 1}")
    video_path = open_file_dialog(directory)
    if video_path:
        video_paths.append(video_path)
    else:
        print("File selection canceled.")
        break

if len(video_paths) == 4:
    process_videos(video_paths)
else:
    print("Please select exactly 3 video files.")
