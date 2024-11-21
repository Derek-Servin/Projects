import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to open a file dialog for selecting a video file
def open_file_dialog(initial_dir, file_types=[("MP4 files", "*.mp4"), ("All files", "*.*")]):
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True)  # Make the file dialog appear on top
    file_path = askopenfilename(initialdir=initial_dir, filetypes=file_types)
    return file_path

# Function to apply dynamic thresholding
def dynamic_thresholding(hsv_frame, base_lower_white, base_upper_white, adjustment_factor):
    lower_white = np.array([base_lower_white[0], base_lower_white[1], base_lower_white[2] - adjustment_factor])
    upper_white = np.array([base_upper_white[0], base_upper_white[1], base_upper_white[2] + adjustment_factor])
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    return white_mask

# Function to process video with adjustable mask
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    adjustment_factor = 10  # Base brightness threshold adjustment
    paused = False  # Pause state

    # Create named windows
    cv2.namedWindow("Lane Marking with Fitted Polynomial", cv2.WINDOW_NORMAL)
    cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Lane Marking with Fitted Polynomial", 0, 0)
    cv2.moveWindow("White Mask", 650, 0)
    cv2.resizeWindow("Lane Marking with Fitted Polynomial", 640, 480)  # Width, Height
    cv2.resizeWindow("White Mask", 640, 480)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Restarting video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                continue

            # Resize frame for consistent processing size
            frame_resized = cv2.resize(frame, (640, 480))
            height, width = frame_resized.shape[:2]


            # Create mask
            mask_start = int(height * (58 / 100.0))
            mask = np.zeros_like(frame_resized)
            mask[mask_start:, :] = 255
            masked_frame = cv2.bitwise_and(frame_resized, mask)

            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

            # Apply dynamic thresholding
            base_lower_white = np.array([0, 0, 200])
            base_upper_white = np.array([180, 30, 255])
            white_mask = dynamic_thresholding(hsv, base_lower_white, base_upper_white, adjustment_factor)

            # Process mask to remove noise
            kernel = np.ones((5, 5), np.uint8)
            white_mask = cv2.dilate(white_mask, kernel, iterations=3)
            white_mask = cv2.erode(white_mask, kernel, iterations=2)

            # Find contours in the mask
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If contours are detected, draw them
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame_resized, [largest_contour], -1, (255, 255, 0), 2)
                for contour in contours:
                    if cv2.contourArea(contour) < 500:  # Adjust area threshold as needed
                        cv2.drawContours(white_mask, [contour], -1, 0, -1)

            # Display frames
            cv2.imshow("Lane Marking with Fitted Polynomial", frame_resized)
            cv2.imshow("White Mask", white_mask)

        # Handle keyboard input
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('p'):  # Pause/Resume
            paused = not paused
        elif key == ord('s'):  # Open file selector
            print("Opening file selector...")
            new_file_path = open_file_dialog(os.path.dirname(video_path))
            if new_file_path:
                print(f"Switching to new file: {new_file_path}")
                cap.release()
                cv2.destroyAllWindows()
                return new_file_path  # Switch to new video

        # Handle pausing logic
        if paused:
            key = cv2.waitKey(25) & 0xFF
            if key == ord('p'):  # Resume
                paused = not paused
            elif key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Open file selector
                print("Opening file selector...")
                new_file_path = open_file_dialog(os.path.dirname(video_path))
                if new_file_path:
                    print(f"Switching to new file: {new_file_path}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return new_file_path  # Switch to new video

    cap.release()
    cv2.destroyAllWindows()
    return None  # No new file selected

# Main loop
directory = "/home/derek/Downloads"
video_path = open_file_dialog(directory)  # Initial file selection
while video_path:
    print(f"Processing file: {video_path}")
    video_path = process_video(video_path)  # Process video and potentially switch

