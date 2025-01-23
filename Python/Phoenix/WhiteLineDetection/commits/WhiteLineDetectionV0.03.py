import cv2
import numpy as np
from scipy.cluster.vq import kmeans

# Path to your video file
video_path = '/home/derek/Downloads/rosbag2_2024_10_19-16_26_28_right.mp4'

# Function to check if a word is in a string
def search_string(string, word):
    return word in string

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    if search_string(video_path, "left"):
        print("Searching only for the left-side polynomial line.")
        side = "left"
    elif search_string(video_path, "right"):
        print("Searching only for the right-side polynomial line.")
        side = "right"
    else:
        print("Error: Neither 'left' nor 'right' specified in the video path.")
        side = None

    while cap.isOpened() and side:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for consistent processing size
        frame_resized = cv2.resize(frame, (640, 480))
        height, width = frame_resized.shape[:2]

        # Create mask to only keep the lower half of the image
        mask = np.zeros_like(frame_resized)
        mask[height // 2:, :] = 255
        masked_frame = cv2.bitwise_and(frame_resized, mask)

        # Convert to HSV and apply color threshold for white
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Process mask to remove noise
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.dilate(white_mask, kernel, iterations=3)
        white_mask = cv2.erode(white_mask, kernel, iterations=2)

        # Find contours and filter small ones to remove noise
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Adjust area threshold as needed
                cv2.drawContours(white_mask, [contour], -1, 0, -1)

        # Extract points from the mask where white is detected
        points = np.column_stack(np.where(white_mask > 0))

        # Check if there are enough points to fit a polynomial curve
        if len(points) > 0:
            x = points[:, 1]  # Width as x-coordinates
            y = points[:, 0]  # Height as y-coordinates

            # Fit a 3rd-degree polynomial to the points
            polynomial_coeffs = np.polyfit(x, y, 3)
            poly_vector = polynomial_coeffs  # Polynomial coefficients vector

            # Create a polynomial curve for visualization
            polynomial_curve = np.poly1d(polynomial_coeffs)
            x_range = np.linspace(min(x), max(x), num=500)
            y_range = polynomial_curve(x_range)

            # Draw the fitted polynomial curve on the frame
            fitted = frame_resized.copy()
            for (x_coord, y_coord) in zip(x_range, y_range):
                if 0 <= int(y_coord) < fitted.shape[0] and 0 <= int(x_coord) < fitted.shape[1]:
                    fitted = cv2.circle(fitted, (int(x_coord), int(y_coord)), radius=2, color=(255, 255, 0), thickness=-1)

            # Display the frames
            cv2.imshow("Lane Marking with Fitted Polynomial", fitted)
            cv2.imshow("White Mask", white_mask)

            # Handle keyboard input for quitting or pausing
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                break
            elif key == ord('p'):  # Press 'p' to pause or resume
                paused = not paused  # Toggle pause state

# Release resources
cap.release()
cv2.destroyAllWindows()
