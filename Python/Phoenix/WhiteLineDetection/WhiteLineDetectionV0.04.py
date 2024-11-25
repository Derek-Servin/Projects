import cv2
import numpy as np

# Path to your video file
video_path = '/home/derek/Downloads/rosbag2_2024_10_19-16_26_28_right.mp4'
# video_path = '/home/derek/Downloads/rosbag2_2024_10_19-16_53_40_left.mp4'

# Function to check if a word is in a string
def search_string(string, word):
    return word in string

paused = False

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

    # Create a window for the trackbars
    cv2.namedWindow("Settings")

    # Initial threshold values for white detection in HSV
    lower_white_hue = 0
    lower_white_saturation = 0
    lower_white_value = 190
    upper_white_hue = 180
    upper_white_saturation = 15
    upper_white_value = 222

    # Function to update lower and upper thresholds from trackbars
    def nothing(x):
        pass

    # Create trackbars for adjusting the thresholds
    cv2.createTrackbar('Lower Hue', 'Settings', lower_white_hue, 179, nothing)
    cv2.createTrackbar('Lower Sat', 'Settings', lower_white_saturation, 255, nothing)
    cv2.createTrackbar('Lower Val', 'Settings', lower_white_value, 255, nothing)
    cv2.createTrackbar('Upper Hue', 'Settings', upper_white_hue, 179, nothing)
    cv2.createTrackbar('Upper Sat', 'Settings', upper_white_saturation, 255, nothing)
    cv2.createTrackbar('Upper Val', 'Settings', upper_white_value, 255, nothing)

    while cap.isOpened() and side:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

        # Resize frame for consistent processing size
        frame_resized = cv2.resize(frame, (640, 480))
        height, width = frame_resized.shape[:2]



        # Create mask based on the side of the video (left or right)
        mask = np.zeros_like(frame_resized)

        # If we are processing the left side, keep the left half
        if side == "left":
            # Create mask to only keep the lower half of the image
            mask = np.zeros_like(frame_resized)
            mask[height // 2:, :] = 255
            masked_frame = cv2.bitwise_and(frame_resized, mask)
        elif side == "right":
            # Create mask to only keep the lower half of the image
            mask = np.zeros_like(frame_resized)
            mask[height // 2:, :] = 255
            masked_frame = cv2.bitwise_and(frame_resized, mask)

        # Apply the mask to the frame
        masked_frame = cv2.bitwise_and(frame_resized, mask)

        # Get trackbar positions for thresholds
        lower_white_hue = cv2.getTrackbarPos('Lower Hue', 'Settings')
        lower_white_saturation = cv2.getTrackbarPos('Lower Sat', 'Settings')
        lower_white_value = cv2.getTrackbarPos('Lower Val', 'Settings')
        upper_white_hue = cv2.getTrackbarPos('Upper Hue', 'Settings')
        upper_white_saturation = cv2.getTrackbarPos('Upper Sat', 'Settings')
        upper_white_value = cv2.getTrackbarPos('Upper Val', 'Settings')

        # Convert to HSV and apply color threshold for white
        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([lower_white_hue, lower_white_saturation, lower_white_value])
        upper_white = np.array([upper_white_hue, upper_white_saturation, upper_white_value])
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
