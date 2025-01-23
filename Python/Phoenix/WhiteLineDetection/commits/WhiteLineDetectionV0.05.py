import cv2
import numpy as np

# Path to your video file
video_path = 'Dataset/vid1.mp4'

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize parameters
hsv_params = {'lower_h': 0, 'lower_s': 0, 'lower_v': 200, 'upper_h': 180, 'upper_s': 255, 'upper_v': 255}
canny_params = {'min_val': 50, 'max_val': 150}
blur_param = 7  # Gaussian blur kernel size
paused = False  # Variable to keep track of pause state
rewind_forward_step = 30  # Number of frames to jump for rewind/forward

# Callback function for trackbars
def nothing(x):
    pass

# Create a window with trackbars integrated in the main window
cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

# Trackbars for HSV thresholds
cv2.createTrackbar("Lower H", "Lane Detection", hsv_params['lower_h'], 180, nothing)
cv2.createTrackbar("Lower S", "Lane Detection", hsv_params['lower_s'], 255, nothing)
cv2.createTrackbar("Lower V", "Lane Detection", hsv_params['lower_v'], 255, nothing)
cv2.createTrackbar("Upper H", "Lane Detection", hsv_params['upper_h'], 180, nothing)
cv2.createTrackbar("Upper S", "Lane Detection", hsv_params['upper_s'], 255, nothing)
cv2.createTrackbar("Upper V", "Lane Detection", hsv_params['upper_v'], 255, nothing)

# Trackbars for Canny and Gaussian blur
cv2.createTrackbar("Canny Min", "Lane Detection", canny_params['min_val'], 500, nothing)
cv2.createTrackbar("Canny Max", "Lane Detection", canny_params['max_val'], 500, nothing)
cv2.createTrackbar("Gaussian Blur", "Lane Detection", blur_param, 20, nothing)

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video frame.")
    cap.release()
    exit()

# Process video frames
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start if video ends
            continue

    # Resize frame for consistent processing size
    frame_resized = cv2.resize(frame, (640, 480))
    original_frame = frame_resized.copy()

    # Mask to isolate only the bottom half of the frame
    height, width = frame_resized.shape[:2]
    mask = np.zeros_like(frame_resized)
    mask[height // 2:, :] = 255  # Only bottom half of the image

    # Apply the mask to isolate the bottom half for processing
    bottom_half_frame = cv2.bitwise_and(frame_resized, mask)

    # Apply Gaussian blur
    blur_param = max(1, cv2.getTrackbarPos("Gaussian Blur", "Lane Detection") * 2 + 1)  # Ensure odd value for kernel size
    blurred_frame = cv2.GaussianBlur(bottom_half_frame, (blur_param, blur_param), 0)

    # Convert to HSV for color thresholding
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Get HSV thresholds from trackbars
    hsv_params['lower_h'] = cv2.getTrackbarPos("Lower H", "Lane Detection")
    hsv_params['lower_s'] = cv2.getTrackbarPos("Lower S", "Lane Detection")
    hsv_params['lower_v'] = cv2.getTrackbarPos("Lower V", "Lane Detection")
    hsv_params['upper_h'] = cv2.getTrackbarPos("Upper H", "Lane Detection")
    hsv_params['upper_s'] = cv2.getTrackbarPos("Upper S", "Lane Detection")
    hsv_params['upper_v'] = cv2.getTrackbarPos("Upper V", "Lane Detection")

    # White and yellow lane masks
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 80, 120])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine white and yellow masks for lane colors
    mask_hsv = cv2.bitwise_or(mask_white, mask_yellow)

    # Remove small noise by filtering contours
    contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Adjust area threshold as needed
            cv2.drawContours(mask_hsv, [contour], -1, 0, -1)

    # Apply Canny Edge Detection on the bottom half
    canny_params['min_val'] = cv2.getTrackbarPos("Canny Min", "Lane Detection")
    canny_params['max_val'] = cv2.getTrackbarPos("Canny Max", "Lane Detection")
    edges = cv2.Canny(blurred_frame, canny_params['min_val'], canny_params['max_val'])

    # Fusion of Color and Edge Detection (bitwise AND)
    fusion_output = cv2.bitwise_and(mask_hsv, edges)

    # Polynomial fitting for detected lane on the fusion output
    poly_frame = original_frame.copy()
    points_lane = np.column_stack(np.where(fusion_output > 0))

    # Only fit the polynomial if there are enough points
    if len(points_lane) > 10:  # Minimum number of points needed for stability
        x = points_lane[:, 1]
        y = points_lane[:, 0]

        try:
            # Fit a 3rd-degree polynomial
            polynomial_coeffs = np.polyfit(x, y, 3)
            polynomial_curve = np.poly1d(polynomial_coeffs)
            
            # Generate points for the polynomial curve
            x_range = np.linspace(min(x), max(x), num=500)
            y_range = polynomial_curve(x_range)
            
            # Draw the polynomial curve on the frame
            for (x_coord, y_coord) in zip(x_range, y_range):
                if 0 <= int(y_coord) < poly_frame.shape[0] and 0 <= int(x_coord) < poly_frame.shape[1]:
                    poly_frame = cv2.circle(poly_frame, (int(x_coord), int(y_coord)), radius=2, color=(255, 0, 0), thickness=-1)
        except np.linalg.LinAlgError as e:
            print("Polynomial fitting failed: ", e)
            # Skip drawing if fitting fails for this frame
    else:
        print("Not enough points for polynomial fitting")

    # Combine all five views into a single window
    combined_top = np.hstack((original_frame, cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)))
    combined_middle = np.hstack((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.cvtColor(fusion_output, cv2.COLOR_GRAY2BGR)))
    combined_bottom = cv2.resize(poly_frame, (combined_top.shape[1], combined_top.shape[0] // 2))
    combined_view = np.vstack((combined_top, combined_middle, combined_bottom))

    # Show the combined view
    cv2.imshow("Lane Detection", combined_view)

    # Handle keyboard input for play/pause, rewind, forward, and quitting
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Press 'p' to pause or play
        paused = not paused
    elif key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('r'):  # Press 'r' to rewind
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - rewind_forward_step))
    elif key == ord('f'):  # Press 'f' to fast forward
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, current_frame + rewind_forward_step))

# Release resources
cap.release()
cv2.destroyAllWindows()
