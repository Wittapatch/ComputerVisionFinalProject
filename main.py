import cv2 as cv
import numpy as np
from contour_detection import find_hand_contour, draw_contour_on_frame, draw_centroid_on_frame, draw_convex_hull_on_frame, draw_defect_count_on_frame
from gesture_detection import draw_gesture_on_frame

def create_background_mask(roi, background_roi):
    # If no background has been saved yet, then we would return a fully black mask
    if background_roi is None:
        return np.zeros(roi.shape[:2], dtype=np.uint8)
    
    # Compare current ROI with the saved empty background
    # Where absdiff calculates the absolute difference for each pixel
    diff = cv.absdiff(roi, background_roi)
    
    # Convert the difference image from BGR color into grayscale
    # This makes the thresholding eaiser because each pixel ahs only one value (instead of 3 channels)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)


    # Apply Gaussian Blue to reduce tiny camera noise
    # Without this, small random pixel changes may appear as false hand pixels
    gray = cv.GaussianBlur(gray, (7, 7), 0)


    # Convert the grayscale difference image into a binary mask
    # If pixel difference > 23, make it white
    # If pixel difference <= 23, make it black
    _, mask = cv.threshold(gray, 23, 255, cv.THRESH_BINARY)
    
    # Create a 5x5 kernel for morphology operations
    # This kernel decide the neighborhood size used for cleaning the mask
    kernel = np.ones((5, 5), dtype=np.uint8)

    # Opening removes small white noise blobs
    # It does erosion first, then dilation
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # Closing fills small black gaps inside the white hand region
    # It does dilation first, then erosion
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask

def put_mask_preview(frame, mask, box, label):
    x, y, w, h = box

    preview_width = w // 3
    preview_height = h // 3

    small_mask = cv.resize(mask, (preview_width, preview_height))

    # Convert grayscale mask to BGR so we can paste it onto color frame
    small_mask_bgr = cv.cvtColor(small_mask, cv.COLOR_GRAY2BGR)

    preview_x = x + 10
    preview_y = y + h - preview_height - 10

    # Put the mask preview onto the main frame
    frame[preview_y:preview_y + preview_height, preview_x:preview_x + preview_width] = small_mask_bgr

    cv.rectangle(frame, (preview_x, preview_y), (preview_x + preview_width, preview_y+ preview_height), (255, 255, 255), 1)


cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Error: Camera cannot be opened")
    exit()

window_name = "Full Screen Camera"

# Making the camera take up the full screen
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# Storing the saved empty background for each player
# Where at the beginning, no background is saved yet
player1_background = None
player2_background = None

# This controls whether the program uses background subtraction
# background subtraction means that we will capture the empty background
# And use it to compare with the frame that hands by subtracting both images to get
# The outline of the hand
use_background_mode = False

while True:
    # Read one frame from the camera
    ret, frame = cam.read()

    if not ret:
        print("Error: Frame could not be read")
        break

    # Making it look like a mirror
    frame = cv.flip(frame, 1)

    # Keep clean camera frame for processing
    raw_frame = frame.copy()

    # Use this one only for drawing boxes, text, previews, contours
    display_frame = raw_frame.copy()

    # Get frame size
    height, width, channels = raw_frame.shape

    box_size = 250

    # Player 1 box on left
    player1_x = 50
    player1_y = 120

    # Player 2 box on right
    player2_x = width - box_size - player1_x
    player2_y = 120

    player1_box = (player1_x, player1_y, box_size, box_size)
    player2_box = (player2_x, player2_y, box_size, box_size)

    # Crop the Region of Interest inside each box
    player1_roi = raw_frame[player1_y: player1_y + box_size, player1_x:player1_x + box_size]
    player2_roi = raw_frame[player2_y: player2_y + box_size, player2_x:player2_x + box_size]

    if use_background_mode:
        player1_mask = create_background_mask(player1_roi, player1_background)
        player2_mask = create_background_mask(player2_roi, player2_background)
    else:
        # if background mode is not active yet, then create an empty black mask
        player1_mask = np.zeros(player1_roi.shape[:2], dtype=np.uint8)
        player2_mask = np.zeros(player2_roi.shape[:2], dtype=np.uint8)

    # Draw boxes for each player
    # Player 1 box
    cv.rectangle(display_frame, (player1_x, player1_y), (player1_x + box_size, player1_y + box_size), (255, 0, 0), 2)
    # Player 2 box
    cv.rectangle(display_frame, (player2_x, player2_y), (player2_x + box_size, player2_y + box_size), (0, 255, 0), 2)

    # Adding the labels for each player
    cv.putText(display_frame, "Player 1", (player1_x, player1_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.putText(display_frame, "Player 2", (player2_x, player2_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    # Find contours from masks
    player1_contour = find_hand_contour(player1_mask)
    player2_contour = find_hand_contour(player2_mask)

    # Draw contours on the main frame
    draw_contour_on_frame(display_frame, player1_contour, player1_box, (255, 0, 0), 2)
    draw_contour_on_frame(display_frame, player2_contour, player2_box, (0, 255, 0), 2)

    # Draw a centroid on the hand
    draw_centroid_on_frame(display_frame, player1_contour, player1_box, (0, 0, 255))
    draw_centroid_on_frame(display_frame, player2_contour, player2_box, (0, 0, 255))

    # Draw convex hulls around each hand
    draw_convex_hull_on_frame(display_frame, player1_contour, player1_box, (255, 255, 0), 2)
    draw_convex_hull_on_frame(display_frame, player2_contour, player2_box, (255, 255, 0), 2)

    # Show the number of finger gaps(convexity defects)
    draw_defect_count_on_frame(display_frame, player1_contour, player1_box, "Player1", (255, 0, 0))
    draw_defect_count_on_frame(display_frame, player2_contour, player2_box, "P2", (0, 255, 0))

    player1_gesture = draw_gesture_on_frame(display_frame, player1_contour, player1_box, "Player1", (255, 0, 0))
    player2_gesture = draw_gesture_on_frame(display_frame, player2_contour, player2_box, "Player2", (0, 255 ,0))

        # Show small mask previews inside each box
    put_mask_preview(display_frame, player1_mask, player1_box, "Player1 Mask")
    put_mask_preview(display_frame, player2_mask, player2_box, "Player2 Mask")

    cv.imshow(window_name, display_frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord("b"):
        # Save Player 1 and Player 2 empty background
        player1_background = player1_roi.copy()
        player2_background = player2_roi.copy()

        use_background_mode = True

    if key == 27:
        break

cam.release()
cv.destroyAllWindows()