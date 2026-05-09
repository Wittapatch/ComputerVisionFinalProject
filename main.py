import cv2 as cv
import numpy as np
from contour_detection import find_hand_contour, draw_contour_on_frame, draw_centroid_on_frame, draw_convex_hull_on_frame, draw_defect_count_on_frame
from gesture_detection import draw_gesture_on_frame
from game_logic import decide_winner
import time
from face_detection import detect_faces, get_largest_face, draw_face_box, crop_face
from photo_editor import open_photo_editor
from winner_screen import show_winner_screen

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

def both_players_ready(player1_gesture, player2_gesture):

    return player1_gesture != "unknown" and player2_gesture != "unknown"


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

# Variables for Game Logic

game_started = False

countdown_started = False
countdown_start_time = 0
countdown_seconds = 3

result_locked = False
locked_player1_gesture = "unknown"
locked_player2_gesture = "unknown"
locked_winner_text = "Waiting"

# Variables for face detection
current_face_box = None
winner_photo_path = None



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
    player1_roi = raw_frame[player1_y: player1_y + box_size, player1_x:player1_x + box_size].copy()
    player2_roi = raw_frame[player2_y: player2_y + box_size, player2_x:player2_x + box_size].copy()

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


    if not game_started:
        cv.putText(display_frame, "B: Get background, S: Start Game, R: Restart Game", (40, height-40), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        winner_text = "Waiting"
    # Count down block
    # If the result is not locked yet, keep checking the current gestures
    else:
        if not result_locked:
            # Start the countdown only when both playes have valid gestures
            if both_players_ready(player1_gesture, player2_gesture):

                # Start countdown once
                if not countdown_started:
                    countdown_started = True
                    countdown_start_time = time.time()

                # Calculate how much time has passed
                elapsed_time = time.time() - countdown_start_time

                # Calculate remaining countdown number 
                remaining_time = countdown_seconds - int(elapsed_time)

                if remaining_time > 0:
                    cv.putText(display_frame, f"Show gesture in: {remaining_time}", (width//2 - 180, 110), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                    winner_text = "Get Ready"
                else:
                    locked_player1_gesture = player1_gesture
                    locked_player2_gesture = player2_gesture 
                    
                    locked_winner_text = decide_winner(locked_player1_gesture, locked_player2_gesture)

                    result_locked = True
                    countdown_started = False
                    winner_text = locked_winner_text
            else:
                countdown_started = False
                winner_text = "Waiting for both hands"
        else:
            winner_text = locked_winner_text
            cv.putText(display_frame, f"Locked P1: {locked_player1_gesture}", (player1_x, player1_y + box_size + 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv.putText(display_frame, f"Locked P2: {locked_player2_gesture}", (player2_x, player2_y + box_size + 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    face_photo_mode = result_locked and locked_winner_text in ["Player 1 Wins", "Player 2 Wins"]
    
    if face_photo_mode:
        display_frame = raw_frame.copy()
        faces = detect_faces(raw_frame)
        current_face_box = get_largest_face(faces)
        if current_face_box is not None:
            x, y, w, h = current_face_box
            cv.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(display_frame, "Winner!!!", (width // 2 - 200, height -100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv.putText(display_frame, "F: take face photo | P: take full photo | R: restart game", (width // 2 - 200, height -60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv.putText(display_frame, winner_text, (width//2 - 170, 70), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
  
    cv.imshow(window_name, display_frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord("b"):
        # Save Player 1 and Player 2 empty background
        player1_background = player1_roi.copy()
        player2_background = player2_roi.copy()

        use_background_mode = True

        game_started = False
        countdown_started = False
        result_locked = False

        locked_player1_gesture = "unknown"
        locked_player2_gesture = "unknown"
        locked_winner_text = "Waiting"

        current_face_box = None
        winner_photo_path = None

    if key == ord("s"):
        if use_background_mode:
            game_started = True
            countdown_started = False
            result_locked = False

            locked_player1_gesture = "unknown"
            locked_player2_gesture = "unknown"
            locked_winner_text = "Waiting"

            current_face_box = None
            winner_photo_path = None

    if key == ord("p"):
        if result_locked and locked_winner_text in ["Player 1 Wins", "Player 2 Wins"]:
            full_photo = raw_frame.copy()
            
            winner_photo_path, restart_requested = open_photo_editor(full_photo, save_path="winner_full_photo.jpg")

            if restart_requested:
                game_started = False
                result_locked = False
                countdown_started = False

                locked_player1_gesture = "unknown"
                locked_player2_gesture = "unknown"
                locked_winner_text = "Waiting"

                current_face_box = None
                winner_photo_path = None
            elif winner_photo_path is not None:
                show_winner_screen(winner_photo_path)
                game_started = False
                result_locked = False
                countdown_started = False

                locked_player1_gesture = "unknown"
                locked_player2_gesture = "unknown"
                locked_winner_text = "Waiting"

                current_face_box = None
                winner_photo_path = None

    if key == ord("f"):
        if result_locked and locked_winner_text in ["Player 1 Wins", "Player 2 Wins"]:
            face_photo = crop_face(raw_frame, current_face_box, padding=30)

            if face_photo is None:
                print("No face detected")
            else:
                winner_photo_path, restart_requested = open_photo_editor(face_photo, save_path="winner_face_photo.jpg")

                if restart_requested:
                    game_started = False
                    result_locked = False
                    countdown_started = False

                    locked_player1_gesture = "unknown"
                    locked_player2_gesture = "unknown"
                    locked_winner_text = "Waiting"

                    current_face_box = None
                    winner_photo_path = None
                elif winner_photo_path is not None:
                    show_winner_screen(winner_photo_path)
                    game_started = False
                    result_locked = False
                    countdown_started = False

                    locked_player1_gesture = "unknown"
                    locked_player2_gesture = "unknown"
                    locked_winner_text = "Waiting"

                    current_face_box = None
                    winner_photo_path = None

    if key == ord("r"):
        game_started = False
        result_locked = False
        countdown_started = False

        locked_player1_gesture = "unknown"
        locked_player2_gesture = "unknown"
        locked_winner_text = "Waiting"

        current_face_box = None
        winner_photo_path = None


    if key == 27:
        break

cam.release()
cv.destroyAllWindows()