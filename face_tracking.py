import cv2 as cv
import numpy as np

def create_face_tracker():
    # Create a dictionary that stores CamShift tracking information

    tracker = {
        # track_window = Current face box used by CamShift
        "track_window": None,
        # roi_hist = HSV histogram of the detected face where its used to find similar face-colored regions
        "roi_hist": None,
        # let the tracker not start yet
        "initialized": False,
        # Stopping condition for CamShift
        # Where it stops after 10 iterations or when movement is smaller than 1 pixel.
        "term_crit": (
            cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
            10,
            1
        )
    }
    return tracker

def initialize_face_tracker(tracker, frame, face_box):
    if face_box is None:
        return False
    
    x, y, w, h = face_box

    tracker["track_window"] = (x, y, w, h)

    # Crop the detected face region
    face_roi = frame[y: y+h, x: x+w]

    if face_roi.size == 0:
        return False
    
    # Convert face ROI to HSV
    # Because CamShift works well with HSV where it tracks the color distribution.
    hsv_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2HSV)

    # Mask out very dark or weak-color pixels
    # This helps the histogram focus on useful face color information.
    mask = cv.inRange(hsv_roi, np.array([0, 30, 32], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))


    # Calculate histogram using Hue and Saturation channels.
    roi_hist = cv.calcHist([hsv_roi], [0, 1], mask, [32, 32], [0, 180, 0, 256])

    # Normalize historgram so values are in a useful range.
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    tracker["roi_hist"] = roi_hist
    tracker["initialized"] = True

    return True

def update_face_tracker(tracker, frame):
    
    if not tracker["initialized"]:
        return None
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Back projection finds pixels similar to the original face histogram
    back_projection = cv.calcBackProject([hsv], [0, 1], tracker["roi_hist"], [0, 180, 0, 256], 1)

    rotated_box, new_track_window = cv.CamShift(back_projection, tracker["track_window"], tracker["term_crit"])

    x, y, w, h = new_track_window

    frame_h, frame_w = frame.shape[:2]

    # Keep tracking box inside the frame
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    # Check if the box becomes too small
    if w <= 10 or h <= 10:
        reset_face_tracker(tracker)
        return None
    
    tracker["track_window"] = (x, y, w, h)
    return tracker["track_window"]
    

def reset_face_tracker(tracker):

    tracker["track_window"] = None
    tracker["roi_hist"] = None
    tracker["initialized"] = False
    
