import cv2 as cv
from contour_detection import count_defects

def calculate_solidity(contour):
    # Solidity tells us how compact the hand shape is
    # where the formula is solidity = contour area / convex hull area
    # We will this for our classification of rock, paper and scissors

    if contour is None:
        return 0
    
    contour_area =cv.contourArea(contour)

    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)

    if hull_area == 0:
        return 0
    
    solidity = contour_area / hull_area

    return solidity


def classify_gesture(contour):
    if contour is None:
        return "unknown"
    
    area = cv.contourArea(contour)

    if area < 1000:
        return "unknown"
    
    defect_count = count_defects(contour)
    solidity = calculate_solidity(contour)

    # Rock = closed fist
    # very few convexity defects
    # It is also compact, so solidity is high
    if defect_count <= 1 and solidity > 0.80:
        return "rock"
    
    # Paper = open hand
    # It usually has many gaps between fingers
    if defect_count >= 3:
        return "paper"
    
    # Scissors = two fingers
    # It usually has 1 or 2 strong finger gaps 
    if 1 <= defect_count <= 2:
        return "scissors"
    
    return "unknown"

def draw_gesture_on_frame(frame, contour, box, label, color):
    x, y, w, h = box
    
    gesture = classify_gesture(contour)
    cv.putText(frame, f"{label}: {gesture}", (x, y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return gesture