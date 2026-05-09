import cv2 as cv
import numpy as np

def find_hand_contour(mask, min_area=1000):
    # Find all outer contours in the binary mask
    # We use RETR_EXTERNAL because we only care about he OUTER boundary of the hand,
    # not inner holes or nested countours.
    # Also we use CHAIN_APPROX_SIMPLE to compress contour points.
    # where OpenCV stores only important boundary points instead of storing every single point on the contour
    # This saves memory and makes processing faster.
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    
    # Choose the contour with the largest area.
    # Where we assume the hand is the largest whie object inside the Region of Interest.
    largest_contour = max(contours, key=cv.contourArea)

    # Getting the area inside the contour (We use it to remove very small contours caused by noise)
    area = cv.contourArea(largest_contour)

    if area < min_area:
        return None
    
    return largest_contour


def draw_contour_on_frame(frame, contour, box, color, thickness):
    if contour is None:
        return
    
    # Getting the box position
    x, y, w, h = box

    # The contour points were found inside the Region of Interst,
    # so their coordinates start from (0, 0) inside that cropped box.

    # But we want to draw them on the FULL camera frame.
    # So we must shift every contour point by the ROI's top-left corner (x, y).
    shifted_contour = contour + np.array([[[x,y]]])

    cv.drawContours(frame, [shifted_contour], -1, color, thickness)

def draw_centroid_on_frame(frame, contour, box, color):
    if contour is None:
        return
    
    x, y, w, h = box

    shifted_contour = contour + np.array([[[x, y]]])

    # cv.moments calculates shape moments from the contour.
    # Moments are useful for finding the center point of the shape.
    M = cv.moments(shifted_contour)

    # M["m00"] is like the area of the contour.
    if M["m00"] == 0:
        return

    # Centroid formula:
    # cx = m10/m00
    # cy = m01 /m00

    cx = int(M["m10"]/ M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Draw a red dot at the centroid
    cv.circle(frame, (cx, cy), 6, color, -1)

def draw_convex_hull_on_frame(frame, contour, box, color, thickness):

    if contour is None:
        return
    
    x, y, w, h = box

    shifted_contour = contour + np.array([[[x, y]]])

    # cv.convexHull() finds the smallest convex shape enclosing the contour
    hull = cv.convexHull(shifted_contour)

    # Draw hull on the full frame
    cv.drawContours(frame, [hull], -1, color, thickness)


def count_defects(contour, depth_threshold=8000):
    if contour is None:
        return
    
    # We set returnPoints = False so that it fives the indices of hull points instead of actual points.
    # Because cv.convexityDefects() needs hull indices, not hull coordinates.
    hull_indices = cv.convexHull(contour, returnPoints=False)

    if hull_indices is None or len(hull_indices) < 4:
        return 0
    
    # cv.convexityDefects() returns gaps between contour and convex hull.
    defects = cv.convexityDefects(contour, hull_indices)

    if defects is None:
        return 0
    
    defect_count = 0

    for i in range(defects.shape[0]):
        # start_idx = first outer point
        # end_idx = second outer point 
        # far_idx = deepest point inside the gap
        # depth = how deep the gap is
        start_idx, end_idx, far_idx, depth = defects[i, 0]

        # Ignor etiny defects cause by noisy contour edges
        if depth > depth_threshold:
            defect_count += 1

    return defect_count

def draw_defect_count_on_frame(frame, contour, box, label, color):
    x, y, w, h = box

    defect_count = count_defects(contour)

    cv.putText(frame, f"{label} defects: {defect_count}", (x, y + h + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
