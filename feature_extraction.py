import cv2 as cv
from contour_detection import count_defects
import pandas as pd
import numpy as np
def extract_features(contour,roi):
    if contour is None:
        return None
    #Get the height and width of the region of interest
    roi_h = roi.shape[0]
    roi_w = roi.shape[1]
    #Calculate number of pixels inside hand contour
    area = cv.contourArea(contour)
    #Return None if the area of the hand is too small
    if area<1000:
        return None
    #Get convex hull of the hand contour
    hull = cv.convexHull(contour)
    #Calculate the area of the convex hull of the hand contour
    hull_area = cv.contourArea(hull)
    #Since we will use hull area for division, it can't be zero or else it will cause an error
    if hull_area == 0:
        return None
    #Calculate solidity of the contour of the hand, which is how much the area of the contour of the hand
    #fills it's convex hull area.
    solidity = area/hull_area
    #Get the x, y coordinate and the width and height of the bounding rectangle of the hand contour. Used to calculate
    #aspect ratio of the hand contour.
    x,y,w,h = cv.boundingRect(contour)
    #Get aspect ratio value of hand gestures
    if h!=0:
        aspect_ratio = w/h
    else:
        aspect_ratio = 0
    #Calculate extent, which is how much the area of the hand fills the area of the bounding rectangle
    if w*h != 0:
        extent = area/(w*h)
    else:
        extent = 0
    #Get defect count, which are the number of dents between the convex and the convex hull.
    #In this case, it is basically the number of gaps between the fingers
    defect_count = count_defects(contour)
    #Calculate how much of the ROI area is covered by the hand's contour area.
    #Useful for the model to understand the distance of the hand from the camera
    if roi_w *roi_h!=0:
        area_ratio = float(area/(roi_w*roi_h))
    #Measures of how much of the area of the ROI is filled by the hull area.
    #Helps measure how spread out the gesture is. For example paper may have a larger value, which rock is smaller.
        hull_area_ratio = float(hull_area/(roi_w*roi_h))
    else:
        area_ratio =0
        hull_area_ratio = 0
    perimeter = cv.arcLength(curve= contour,closed= True)
    #Circularity is how much the contour is circular. Rock would be have higher circularity value than scissors
    if perimeter != 0:
        circularity = 4*np.pi *area/(perimeter**2)
    else:
        circularity=0
    #Hu moments summarizes overall contour mathematically. 
    moments = cv.moments(contour)
    hu = cv.HuMoments(moments).flatten()
    #These values are extremely small, so we have to log transform it first and add 10^-10 to prevent logging 0 valuue
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    #m00 is the zeroth order moment. It represents the area inside the contour
    #m10 is the x-weighted area. It is sum of all x-positions weighted by the pixels of the contour
    #m01 is the y-weighted area. It is the sum of all y-positions weighted by the pixels/area of the contour
    #cx and cy represents the centroid of the contour
    if moments["m00"] != 0:
        cx = moments["m10"]/moments["m00"]
        cy = moments["m01"]/moments["m00"]
        cx_ratio = cx/roi_w
        cy_ratio = cy/roi_h
    #Put all these features into a pandas series
    features = pd.Series({
    "defect_count": defect_count,
    "solidity": solidity,
    "aspect_ratio": aspect_ratio,
    "extent": extent,
    "area_ratio": area_ratio,
    "hull_area_ratio": hull_area_ratio,
    "circularity":circularity,
    "hu_log_1":hu_log[0],
    "hu_log_2":hu_log[1],
    "hu_log_3":hu_log[2],
    "hu_log_4":hu_log[3],
    "hu_log_5":hu_log[4],
    "hu_log_6":hu_log[5],
    "hu_log_7":hu_log[6],
    "cx_ratio":cx_ratio,
    "cy_ratio":cy_ratio
    })
    return features
    
def create_hand_mask(roi):
    # Convert BGR to YCrCb color space
    ycrcb = cv.cvtColor(roi, cv.COLOR_BGR2YCrCb)

    # Skin-color range
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    # Create binary skin mask
    mask = cv.inRange(ycrcb, lower, upper)

    # Clean the mask
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    return mask
