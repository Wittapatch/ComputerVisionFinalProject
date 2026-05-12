import cv2 as cv
from contour_detection import count_defects
import pandas as pd
def extract_features(contour,roi_shape):
    """
    We want to convert one hand contour into numerical features for ML.
    
    Features:
    
    """

    if contour is None:
        return None
    #Get the height and width of the region of interest
    roi_h = roi_shape[0]
    roi_w = roi_shape[1]
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
    area_ratio = area/(roi_w*roi_h)
    #Measures of how much of the area of the ROI is filled by the hull area.
    #Helps measure how spread out the gesture is. For example paper may have a larger value, which rock is smaller.
    hull_area_ratio = hull_area/(roi_w*roi_h)
    #Put all these features into a pandas series
    features = pd.Series({
    "defect_count": defect_count,
    "solidity": solidity,
    "aspect_ratio": aspect_ratio,
    "extent": extent,
    "area_ratio": area_ratio,
    "hull_area_ratio": hull_area_ratio
    })
    return features
    
