import cv2 as cv
from contour_detection import count_defects

def extract_features(contour):
    """
    We want to convert one hand contour into numerical features for ML.
    
    Features:
    
    """

    if contour is None:
        return None
    
