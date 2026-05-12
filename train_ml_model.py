import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from contour_detection import find_hand_contour
from feature_extraction import extract_features




