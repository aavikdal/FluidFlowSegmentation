# Imports
import numpy as np
import cv2
from skimage.filters import threshold_multiotsu

def multiThreshold(img, margin):
    # Denoise image
    img = cv2.fastNlMeansDenoising(img)
    
    # Calculate two threshold values
    thresholds = threshold_multiotsu(img)
    
    # If the distance between the threshold is below a margin, consider there to be two classes
    if (thresholds[1] - thresholds[0]) < margin:
        ret, segmented = cv2.threshold(img, 2, 2, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        segmented = np.where(segmented == 0, 1, 2)
        
    # If not, consider there to be three classes     
    else:
        segmented = np.digitize(img, bins=thresholds).astype(np.uint8)
    return segmented