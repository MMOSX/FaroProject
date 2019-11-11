#%% Import section
import imutils
import cv2
import dlib
import numpy as np
#%% Global parameters definition:
bluring_kernel_size = (3, 3) #TODO try different size that can be usefull
morphological_kernel = np.ones((5, 5), np.uint8) #TODO try different kernel that can be usefull (to search)
canny_low = 100
canny_high = 200
number_of_contours = 10
#%% Opening Reference Image
reference_image = cv2.imread('TestImages/rear/t-pantaloni-rearCIE.jpg', cv2.IMREAD_GRAYSCALE)
#%% Bluring
reference_image = cv2.blur(reference_image, bluring_kernel_size)
#%% Canny
reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
#%% Morphological operation
#reference_image_canny_morph = cv2.morphologyEx(reference_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
#%% Finding Contours on binary image
# cv2.RETR_CCOMP -> return the contours complete list (all contours)
# cv2.CHAIN_APPROX_NONE -> this not approximate a connection between contours
reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# Rerder contours in descending mode to have the biggest area for first
reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse= True)
#%%
rects = []
#for c in reference_contours:
peri = cv2.arcLength(reference_contours[5], True)
approx = cv2.approxPolyDP(reference_contours[5], 0.02 * peri, True)
x, y, w, h = cv2.boundingRect(approx)
reference_image_copy = reference_image.copy
if h >= 15:
    # if height is enough
    # create rectangle for bounding
    rect = (x, y, w, h)
    rects.append(rect)
    cv2.rectangle(reference_image, (x, y), (x+w, y+h), (0, 255, 0), 1)

cv2.imwrite('result-test.jpg', reference_image)