#%% Import section
import cv2
import numpy as np
#%% Global parameters definition:
bluring_kernel_size = (3, 3) #TODO try different size that can be usefull
morphological_kernel = np.ones((5, 5), np.uint8) #TODO try different kernel that can be usefull (to search)
canny_low = 100
canny_high = 200
number_of_contours = 10
#%% Opening Reference Image
reference_image = cv2.imread('TestImages/rear/a-pantaloni-rearCIC.jpg', cv2.IMREAD_GRAYSCALE)
#%% Bluring
reference_image = cv2.blur(reference_image, bluring_kernel_size)
#%% Canny
reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
#%% Finding Contours on binary image
# cv2.RETR_CCOMP -> return the contours complete list (all contours)
# cv2.CHAIN_APPROX_NONE -> this not approximate a connection between contours
reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# Rerder contours in descending mode to have the biggest area for first
reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse=True)
#%% approximate contour with poly line
#for c in reference_contours:
peri = cv2.arcLength(reference_contours[5], True)
approx = cv2.approxPolyDP(reference_contours[5], 0.2 * peri, True)
img = reference_image.copy()
cv2.drawContours(img, [approx], -1, (0, 0, 0))
cv2.imwrite('result-contours-new-test.jpg', img)