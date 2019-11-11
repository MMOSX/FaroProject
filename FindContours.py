#%%
import cv2
import numpy as np
#%%
bluring_kernel_size = (3, 3)
canny_low = 150
canny_high = 200
morphological_kernel = np.ones((5, 5), np.uint8)
#%% Opening Reference Image
reference_image = cv2.imread('TestImages/rear/t-pantaloni-rearCIE.jpg', cv2.IMREAD_GRAYSCALE)
#%% Bluring
reference_image = cv2.blur(reference_image, bluring_kernel_size)
#%% Canny
reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
#%% Morphological operation
reference_image_canny_morph = cv2.morphologyEx(reference_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
#reference_image_canny_morph = cv2.morphologyEx(reference_image_canny_morph, cv2.MORPH_CLOSE, morphological_kernel)
#%% Finding Contours on binary image
# cv2.RETR_CCOMP -> return the contours complete list (all contours)
# cv2.CHAIN_APPROX_NONE -> this not approximate a connection between contours
reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny_morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# Rerder contours in descending mode to have the biggest area for first
reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse=True)
hull = []
# calculate points for each contour
for i in range(len(reference_contours)):
    # creating convex hull object for each contour
    peri = cv2.arcLength(reference_contours[i], True)
    approx = cv2.approxPolyDP(reference_contours[i], 0.02 * peri, True)
    hull.append(approx)

# create an empty black image
drawing = np.zeros((reference_image_canny.shape[0], reference_image_canny.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(reference_contours)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull
    # draw ith contour
    #cv2.drawContours(drawing, reference_contours, i, color_contours, 1, 8, reference_hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, hull, i, color, 1, 8)

cv2.imwrite('Approx-result-Morph.jpg', drawing)