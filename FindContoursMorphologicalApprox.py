#%%
import cv2
import numpy as np
#%%
bluring_kernel_size = (3, 3)
canny_low = 150
canny_high = 200
morphological_kernel = np.ones((5, 5), np.uint8)
#%% Opening Reference Image
reference_image = cv2.imread('TestImages/rear/t-pantaloni-rearCIE.jpg')
reference_image_hsv = reference_image.copy()#cv2.cvtColor(reference_image, cv2.COLOR_BGR2HSV)
#%% Bluring
reference_image = cv2.blur(reference_image, bluring_kernel_size)
#%% Canny
reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
#%% filter black color
mask1 = cv2.inRange(reference_image_hsv, np.array([0, 0, 0]), np.array([180, 255, 125]))
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
mask1 = cv2.Canny(mask1, 150, 200)
mask1 = cv2.GaussianBlur(mask1, (1, 1), 0)
mask1 = cv2.Canny(mask1, 150, 200)
#%%
cnts, hierarchy = cv2.findContours(mask1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # get largest five contour area
#%%
# create an empty black image
drawing = np.zeros((reference_image_canny.shape[0], reference_image_canny.shape[1], 3), np.uint8)
for i in range(len(cnts)):
    color_contours = (0, 255, 0)  # green - color for contours
    color = (255, 0, 0)  # blue - color for convex hull
    # draw ith contour
    #cv2.drawContours(drawing, reference_contours, i, color_contours, 1, 8, reference_hierarchy)
    # draw ith convex hull object
    cv2.drawContours(drawing, cnts, i, color, 1, 8)

cv2.imwrite('Approx-result-Morph-RECTMorph2.jpg', drawing)