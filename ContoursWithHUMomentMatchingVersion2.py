#%% Import section
import numpy as np
import cv2
#%%
reference_file = open('Reference.txt', 'w')
test_file = open('Test.txt', 'w')
#%% Opening Image
reference_image = cv2.imread('ReferenceImages/rearCIC.jpg', cv2.IMREAD_GRAYSCALE)
test_image = cv2.imread('TestImages/rear/a-pantaloni-rearCIC.jpg', cv2.IMREAD_GRAYSCALE)
#%% Bluring
kernel_size = (3,3)
reference_image = cv2.blur(reference_image, kernel_size)
test_image = cv2.blur(test_image, kernel_size)
#%% Canny
reference_image_canny = cv2.Canny(reference_image, 100, 200)
test_image_canny = cv2.Canny(test_image, 100, 200)
#%% Morphological operation
kernel = np.ones((5,5),np.uint8)
reference_image_canny_morph = cv2.morphologyEx(reference_image_canny, cv2.MORPH_CLOSE, kernel)
test_image_canny_morph = cv2.morphologyEx(test_image_canny, cv2.MORPH_CLOSE, kernel)
#%% Finding Contours on binary image
#test_image_connected = cv2.connectedComponents(test_image_canny)
reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
test_contours, test_hierarchy = cv2.findContours(test_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse= True)
test_contours = sorted(test_contours, key=cv2.contourArea, reverse= True)
#%% Sercing Hu Moment Matches
all_match_score = np.zeros((len(reference_contours[0:8]), len(test_contours[0:8])))
for i, ref_c in enumerate(reference_contours[0:8]):
    for k, test_c in enumerate(test_contours[0:8]):
        # We use the third measure for matching and 0 is default mode because in python that parameters is unused
        all_match_score[i][k] = cv2.matchShapes(ref_c, test_c, 3, 0)
        #matched_contours[(i,k)] = cv2.matchShapes(ref_c, test_c, 3, 0)
#%% Showing Contours:
# Draw the outline of the detected contour:
#draw_contour_outline(reference_image, reference_contours, (255, 0, 0), 10)
#draw_contour_outline(test_image, test_image, (255, 0, 0), 10)
cv2.imwrite('ContoursHU/Reference_CannyBlur.jpg', reference_image_canny)
cv2.imwrite('ContoursHU/Test_cannyBlur.jpg', test_image_canny)
reference_image_copy = reference_image.copy()
test_image_copy = test_image.copy()
cv2.drawContours(reference_image_copy, reference_contours[3], -1, (0, 255, 0), 2)
cv2.drawContours(test_image_copy, test_contours[2], -1, (0, 255, 0), 2)
cv2.imwrite('ContoursHU/Reference_ContoursBlur.jpg', reference_image_copy)
cv2.imwrite('ContoursHU/Test_ContoursBlur.jpg', test_image_copy)
# cv2.imshow('Reference Canny', reference_image_canny)
# cv2.imshow('Test Canny', test_image_canny)
# cv2.imshow("Reference Image", reference_image)
# cv2.imshow("Test Image", test_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#%% Bounding rect usage:
x, y, w, h = cv2.boundingRect(reference_contours[0])
cv2.rectangle(reference_image, (x,y), (x+w, y+h),(255,0,255), 5)
x, y, w, h = cv2.boundingRect(test_contours[2])
cv2.rectangle(test_image, (x,y), (x+w,y+h), (255,0,255), 5)
cv2.imshow('Reference', reference_image)
cv2.imshow('Test', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% Contour matching find:
## Remember that the min value is the right matched
matched_contours = {}
for i in range(0, len(reference_contours[0:8])):
    matched_contours[i] = np.argmin(all_match_score[i])
