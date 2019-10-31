#%% Import section
import numpy as np
import cv2
#%% define centroid function:
def centroid(moments):
    """Returns centroid based on moments"""

    x_centroid = round(moments['m10'] / moments['m00'])
    y_centroid = round(moments['m01'] / moments['m00'])
    return x_centroid, y_centroid
def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)
#%% Opening Image
reference_image = cv2.imread('ReferenceImages/rearCIC.jpg', cv2.IMREAD_GRAYSCALE)
test_image = cv2.imread('TestImages/rear/a-petrelli-rearCIC.jpg', cv2.IMREAD_GRAYSCALE)
#%% Thresholding images
#thresh, reference_image_binary = cv2.threshold(reference_image, 127, 255, 0)
#thresh, test_image_binary = cv2.threshold(test_image, 127, 255, 0)
#%% Bluring
reference_image = cv2.blur(reference_image, (3,3))
test_image = cv2.blur(test_image, (3,3))
#%% Canny
reference_image_canny = cv2.Canny(reference_image, 100, 200)
test_image_canny = cv2.Canny(test_image, 100, 200)
#%% Morphological operation
kernel = np.ones((3,3),np.uint8)
reference_image_canny_morph = cv2.morphologyEx(reference_image_canny, cv2.MORPH_CLOSE, kernel)
test_image_canny_morph = cv2.morphologyEx(test_image_canny, cv2.MORPH_CLOSE, kernel)
#%% Finding Contours on binary image
#test_image_connected = cv2.connectedComponents(test_image_canny)
reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
test_contours, test_hierarchy = cv2.findContours(test_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse= True)
test_contours = sorted(test_contours, key=cv2.contourArea, reverse= True)
#%% Computing moments
reference_moments = cv2.moments(reference_contours[0])
test_moments = cv2.moments(test_contours[0])
#%% Computing Centroid of the moments
reference_x, reference_y = centroid(reference_moments)
test_x, test_y = centroid(test_moments)
#%% Computing Hu moments
reference_hu_moments = cv2.HuMoments(reference_moments)
test_hu_moments = cv2.HuMoments(test_moments)
#%% Print Result:
print('Reference Moment Result: {0}'.format(reference_moments))
print('Test Moment Result: {0}'.format(test_moments))
print('Reference Centroid position: ({0},{1})'.format(reference_x, reference_y))
print('Test Centroid position: ({0},{1})'.format(test_x, test_y))
print('Reference Hu Moment Result: {0}'.format(reference_hu_moments))
print('Test Hu Moment Result: {0}'.format(test_hu_moments))
#%% Showing Contours:
# Draw the outline of the detected contour:
#draw_contour_outline(reference_image, reference_contours, (255, 0, 0), 10)
#draw_contour_outline(test_image, test_image, (255, 0, 0), 10)
cv2.imwrite('ContoursHU/Reference_CannyBlur.jpg', reference_image_canny)
cv2.imwrite('ContoursHU/Test_cannyBlur.jpg', test_image_canny)
cv2.drawContours(reference_image, reference_contours[3], -1, (0,255,0),2)
cv2.drawContours(test_image, test_contours[0], -1, (0,255,0),2)
cv2.imwrite('ContoursHU/Reference_ContoursBlur.jpg', reference_image)
cv2.imwrite('ContoursHU/Test_ContoursBlur.jpg', test_image)
# cv2.imshow('Reference Canny', reference_image_canny)
# cv2.imshow('Test Canny', test_image_canny)
# cv2.imshow("Reference Image", reference_image)
# cv2.imshow("Test Image", test_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()