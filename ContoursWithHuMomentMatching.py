#%% Import section
import numpy as np
import cv2
#%%
reference_file = open('Reference.txt', 'w')
test_file = open('Test.txt', 'w')
#%% define centroid function:
def centroid(moments):
    """Returns centroid based on moments"""
    if moments['m00'] == 0:
        print('Zero Value')
        return 0,0
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
#%% Bilateral filter #TODO da capire bene come usarlo potrebbe essere migliore del gaussian
test_image = cv2.bilateralFilter(test_image, 9, 9, 9)
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
#%% Computing reference Moments
reference_hu_moment_dict = {}
for index, contour in enumerate(reference_contours):
    reference_hu_moment_dict[index] = { 'moment': cv2.moments(contour),
                                        'hu_moment': cv2.HuMoments(cv2.moments(contour)),
                                        #'log_hu_moment': -np.sign(cv2.HuMoments(cv2.moments(contour)))*np.log10(np.abs(cv2.HuMoments(cv2.moments(contour))))
                                        }
#%% Computing test Moments
test_hu_moment_dict = {}
for index, contour in enumerate(test_contours):
    test_hu_moment_dict[index] = { 'moment': cv2.moments(contour),
                                   'hu_moment': cv2.HuMoments(cv2.moments(contour)),
                                   #'log_hu_moment': -np.sign(cv2.HuMoments(cv2.moments(contour)))*np.log10(np.abs(cv2.HuMoments(cv2.moments(contour))))
                                   }
#%% Computing Reference Centroid of the moments
reference_centroid = {}
for index in reference_hu_moment_dict.keys():
    reference_centroid[index] = centroid(reference_hu_moment_dict[index]['moment'])
#%% Computing Test Centroid of the moments
test_centroid = {}
for index in test_hu_moment_dict.keys():
    reference_centroid[index] = centroid(test_hu_moment_dict[index]['moment'])
#%% Print Result:
for index in reference_hu_moment_dict.keys():
    print('Reference Moment at contour index: ', index, file= reference_file)
    print(reference_hu_moment_dict[index], file= reference_file)
for index in test_hu_moment_dict.keys():
    print('Test Moment at contour index: ', index, file= test_file)
    print(test_hu_moment_dict[index], file= test_file)
for index in reference_centroid.keys():
    print('Reference centroid at contour index: ', index, file= reference_file)
    print(reference_centroid[index], file= reference_file)
for index in test_centroid.keys():
    print('Test centroid at contour index: ', index, file= test_file)
    print(test_centroid[index], file= test_file)
#%% Sercing Hu Moment Matches
matched_contours = {}
for i, ref_c in enumerate(reference_contours):
    for k, test_c in enumerate(test_contours):
        matched_contours[(i,k)] = cv2.matchShapes(ref_c, test_c, 3, 0)


#%% Showing Contours:
# Draw the outline of the detected contour:
#draw_contour_outline(reference_image, reference_contours, (255, 0, 0), 10)
#draw_contour_outline(test_image, test_image, (255, 0, 0), 10)
cv2.imwrite('ContoursHU/Reference_CannyBlur.jpg', reference_image_canny)
cv2.imwrite('ContoursHU/Test_cannyBlur.jpg', test_image_canny)
reference_image_copy = reference_image.copy()
test_image_copy = test_image.copy()
cv2.drawContours(reference_image_copy, reference_contours[0], -1, (0, 255, 0), 2)
cv2.drawContours(test_image_copy, test_contours[0], -1, (0, 255, 0), 2)
cv2.imwrite('ContoursHU/Reference_ContoursBlur.jpg', reference_image)
cv2.imwrite('ContoursHU/Test_ContoursBlur.jpg', test_image)
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
x, y, w, h = cv2.boundingRect(test_contours[0])
cv2.rectangle(test_image, (x,y), (x+w,y+h), (255,0,255), 5)
cv2.imshow('Reference', reference_image)
cv2.imshow('Test', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()