#%% Import section
import os
import numpy as np
import cv2
#%% Function for concatenating result:
def concatenate(image1, image2):
    total_width = image1.shape[1] + image2.shape[1] + 1
    if image1.shape[0] >= image2.shape[0]:
        total_heigh = image1.shape[0]
    else:
        total_heigh = image2.shape[0]
    new_image = np.zeros((total_heigh,total_width))
    new_image[0:image1.shape[0], 0:image1.shape[1]] = image1
    new_image[0:image2.shape[0], image1.shape[1] + 1:image1.shape[1] + image2.shape[1] + 1] = image2
    return new_image
#%% Global parameters definition:
bluring_kernel_size = (3, 3) #TODO try different size that can be usefull
morphological_kernel = np.ones((5, 5), np.uint8) #TODO try different kernel that can be usefull (to search)
canny_low = 100
canny_high = 200
number_of_contours = 10
#%% Opening Reference Image
reference_image = cv2.imread('ReferenceImages/rearCIC.jpg', cv2.IMREAD_GRAYSCALE)
#%% Bluring
reference_image = cv2.blur(reference_image, bluring_kernel_size)
#%% Canny
reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
#%% Morphological operation
reference_image_canny_morph = cv2.morphologyEx(reference_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
#%% Finding Contours on binary image
# cv2.RETR_CCOMP -> return the contours complete list (all contours)
# cv2.CHAIN_APPROX_NONE -> this not approximate a connection between contours
reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# Rerder contours in descending mode to have the biggest area for first
reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse= True)
# keep only first 10 contours
reference_contours = reference_contours[0:number_of_contours]
#%% Starting searching on test image:
# In this section the parameters are the same of the reference creation
# Definition of the test folder
test_images_path = 'TestImages/rear/'
saving_base_path = 'ContoursHU/New/'
for root, directories, files in os.walk(test_images_path):
    for file in files:
        # Read test file
        test_immage_path = os.path.join(root,file)
        test_image_name = file.split('.')[0]
        saving_complete_path = saving_base_path + test_image_name
        if file.endswith('.DS_Store'):
            continue
        if not os.path.exists(saving_complete_path):
            os.mkdir(saving_complete_path)
        test_image = cv2.imread(test_immage_path, cv2.IMREAD_GRAYSCALE)
        # Blur test image
        test_image = cv2.blur(test_image, bluring_kernel_size)
        # Edge detector
        test_image_canny = cv2.Canny(test_image, canny_low, canny_high)
        # Morphological operation
        test_image_canny_morph = cv2.morphologyEx(test_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
        # Contours finding
        test_contours, test_hierarchy = cv2.findContours(test_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        test_contours = sorted(test_contours, key=cv2.contourArea, reverse=True)
        test_contours = test_contours[0:number_of_contours]
        # Search for mathing:
            # use a numpy nd_array for saving matching result is usefull for the fastest way to get match after
            # We use the third measure for matching and 0 is default mode because in python that parameters is unused
        all_match_score = np.zeros((len(reference_contours), len(test_contours)))
        # row -> reference
        # column -> test
        for i, ref_c in enumerate(reference_contours):
            for k, test_c in enumerate(test_contours):
                all_match_score[i][k] = cv2.matchShapes(ref_c, test_c, 3, 0)
        # Saving result
        np.savetxt(saving_complete_path + '/' + test_image_name + '-matches.csv', all_match_score, delimiter=';')
        for index, contour in enumerate(reference_contours):
            reference_image_copy = reference_image.copy()
            cv2.drawContours(reference_image_copy, contour, -1, (0, 255, 0), 2)
            cv2.imwrite(saving_complete_path + '/rearCIC' + 'reference_contour-' + str(index) + '.png',
                        reference_image_copy)
        for index, contour in enumerate(test_contours):
            test_image_copy = test_image.copy()
            cv2.drawContours(test_image_copy, contour, -1, (0, 255, 0), 2)
            cv2.imwrite(saving_complete_path + '/' + test_image_name + '-test_contour-' + str(index) + '.png',
                        test_image_copy)
        # Concatenating image with contours
        # for reference_index in range(0, len(reference_contours)):
        #     test_index = np.argmin(all_match_score[reference_index])
        #     if not os.path.exists(saving_complete_path):
        #         os.mkdir(saving_complete_path)
        #     cv2.imwrite(saving_complete_path + '/Reference_Canny.jpg', reference_image_canny)
        #     cv2.imwrite(saving_complete_path + '/Test_canny.jpg', test_image_canny)
        #     ## Create new figure to have the possibility to compare result
        #     reference_image_copy = reference_image.copy()
        #     cv2.drawContours(reference_image_copy, reference_contours[reference_index], -1, (0, 255, 0), 2)
        #     cv2.drawContours(test_image, test_contours[test_index], -1, (0, 255, 0), 2)
        #     concatenate_image = concatenate(reference_image_copy, test_image)
        #     cv2.imwrite(saving_complete_path + '/' + test_image_name + 'reference_contour-' +
        #                 str(reference_index) + '-test_contour-' + str(test_index) + '.png', concatenate_image)
