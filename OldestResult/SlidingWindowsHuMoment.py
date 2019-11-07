#%% Import section
import os
import numpy as np
import cv2
#%%
bluring_kernel_size = (3, 3) #TODO try different size that can be usefull
morphological_kernel = np.ones((5, 5), np.uint8) #TODO try different kernel that can be usefull (to search)
canny_low = 100
canny_high = 200
number_of_contours = 10
#%% sliding window parameter
stepSize = 50
(w_width, w_height) = (50, 50) # window size
#%% Test folder
tests_path = 'TestImages/rear/'
#%% Opening reference
references_path = 'ReferencePiece/rear/'
#%% Creating reference data
reference_data = {}
for root, directories, files in os.walk(references_path):
    for file in files:
        reference_name = file.split('.')[0]
        reference_path = os.path.join(root + file)
        reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        # Bluring
        reference_image = cv2.blur(reference_image, bluring_kernel_size)
        # Canny
        reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
        # Morphological operation
        reference_image_canny_morph = cv2.morphologyEx(reference_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
        reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # Rerder contours in descending mode to have the biggest area for first
        reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse=True)
        # keep only first 10 contours
        reference_contours = reference_contours[0:number_of_contours]
        reference_data[reference_name] = reference_contours

#%%
test_data = {}
for root, directories, files in os.walk(tests_path):
    for file in files:
        if file.endswith('.DS_Store'):
            continue
        test_name = file.split('.')[0]
        test_path = os.path.join(root + '/' + file)
        test_image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        for x in range(0, test_image.shape[1] - w_width, stepSize):
            for y in range(0, test_image.shape[0] - w_height, stepSize):
                window = test_image[x:x + w_width, y:y + w_height, :]
                # Bluring
                test_image_b = cv2.blur(window, bluring_kernel_size)
                # Canny
                test_image_canny = cv2.Canny(test_image_b, canny_low, canny_high)
                # Morphological operation
                test_image_canny_morph = cv2.morphologyEx(test_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
                test_contours, test_hierarchy = cv2.findContours(test_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                # Rerder contours in descending mode to have the biggest area for first
                test_contours = sorted(test_contours, key=cv2.contourArea, reverse=True)
                # keep only first 10 contours
                test_contours = test_contours[0:number_of_contours]
                if not test_name in test_data.keys():
                    test_data[test_name] = []
                test_data[test_name].append({
                    'x': x,
                    'y': y,
                    'contours': test_contours
                })
