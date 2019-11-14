#%% Import Section
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime
#%% parameters
eps = 1e-7
#%% path section
tests_path = 'TestImages/'
references_path = 'ReferenceImages/'
result_path = 'Result/'
#%% parameter section
useTwo = False
# good point parameters
distanca_coefficient = 0.75
# gms parameter
gms_thresholdFactor = 3
gms_withRotation = True
gms_withScale = True
# flann parameter
flann_trees = 5
flann_checks = 50
#%% Initialize
# SIFT init
sift = cv2.xfeatures2d.SIFT_create()
# FLANN parameter
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)
search_params = dict(checks=flann_checks)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
#%% Starting reference creation
reference_init = datetime.now()
reference_dictionaries = {}
for root, directories, files in os.walk(references_path):
    for file in files:
        if file.endswith('.DS_Store'):
            continue
        # opening reference image as grayscale
        reference_name = file.split('.')[0]
        reference_image_path = os.path.join(root, file)
        reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
        reference_descriptors /= (reference_descriptors.sum(axis=1, keepdims=True) + eps)
        reference_descriptors = np.sqrt(reference_descriptors)
        reference_dictionaries[reference_name] = (reference_image, reference_keypoints, reference_descriptors)
reference_end = datetime.now()
reference_time = reference_end - reference_init
# %% Starting test creation
test_init = datetime.now()
test_dictionaries = {}
for root, directories, files in os.walk(tests_path):
    for file in files:
        if file.endswith('.DS_Store'):
            continue
        test_name = file.split('.')[0]
        test_image_path = os.path.join(root, file)
        test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        test_keypoints, test_descriptors = sift.detectAndCompute(test_image, None)
        test_descriptors /= (test_descriptors.sum(axis=1, keepdims=True) + eps)
        test_descriptors = np.sqrt(test_descriptors)
        test_dictionaries[test_name] = (test_image, test_keypoints, test_descriptors)

test_end = datetime.now()
test_time = test_end - test_init
#%% Starting Matching
match_init = datetime.now()
recognize_result = {}
for test_name in test_dictionaries.keys():
    recognize_result[test_name] = {}
    test_descriptors = test_dictionaries[test_name][2]
    test_keypoints = test_dictionaries[test_name][1]
    test_image = test_dictionaries[test_name][0]
    for reference_name in reference_dictionaries.keys():
        reference_descriptors = reference_dictionaries[reference_name][2]
        reference_keypoints = reference_dictionaries[reference_name][1]
        reference_image = reference_dictionaries[reference_name][0]
        flann_matches = flann.knnMatch(reference_descriptors, test_descriptors, k=2)
        matches_copy = []
        for i, (m, n) in enumerate(flann_matches):
            if m.distance < distanca_coefficient * n.distance:
                matches_copy.append(m)
            elif useTwo:
                matches_copy.append(n)
        gsm_matches = cv2.xfeatures2d.matchGMS(reference_image.shape, test_image.shape, keypoints1=reference_keypoints,
                                               keypoints2=test_keypoints, matches1to2=matches_copy,
                                               withRotation= gms_withRotation, withScale=gms_withScale,
                                               thresholdFactor=gms_thresholdFactor)
        recognize_result[test_name][reference_name] = {
            'scoreFlann': len(flann_matches),
            'scoreGsm': len(gsm_matches),
            'flannMatch': flann_matches,
            'gsmMatch': gsm_matches,
        }
match_end = datetime.now()
match_time = match_end - match_init
#%% showing result
match_extract_init = datetime.now()
match_pair = []
for test_name in recognize_result.keys():
    results = recognize_result[test_name]
    reference_key = ""
    maximum = 0
    for reference_name in results.keys():
        if results[reference_name]['scoreGsm'] > maximum:
            maximum = results[reference_name]['scoreGsm']
            reference_key = reference_name
    match_pair.append((test_name, reference_key))
match_extract_end = datetime.now()
match_extract_time = match_extract_end - match_extract_init
#%%
with open(result_path + 'Match_result.txt', 'w') as result_file:
    result_file.write('Section execution time:\n Reference Keypoint and feature extraction: ' + str(reference_time) +
                      '\n Test Keypoint and Feature extraction: ' + str(test_time) +
                      '\n Match time: ' + str(match_time) +
                      '\n Extract match time: ' + str(match_extract_time))
    result_file.write('Test\t' + 'Refere\n')
    for (t, r) in match_pair:
        result_file.write(t + '\t' + r + '\n')
        if (t == "") or (r == ""):
            continue
        reference_image = reference_dictionaries[r][0]
        test_image = test_dictionaries[t][0]
        matches = recognize_result[t][r]['gsmMatch']
        result_match_img = cv2.drawMatches(reference_image, reference_keypoints, test_image, test_keypoints, matches, None)
        plt.imshow(result_match_img)
        plt.savefig(result_path + '/' + t + '-' + r + '.jpg')
