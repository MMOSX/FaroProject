#%% Import Section
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
#%% Parameters
eps = 1e-7
#%% Create descriptor
sift = cv2.xfeatures2d.SIFT_create()
#%% #%% Parsing input
parser = argparse.ArgumentParser()
parser.add_argument('--reference_folder', help='Please provide the paths for the reference images', required=True)
parser.add_argument('--test_folder', help='Please provide the paths for the test images', required=True)
parser.add_argument('--result_folder', help='Please provide the paths for the result images', default='Results')
args = parser.parse_args()
#%% Check reference path exist
references_path = args.reference_folder
tests_path = args.test_folder
results_path = args.result_folder
if not os.path.isdir(references_path):
    print('The reference images folder does not a directory')
if not os.path.exists(references_path):
    print('The reference images folder does not exist')
if not os.path.isdir(tests_path):
    print('The test images folder does not a directory')
if not os.path.exists(tests_path):
    print('The test images folder does not exist')
if not os.path.isdir(results_path):
    print('The reference images folder does not exist.')
    print('Creating the Results folder')
    os.mkdir(results_path)
    print('Folder created')
#%% Starting reference creation
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

#%% Starting test creation
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

#%% Creating FLANN Matcher
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
for reference_name in reference_dictionaries.keys():
    reference_descriptors = reference_dictionaries[reference_name][2]
    reference_keypoints = reference_dictionaries[reference_name][1]
    reference_image = reference_dictionaries[reference_name][0]
    for test_name in test_dictionaries.keys():
        test_descriptors = test_dictionaries[test_name][2]
        test_keypoints = test_dictionaries[test_name][1]
        test_image = test_dictionaries[test_name][0]
        matches = flann.knnMatch(reference_descriptors, test_descriptors, k=2)
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        result_image = cv2.drawMatchesKnn(reference_image, reference_keypoints, test_image, test_keypoints, matches,
                                          None, **draw_params)
        plt.imshow(result_image, )
        plt.axis('off')
        plt.savefig(results_path + '/' + reference_name + test_name + '.jpg')
