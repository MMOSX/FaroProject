#%% Import Section
import cv2
import numpy as np
from matplotlib import pyplot as plt
#%% parameters
eps = 1e-7
#%% path section
tests_paths = 'TestImages/rear/'
references_paths = 'ReferenceImages/'
reference_image_path = references_paths + 'rearCIC.png'
test_image_path = tests_paths + 't-pantaloni-rearCIE.jpg'
#%% SIFT init
sift = cv2.xfeatures2d.SIFT_create()
#%% Opening image
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
#%% SIFT detection
reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
test_keypoints, test_descriptors = sift.detectAndCompute(test_image, None)
#%% RootSIFT
reference_descriptors /= (reference_descriptors.sum(axis=1, keepdims=True) + eps)
reference_descriptors = np.sqrt(reference_descriptors)
test_descriptors /= (test_descriptors.sum(axis=1, keepdims=True) + eps)
test_descriptors = np.sqrt(test_descriptors)
#%% Matcher
bf_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)
#%% Matching
matches = bf_matcher.knnMatch(reference_descriptors, test_descriptors, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(reference_image, reference_keypoints, test_image, test_keypoints, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3), plt.show()
