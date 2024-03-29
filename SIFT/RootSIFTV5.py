#%% Import Section
import cv2
import numpy as np
from matplotlib import pyplot as plt
#%% parameters
eps = 1e-7
#%% path section
tests_paths = 'TestImages/front/'
references_paths = 'ReferenceImages/'
reference_image_path = references_paths + 'frontCIC.png'
test_image_path = tests_paths + 't-pantaloni-frontCIC.jpg'
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
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(reference_descriptors, test_descriptors, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(reference_image, reference_keypoints, test_image, test_keypoints, matches, None, **draw_params)
plt.imshow(img3,)
plt.savefig('frontCIERootSIFResult.jpg'), plt.show()
matches
src_pts = np.float32([reference_keypoints[m.queryIdx].pt for i, (m, n) in enumerate(matches)]).reshape(-1, 1, 2)
dst_pts = np.float32([test_keypoints[m.trainIdx].pt for i, (m, n) in enumerate(matches)]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
matchesMask = mask.ravel().tolist()
h, w = reference_image.shape[:2]
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

dst = cv2.perspectiveTransform(pts, M)
dst += (w, 0)

test_perspective_image = cv2.warpPerspective(test_image, M, (test_image.shape[0], test_image.shape[1]))

# Draw bounding box in Red
img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

dst += (w, 0)  # adding offset
plt.imshow(img3)
plt.imshow(test_perspective_image)
plt.show()