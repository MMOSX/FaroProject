#%% Import Section
import cv2
import numpy as np
from matplotlib import pyplot as plt
#%% Function definition
def draw_matches(img1, kp1, img2, kp2, matches, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)

    plt.figure(figsize=(15, 15))
    plt.imshow(new_img)
    plt.show()
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
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(reference_descriptors, test_descriptors, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.5 * n.distance:
        matchesMask[i]=[1, 0]
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)

#draw_matches(reference_image, reference_keypoints, test_image, test_descriptors, matches)

img3 = cv2.drawMatchesKnn(reference_image, reference_keypoints, test_image, test_keypoints, matches, None, **draw_params)
plt.imshow(img3,), plt.show()