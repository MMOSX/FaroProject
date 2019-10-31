import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv.imread('ReferenceImages/TestaCIC.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('TestImages/front/a-petrelli-frontCIC.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
orb = cv.ORB_create()
fast = cv.FastFeatureDetector_create()
# find the keypoints and descriptors with SIFT
# find the keypoints with ORB
kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

des1 = orb.compute(img1,kp1)
des2 = orb.compute(img2,kp2)
bf_matcher = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck= False)
matches = bf_matcher.match(des1, des2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)