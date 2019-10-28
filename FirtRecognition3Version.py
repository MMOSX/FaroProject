#%%
import cv2 as cv
import numpy as np
import argparse
from math import sqrt
#%%
# parser = argparse.ArgumentParser(description='Code for AKAZE local features matching tutorial.')
# parser.add_argument('--input1', help='Path to input image 1.', default='BaseDocument/Data/templates/ic/ita/CIC/front.png')
# parser.add_argument('--input2', help='Path to input image 2.', default='FronteCI.jpg')
# #parser.add_argument('--homography', help='Path to the homography matrix.', default='H1to3p.xml')
# args = parser.parse_args()
img1 = cv.imread(cv.samples.findFile('BaseDocument/Data/templates/ic/ita/CIC/front.png'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(cv.samples.findFile('FronteCI.jpg'), cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
#fs = cv.FileStorage(cv.samples.findFile(args.homography), cv.FILE_STORAGE_READ)
#homography = fs.getFirstTopLevelNode().mat()
akaze = cv.AKAZE_create()
kpts1, desc1 = akaze.detectAndCompute(img1, None)
kpts2, desc2 = akaze.detectAndCompute(img2, None)
# Match the features
bf = cv.BFMatcher(cv.NORM_HAMMING)
matches = bf.knnMatch(desc1,desc2, k=2)    # typo fixed

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
im3 = cv.drawMatchesKnn(img1, kpts1, img2, kpts2, good[1:20], None, flags=2)
cv.imshow("AKAZE matching", im3)
cv.waitKey(0)