'''
This file reproduce the faro sift use in python
'''
#%% Import section
import numpy as np
import cv2 as cv
#%% Im read
img = cv.imread('ReferenceImages/rearCIC.png')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#%% sift creator
sift = cv.xfeatures2d.SIFT_create()
#%% sift detection
kp = sift.detect(gray,None)
img = cv.drawKeypoints(gray, kp, img)

cv.imwrite('sift_keypoints.jpg', img)
