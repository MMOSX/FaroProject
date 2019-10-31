import cv2, sys, os
from math import copysign, log10
#%% Opening Section
reference_image = cv2.imread('ReferenceImages/TestaCIC.png')
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
_, reference_imageim = cv2.threshold(reference_image, 128, 255, cv2.THRESH_BINARY)
test_image = cv2.imread('TestImages/rear/a-petrelli-frontCIC.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
_, test_image = cv2.threshold(test_image, 128, 255, cv2.THRESH_BINARY)
#%% Reference Hu Moment calculation
# Calculate Moments
reference_moment = cv2.moments(reference_image)
# Calculate Hu Moments
reference_huMoments = cv2.HuMoments(reference_moment)
# Log scale hu moments
for i in range(0,7):
  reference_huMoments[i] = -1 * copysign(1.0, reference_huMoments[i]) * log10(abs(reference_huMoments[i]))
#%% Test Hu Moment Calculation
# Calculate Moments
test_moment = cv2.moments(test_image)
# Calculate Hu Moments
test_huMoments = cv2.HuMoments(test_moment)
for i in range(0,7):
  test_huMoments[i] = -1 * copysign(1.0, test_huMoments[i]) * log10(abs(test_huMoments[i]))

#%%
