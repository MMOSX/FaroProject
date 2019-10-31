#%%
import numpy as np
import cv2
#%%
reference_image = cv2.imread('ReferenceImages/rearCIC.png')
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
reference_image_s, reference_thre = cv2.threshold(reference_image, 100, 255,0)
reference_image = cv2.Canny(reference_thre, 100, 200)
# cv2.imshow("store gray iamge",reference_thre)
# cv2.waitKey(0)
#%%
reference_contours, reference_hierarchy = cv2.findContours(reference_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
temp_cont = sorted(reference_contours, key=cv2.contourArea, reverse= True)
new_temp_cont = temp_cont[1]


#%%
test_image = cv2.imread('TestImages/rear/a-petrelli-rearCIC.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image_s, test_thre = cv2.threshold(test_image, 127, 255, 0)
test_image = cv2.Canny(test_thre, 100, 200)
test_contours, test_hierarchy = cv2.findContours(test_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
test_temp_cont = sorted(test_contours, key=cv2.contourArea, reverse= True)

#%%
for c in test_temp_cont:
    match = cv2.matchShapes(new_temp_cont, c, 1, 0.0)
    print(match)
    if match < 0.15:
        closest_match = c
    else:
        closest_match = []
#%%
cv2.drawContours(reference_image, reference_contours[0], 0, (0,255,0),2)
cv2.imshow("Reference Image", reference_image)
cv2.drawContours(test_image, test_temp_cont[0], 0, (0,255,0),2)
cv2.imshow("Test Image", test_image)
cv2.drawContours(test_image, closest_match, -1, (0,255,0),2)
cv2.imshow("output", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()