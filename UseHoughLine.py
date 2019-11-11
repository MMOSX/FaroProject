#%% Import section
import cv2
import numpy as np
#%%
canny_low = 100
canny_high = 200
#%% Opening Reference Image
reference_image = cv2.imread('TestImages/rear/t-pantaloni-rearCIE.jpg', cv2.IMREAD_GRAYSCALE)
#%% Canny
reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
#%% HOUGH
lines = cv2.HoughLines(reference_image_canny, 1, np.pi/180, 200)
#%%
# The below for loop runs till r and theta values
# are in the range of the 2d array
for r, theta in lines[0]:
    # Stores the value of cos(theta) in a
    a = np.cos(theta)

    # Stores the value of sin(theta) in b
    b = np.sin(theta)

    # x0 stores the value rcos(theta)
    x0 = a * r

    # y0 stores the value rsin(theta)
    y0 = b * r

    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000 * (-b))

    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000 * (a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000 * (-b))

    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000 * (a))

    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(reference_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

# All the changes made in the input image are finally
# written on a new image houghlines.jpg
cv2.imwrite('linesDetected.jpg', reference_image)