#%% Import Section
import numpy as np
from skimage.io import imread
from skimage.measure import moments, moments_hu, find_contours, approximate_polygon
from skimage.feature import canny
from matplotlib import pyplot as plt
#%% Opening image section
reference_image = imread('ReferenceImages/rearCIC.jpg', as_gray= True)
test_image = imread('TestImages/rear/a-petrelli-rearCIC.jpg', as_gray= True)
#%% Contours finding:
reference_image_canny = reference_image#canny(reference_image, 3)
test_image_canny = test_image#canny(test_image, 3)
reference_contours = find_contours(reference_image_canny, 0.8, fully_connected='high')
test_contours = find_contours(test_image_canny, 0.8, fully_connected='high')
#%%
print('Reference Contours: ', reference_contours)
print('Test Contours: ', test_contours)
#%% Find moment
reference_moments = moments(reference_contours[0])
test_moments = moments(test_contours[0])
reference_hu_moments = moments_hu(reference_moments)
test_hu_moments = moments_hu(test_moments)
#%%
print(reference_hu_moments)
print(test_hu_moments)
#%%

#%%
# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(reference_image, cmap=plt.cm.gray)

#for n, contour in enumerate(reference_contours[0]):
ax.plot(reference_contours[0][:, 1], reference_contours[0][:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

fig, ax = plt.subplots()
ax.imshow(test_image, cmap=plt.cm.gray)

#for n, contour in enumerate(reference_contours[0]):
ax.plot(test_contours[0][:, 1], test_contours[0][:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

fig, ax = plt.subplots()
ax.imshow(test_image, cmap=plt.cm.gray)

#for n, contour in enumerate(reference_contours[0]):
# for n, contour in enumerate(test_contours):
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     print(n)
#     if n == 0:
#         break

ax.plot(test_contours[0][:,1], test_contours[0][:,0], linewidth=3)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
#%% Polygon writing:
fig, ax = plt.subplots(ncols=1, figsize=(10, 10))


ax.imshow(test_image)

# approximate / simplify coordinates of the two ellipses
# for contour in test_contours:
#     coords = approximate_polygon(contour, tolerance=2.5)
#     ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
#
#     print("Number of coordinates:", len(contour), len(coords))

ax.plot(test_contours[0][:,1], test_contours[0][:,0], linewidth=8)

ax.axis('off')

plt.show()
#%%
first = True
added = False
reference_contours_list = []
for contours in reference_contours:
    for con in contours:
        if first:
            first = False
            reference_contours_list.append(np.array(np.array([con])))
        else:
            for index, past_con in enumerate(reference_contours_list):
                for past_c in past_con:
                    if con[0] in range(int(round(past_c[0])) - 1, int(round(past_c[0])) + 2):
                        reference_contours_list[index] = np.append(reference_contours_list[index], np.array(np.array([con])), axis=0)
