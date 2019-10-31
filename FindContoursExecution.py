'''
This code use canny edge detector and find contours to generate an image that contain the border remarked
'''
#%%
import os
from skimage.feature import canny
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import measure
#%% Parameter section
# Path subsection
reference_images_path = 'ReferenceImages/'
test_images_path = 'TestImages/'
# Constant subsection
contour_constant = 0.8
#%% Construction of reference dictionary
print('Generating Reference Contours')
reference_images_edges = {}
for root, directories, files in os.walk(reference_images_path):
    for file in files:
        if file.endswith('.DS_Store'):
            continue
        else:
            reference_image_name = file.split('.')[0]
            reference_image_path = os.path.join(root, file)
            reference_image = imread(reference_image_path, as_gray= True)
            reference_image_edges = canny(reference_image)
            reference_images_edges[reference_image_name] = {
                'image': reference_image_edges,
                'contours': measure.find_contours(reference_image_edges, contour_constant)
            }

#%% Construction of test dictionary
print('Generating Test Contours')
test_images_edges = {}
for root, directories, files in os.walk(test_images_path):
    for file in files:
        if file.endswith('.DS_Store'):
            continue
        else:
            test_image_name = file.split('.')[0]
            test_image_path = os.path.join(root, file)
            test_image = imread(test_image_path, as_gray= True)
            test_image_edges = canny(test_image)
            test_images_edges[test_image_name] = {
                'image': test_image_edges,
                'contours': measure.find_contours(test_image_edges, contour_constant)
            }

#%% Showing and Saving section
# Display the test image and plot all contours found
print('Saving Reference Contours')
for reference_name in reference_images_edges.keys():
    reference_edges = reference_images_edges[reference_name]
    # Create figure
    fig, ax = plt.subplots()
    # show figure
    ax.imshow(reference_edges['image'])
    # create contours on image
    for n, contour in enumerate(reference_edges['contours']):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.show()
    plt.imsave('ResultContours/Edges_of-' + reference_name + '.png', reference_edges['image'])
#%%
# Display the test image and plot all contours found
print('Saving Test Contours')
for test_name in test_images_edges.keys():
    test_edges = test_images_edges[test_name]
    # Create figure
    fig, ax = plt.subplots()
    # show figure
    ax.imshow(test_edges['image'])
    # create contours on image
    for n, contour in enumerate(test_edges['contours']):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.show()
    plt.imsave('ResultContours/Edges_of-' + test_name + '.png', test_edges['image'])