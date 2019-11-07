#%% Import section
import os
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
#%% function definition for numpy matrix minimun indexes research
def nd_min(array):
    '''

    :param array: numpy 2D array
    :return: the indexes of the matrix where the minimum is located
    '''
    #print(type(array))
    minimum_coordinate = np.where(array == np.amin(array))
    list_minimum_index = list(zip(minimum_coordinate[0],minimum_coordinate[1]))
    return list_minimum_index[0][0], list_minimum_index[0][1]


#%%
bluring_kernel_size = (3, 3) #TODO try different size that can be usefull
morphological_kernel = np.ones((5, 5), np.uint8) #TODO try different kernel that can be usefull (to search)
canny_low = 100
canny_high = 200
number_of_contours = 5
#%% sliding window parameter
stepSize = 50
(w_width, w_height) = (50, 50) # window size
#%% Test folder
tests_path = 'TestImages/rear/'
#%% Opening reference
references_path = 'ReferencePiece/rear/'
#%% Creating reference data
reference_data = {}
for root, directories, files in os.walk(references_path):
    for file in files:
        reference_name = file
        reference_path = os.path.join(root + file)
        reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        # Bluring
        reference_image = cv2.blur(reference_image, bluring_kernel_size)
        # Canny
        reference_image_canny = cv2.Canny(reference_image, canny_low, canny_high)
        # Morphological operation
        reference_image_canny_morph = cv2.morphologyEx(reference_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
        reference_contours, reference_hierarchy = cv2.findContours(reference_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # Rerder contours in descending mode to have the biggest area for first
        reference_contours = sorted(reference_contours, key=cv2.contourArea, reverse=True)
        # keep only first 10 contours
        reference_contours = reference_contours[0:number_of_contours]
        reference_data[reference_name] = reference_contours

#%%
test_data = {}
for root, directories, files in os.walk(tests_path):
    for file in files:
        if file.endswith('.DS_Store'):
            continue
        test_name = file
        test_path = os.path.join(root + file)
        test_image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        for x in range(0, test_image.shape[1] - w_width, stepSize):
            for y in range(0, test_image.shape[0] - w_height, stepSize):
                window = test_image[x:x + w_width, y:y + w_height]
                # Bluring
                test_image_b = cv2.blur(window, bluring_kernel_size)
                # Canny
                test_image_canny = cv2.Canny(test_image_b, canny_low, canny_high)
                # Morphological operation
                #test_image_canny_morph = cv2.morphologyEx(test_image_canny, cv2.MORPH_CLOSE, morphological_kernel)
                test_contours, test_hierarchy = cv2.findContours(test_image_canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                # Rerder contours in descending mode to have the biggest area for first
                test_contours = sorted(test_contours, key=cv2.contourArea, reverse=True)
                # keep only first 10 contours
                test_contours = test_contours[0:number_of_contours]
                if not test_name in test_data.keys():
                    test_data[test_name] = []
                test_data[test_name].append({
                    'x': x,
                    'y': y,
                    'contours': test_contours
                })
#%% Contours Matching
all_match_score = {}
# row -> reference
# column -> test
for test_name in test_data.keys():
    for test_name_contours in test_data[test_name]:
        test_x = test_name_contours['x']
        test_y = test_name_contours['y']
        test_contours = test_name_contours['contours']
        for reference_name in reference_data.keys():
            reference_contours = reference_data[reference_name]
            #print(len(test_contours))
            all_match_score[test_name + '-' + reference_name + '-X-' + str(test_x) + '-Y-' + str(test_y)] = \
                np.zeros((len(reference_contours), len(test_contours)))
            for i, ref_c in enumerate(reference_contours):
                for k, test_c in enumerate(test_contours):
                    all_match_score[test_name + '-' + reference_name + '-X-' + str(test_x) + '-Y-' + str(test_y)][i][k] \
                        = cv2.matchShapes(ref_c, test_c, 3, 0)
#%% Drawing contours
for match_name in all_match_score.keys():
    match_score = all_match_score[match_name]
    #print(type(match_score))
    if match_score.size == 0:
        continue
    ref_index, test_index = nd_min(match_score)
    key_list = match_name.split('-')
    test_name = key_list[0] + '-' + key_list[1] + '-' + key_list[2]
    reference_name = key_list[3]
    test_x = int(key_list[5])
    test_y = int(key_list[7])
    #TODO prendere gli indici dei minimi e utilizzare solo i contorni che sono minimi nel match, la funzione è stata definita sopra
    # TODO controlla che l'array che passi non sia vuoto con if shape0 != 0 and shape1 != 0:
    test_name_contours = test_data[test_name]
    reference_contours = reference_data[reference_name]
    reference_image = cv2.imread(references_path + reference_name, cv2.IMREAD_GRAYSCALE)
    test_contours = list(filter(lambda test: test['x'] == test_x and test['y'] == test_y, test_data[test_name]))[0]['contours'] # Filtra dentro una lista di dizionari
    if cv2.contourArea(test_contours[test_index]) < 250:
        continue
    test_complete_path = tests_path + test_name
    test_image = cv2.imread(tests_path + test_name, cv2.IMREAD_GRAYSCALE)
    test_piece = test_image[test_x:test_x + w_width, test_y:test_y + w_height]
    color = (0,0,0)#np.random.choice(range(256), size=3).tolist()
    rect = cv2.boundingRect(test_contours[test_index])
    x, y, w, h = rect
    cv2.rectangle(test_piece, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.drawContours(test_piece, test_contours[test_index], -1, color=color, thickness=2)
    test_image[test_x:test_x + w_width, test_y:test_y + w_height] = test_piece
    cv2.drawContours(reference_image, reference_contours[ref_index], -1, color=color, thickness=2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
    plt.gray()
    ax[0].imshow(reference_image)
    ax[0].axis('off')
    ax[0].set_xlabel('Reference piece')
    ax[1].imshow(test_image)
    ax[1].axis('off')
    ax[1].set_xlabel('Test Image')
    plt.savefig('SlidingWindowsResult/' + test_name + '-' + reference_name + '-X-' + str(test_x) + '-Y-' + str(test_y) + '.jpg')
    plt.close('all')
# %% Saving result
pd.Series(all_match_score).to_json('SlidingWindowsResult/slidingWindowsResult.json')