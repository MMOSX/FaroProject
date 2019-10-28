#%% Import
import cv2
import numpy as np
#%% Parameter definition
# Paths
reference_images_path = 'BaseDocument/Data/templates/ic/ita/CIC/front.png'
test_image_path = 'FronteCI.jpg'
# Size of keypoints vector
vector_size = 32
#%% Open images
reference_image = cv2.imread(reference_images_path, cv2.IMREAD_COLOR)
test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
#%% Feature extraction
# Reference image elaboration
try:
    # Definition of the extractor
    kaze_extractor = cv2.KAZE_create()
    # KeyPoint extractor
    kaze_keypoint = kaze_extractor.detect(reference_image)
    # Getting first 32 of them.
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kaze_keypoint_sorted = sorted(kaze_keypoint, key=lambda x: -x.response)[:vector_size]
    # computing descriptors vector
    kaze_keypoint_sorted, descriptors_vector = kaze_extractor.compute(reference_image, kaze_keypoint_sorted)
    # Flatten all of them in one big vector - our feature vector
    descriptors_vector_to_one = descriptors_vector.flatten()
    # Check if we have 32 keypoints
    needed_size = (vector_size * 64)
    if descriptors_vector_to_one.size < needed_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        descriptors_vector_to_one = np.concatenate([descriptors_vector_to_one, np.zeros(needed_size -
                                                                                        descriptors_vector_to_one.size)])
except cv2.error as e:
    print('Error: ', e)
# Test image elaboration
try:
    # Definition of the extractor
    kaze_extractor = cv2.KAZE_create()
    # KeyPoint extractor
    kaze_keypoint = kaze_extractor.detect(test_image)
    # Getting first 32 of them.
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kaze_keypoint_sorted = sorted(kaze_keypoint, key=lambda x: -x.response)[:vector_size]
    # computing descriptors vector
    kaze_keypoint_sorted, descriptors_vector = kaze_extractor.compute(reference_image, kaze_keypoint_sorted)
    # Flatten all of them in one big vector - our feature vector
    descriptors_vector_to_one = descriptors_vector.flatten()
    # Check if we have 32 keypoints
    needed_size = (vector_size * 64)
    if descriptors_vector_to_one.size < needed_size:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        descriptors_vector_to_one = np.concatenate([descriptors_vector_to_one, np.zeros(needed_size -
                                                                                        descriptors_vector_to_one.size)])
except cv2.error as e:
    print('Error: ', e)

#%% Features Matcher
