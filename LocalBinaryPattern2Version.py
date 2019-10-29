#%% Import section
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import skimage.feature as features
from skimage.io import imread
import math
import os
#%% Parameter section
# settings for LBP
METHOD = 'uniform'
P = 16
R = 2
matplotlib.rcParams['font.size'] = 9
# Parameters setting
reference_images_path = 'ReferenceImages'
test_images_path = 'TestImages/'
angles = [0, 45, 60, 70, 90, 115, 145, 285]

#%% Function definition
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def bhattacharyya(a, b):
    """ Bhattacharyya distance between distributions (lists of floats). """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))

def match(refs, img):
    best_score = 10
    best_name = None
    lbp = features.local_binary_pattern(img, P, R, METHOD)
    hist, _ = np.histogram(lbp, density=True, bins=P + 2, range=(0, P + 2))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=P + 2,
                                   range=(0, P + 2))
        score = kullback_leibler_divergence(hist, ref_hist)
        result_hist_comparison = bhattacharyya(ref_hist, hist)
        if result_hist_comparison < best_score:
            best_score = score
            best_name = name
    return lbp, best_name, result_hist_comparison

#%% features extraction
reference_images_lbp = {}
for root, directories, files in os.walk(reference_images_path):
    for file in files:
        if file.endswith('.json') or file.endswith('.DS_Store'):
            continue
        else:
            reference_image_path = os.path.join(root,file)
            reference_image = imread(reference_image_path, as_gray= True)
            reference_name = file.split('.')[0]
            reference_images_lbp[reference_image_path] = features.local_binary_pattern(reference_image, P, R, METHOD)
#%% Test Extractor and matches
result_test_images = {}
for root, directories, files in os.walk(test_images_path):
    for file in files:
        test_image_path = os.path.join(root,file)
        test_image = imread(test_image_path, as_gray= True)
        test_name = file.split('.')[0]
        for angle in angles:
             test_lbp, name, score = match(reference_images_lbp, nd.rotate(test_image, angle=angle, reshape=False))
             result_test_images[test_image_path + '-' + str(angle)] = {'lbp': test_lbp,
                                              'angle': angle,
                                            'name_matched': name,
                                            'score_matched': score}
#%% Result
# classify rotated textures
print('Result of matches using LBP:')
for name in result_test_images.keys():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    plt.gray()
    result = result_test_images[name]
    name = name.split('-')
    test_image = imread(name[0], as_gray= True)
    reference_image = imread(result['name_matched'], as_gray= True)
    ax1.imshow(reference_image)
    ax2.imshow(test_image)
    ax3.hist(reference_images_lbp[result['name_matched']].ravel(), density=True, bins=P + 2, range=(0, P + 2))
    ax4.hist(result['lbp'].ravel(), density=True, bins=P + 2, range=(0, P + 2))
    plt.savefig('ResultMatched/Angle_' + name[1] + '-Test_' + name[0].split('/')[1].split('.')[0] + '-Reference_' +
                result['name_matched'].split('/')[1].split('.')[0] + '.pdf')
    plt.show()
    print('For test image: ' + name[0] + ' with angle: ' + str(result['angle']) +
          ' the matched reference image is: ' + result['name_matched'] +
          ' with score: ' + str(result['score_matched']))
