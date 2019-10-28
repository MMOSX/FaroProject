#%% Import section
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import skimage.feature as features
from skimage.io import imread
#%% Parameter section
# settings for LBP
METHOD = 'uniform'
P = 16
R = 2
matplotlib.rcParams['font.size'] = 9
# Parameters setting
reference_image_path = 'BaseDocument/Data/templates/ic/ita/CIC/front.png'
test_image_path = 'FronteCI.jpg'
# Opening image
reference_image = imread(reference_image_path, as_gray= True)
test_image = imread(test_image_path, as_gray= True)

#%% Function definition
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(refs, img):
    best_score = 10
    best_name = None
    lbp = features.local_binary_pattern(img, P, R, METHOD)
    hist, _ = np.histogram(lbp, density=True, bins=P + 2, range=(0, P + 2))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=P + 2,
                                   range=(0, P + 2))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return lbp, best_name, best_score

#%% features extraction
reference_lbp = features.local_binary_pattern(reference_image, P, R, METHOD)
refs = {
    'CI': features.local_binary_pattern(reference_image, P, R, METHOD),
}
#%% Result
# classify rotated textures
print('Result references using LBP:')
print('Test_image with no rotation')
test_lbp, name, score = match(refs, test_image)
print(name, score)
print('Test_image with 70° rotation')
test_lbp_rotate70, name_rotate70, score_rotate70 = match(refs, nd.rotate(test_image, angle=70, reshape=False))
print(name_rotate70, score_rotate70)
print('Test_image with 145° rotation')
test_lbp_rotate145, name_rotate145, score_rotate145 = match(refs, nd.rotate(test_image, angle=145, reshape=False))
print(name_rotate145, score_rotate145)
print('Test_image with 90° rotation')
test_lbp_rotate90, name_rotate90, score_rotate90 = match(refs, nd.rotate(test_image, angle=90, reshape=False))
print(name_rotate90, score_rotate90)
#%% plotting result
# plot histograms of LBP of textures
fig, ((ax1, ax2, ax3, ax4, ax9), (ax5, ax6, ax7, ax8, ax10)) = plt.subplots(nrows=2, ncols=5, figsize=(9, 6))
plt.gray()

ax1.imshow(reference_image)
ax1.axis('off')
ax5.hist(reference_lbp.ravel(), density=True, bins=P + 2, range=(0, P + 2))
#ax5.set_ylabel('Percentage')

ax2.imshow(test_image)
ax2.axis('off')
ax6.hist(test_lbp.ravel(), density=True, bins=P + 2, range=(0, P + 2))
#ax6.set_ylabel('Percentage')

ax3.imshow(nd.rotate(test_image, angle=70, reshape=False))
ax3.axis('off')
ax7.hist(test_lbp_rotate70.ravel(), density=True, bins=P + 2, range=(0, P + 2))
#ax7.set_xlabel('Uniform LBP values')

ax4.imshow(nd.rotate(test_image, angle=145, reshape=False))
ax4.axis('off')
ax8.hist(test_lbp_rotate145.ravel(), density=True, bins=P + 2, range=(0, P + 2))

ax9.imshow(nd.rotate(test_image, angle=90, reshape=False))
ax9.axis('off')
ax10.hist(test_lbp_rotate90.ravel(), density=True, bins=P + 2, range=(0, P + 2))

plt.show()