from skimage.io import imread
from skimage.feature import (match_descriptors, corner_peaks, corner_harris, plot_matches, BRIEF)
import matplotlib.pyplot as plt

reference_img = imread('ReferenceImages/rearCIC.png', as_gray=True)
test_img = imread('TestImages/a-petrelli-rearCI.jpg', as_gray=True)

reference_kpts = corner_peaks(corner_harris(reference_img), min_distance=5)
test_kpts = corner_peaks(corner_harris(test_img), min_distance=5)

extractor = BRIEF()

extractor.extract(reference_img, reference_kpts)
reference_kpts = reference_kpts[extractor.mask]
reference_dsc = extractor.descriptors

extractor.extract(test_img, test_kpts)
test_kpts = test_kpts[extractor.mask]
test_dsc = extractor.descriptors

matches12 = match_descriptors(reference_dsc, test_dsc, cross_check=True)

fig, ax = plt.subplots(nrows=1, ncols=1)

plt.gray()

plot_matches(ax, reference_img, test_img, reference_kpts, test_kpts, matches12)
ax.axis('off')
ax.set_title("Original Image vs. Transformed Image")

plt.show()