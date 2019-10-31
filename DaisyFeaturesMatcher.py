from skimage.feature import daisy, match_descriptors, plot_matches, canny
from skimage.filters import sobel, prewitt, roberts
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
# ANNOTATION canny non funziona
reference_img = imread('ReferenceImages/rearCIC.png', as_gray=True)
reference_img_edges = canny(reference_img)
test_img = imread('TestImages/a-petrelli-rearCI.jpg', as_gray=True)
test_img_edges = canny(test_img)
test_img = resize(test_img, reference_img.shape, anti_aliasing=True)
reference_descs, reference_descs_img = daisy(reference_img, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)
test_descs, test_descs_img = daisy(test_img, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)

#matches = match_descriptors(reference_descs, test_descs, cross_check=True)

fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3)
ax1.axis('off')
ax1.imshow(reference_descs_img)
reference_descs_num = reference_descs.shape[0] * reference_descs.shape[1]
ax1.set_title('%i DAISY descriptors extracted:' % reference_descs_num)
ax2.axis('off')
ax2.imshow(test_descs_img)
test_descs_num = test_descs.shape[0] * test_descs.shape[1]
ax2.set_title('%i DAISY descriptors extracted:' % test_descs_num)
ax3.imshow(test_img_edges)
# #test_descs_num = test_descs.shape[0] * test_descs.shape[1]
ax3.set_title('Edge Result')
ax3.axis('off')
plt.show()

# fig2, ax = plt.subplots(nrows=2, ncols=1)
#
# plt.gray()
#
# plot_matches(ax[0], reference_img, test_img, reference_descs, test_descs, matches)
# ax[0].axis('off')
# ax[0].set_title("Original Image vs. Transformed Image")
#
# plt.show()

fg = plt.figure()
plt.imshow(test_img_edges)
plt.axis('off')
plt.show()
plt.imsave('ResultCannu' + 'rearCIC' + '.png', reference_img_edges)
plt.imsave('ResultCanny' + 'a-petrelli-rearCI' + '.png', test_img_edges)