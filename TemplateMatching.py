import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.feature import match_template


reference_img = imread('ReferenceImages/rearCIC.png', as_gray=True)
test_img = imread('TestImages/a-petrelli-rearCI.jpg', as_gray=True)
test_img = resize(test_img, (reference_img.shape[0] * 2, reference_img.shape[1] * 2), anti_aliasing=True)

result = match_template(test_img, reference_img)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

ax1.imshow(reference_img, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(test_img, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
hcoin, wcoin = reference_img.shape
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()

imsave('reference_imageTemplate.png', reference_img)
imsave('Test_imageTemplate.png', test_img)
imsave('Result.png', result)