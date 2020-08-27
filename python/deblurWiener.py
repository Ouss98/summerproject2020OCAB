import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

blurry_image = cv2.imread('./peugeot-208.jpg')
img = color.rgb2gray(blurry_image) # RGB to Grayscale

# Blurring of the image
psf = np.ones((5, 5)) / 25 # Point Spread Function
img = conv2(img, psf, 'same') # 2D convolution --> blurry image
# Add noise to image
img += 0.1 * img.std() * np.random.standard_normal(img.shape) 
# Deconvolve - Restore image
deconvolved, _ = restoration.unsupervised_wiener(img, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)
plt.gray()




ax[0].imshow(img, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()