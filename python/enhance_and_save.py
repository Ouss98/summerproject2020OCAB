import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from skimage import color, restoration
from scipy.signal import convolve2d as conv2

def deblurUnsupWiener(img):
	image = color.rgb2gray(img)
	psf = np.ones((2, 2)) / 7
	image += 0.01 * image.std() * np.random.standard_normal(image.shape)
	correctedImg, _ = restoration.unsupervised_wiener(image, psf)
	return correctedImg

folder = os.listdir(r'C:\Users\Oussama\Desktop\python_tuto\opencv\savedFrames')
sortedFolder = sorted(folder,key=lambda x: int(os.path.splitext(x)[0]), reverse=True)

for filename in sortedFolder:
    img = cv2.imread('./savedFrames/' + filename)
    img = deblurUnsupWiener(img)
    plt.axis('off')
    plt.imshow(img)
    plt.imsave('./savedFrames/' + filename, img, cmap='gray')