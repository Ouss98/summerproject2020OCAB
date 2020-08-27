import numpy as np
import cv2

if __name__ == '__main__' :

    # Read image
    im = cv2.imread('peugeot-208.jpg')
    
    # Select ROI
    fromCenter = False
    r = cv2.selectROI('Image', im, fromCenter)
    
    # Define path
    path = 'cropped_img.jpg'

    # Crop image and save it
    if r != (0, 0, 0, 0):
        imgCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        cv2.imwrite(path, imgCrop)

    # Display cropped image
    cv2.imshow('Region of interest', imgCrop)
    cv2.waitKey(0)
