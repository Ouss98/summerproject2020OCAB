import cv2
import numpy as np
import glob
import imutils
import os


def getSize(image):
    height, width, layers = image.shape
    size = (width,height)
    return size
 
img_array = []


folder = os.listdir(r'C:\Users\Oussama\Desktop\python_tuto\opencv\savedFrames')
sortedFolder = sorted(folder,key=lambda x: int(os.path.splitext(x)[0]), reverse=True)


biggestImage = cv2.imread('./savedFrames/0.jpg')
size = getSize(biggestImage)

for filename in sortedFolder:
    img = cv2.imread('./savedFrames/' + filename)
    if (getSize(img) != size):
        img = cv2.resize(img, size)
    img_array.append(img)
 
 
video = cv2.VideoWriter('cropped_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(img_array)):
    video.write(img_array[i])
    
cv2.destroyAllWindows()
video.release()