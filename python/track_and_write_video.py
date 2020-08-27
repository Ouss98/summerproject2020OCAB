from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
import os

from skimage import color, restoration
from scipy.signal import convolve2d as conv2

def croppedRect(roi, x1, x2, y1, y2 ):
	roi = roi[y1:y2, x1:x2]
	return roi

watch_cascade = cv2.CascadeClassifier('../plate_detect_devto/cascade.xml')
def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
        if top_bottom_padding_rate>0.2:
            print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
            exit(1)
        height = image_gray.shape[0]
        padding = int(height*top_bottom_padding_rate)
        scale = image_gray.shape[1]/float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
        image_color_detected = image[padding:resize_h-padding,0:image_gray.shape[1]]
        image_gray = cv2.cvtColor(image_color_detected,cv2.COLOR_RGB2GRAY)
        watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
        
        for (x, y, w, h) in watches:

            x -= w * 0.14
            w += w * 0.28
            y -= h * 0.15
            h += h * 0.3

            cv2.rectangle(image_color_detected, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

            image = image_color_detected
            cv2.waitKey(0)
        return image

def illumCorrection(img):
	a = np.double(img)
	b = a + 15
	correctedImg = np.uint8(b) #unsigned integer (0 to 255)
	return correctedImg

def deblurUnsupWiener(img):
	image = color.rgb2gray(img)
	psf = np.ones((5, 5)) / 25
	image = conv2(image, psf, 'same')
	image += 0.1 * image.std() * np.random.standard_normal(image.shape)
	correctedImg, _ = restoration.unsupervised_wiener(image, psf)
	return correctedImg


myDir = r'C:\Users\Oussama\Desktop\python_tuto\opencv\savedFrames'
filelist = [ f for f in os.listdir(myDir) if f.endswith(".jpg") ]
for f in filelist:
    os.remove(os.path.join(myDir, f))

# Parse two command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# Handle the different types of trackers
# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# If we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())

# Otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	# Initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}
	# Grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# Initialize bounding box coordinates of the object we are going to track
initBB = None

# If a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src = 0).start()
	time.sleep(1.0)

# Otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# Initialize the FPS throughput estimator
fps = None

frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

# Loop over frames from the video stream
while True:
	# Grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# Check to see if we have reached the end of the stream
	if frame is None:
		break
	# Resize the frame (so we can process it faster) and grab the
	# frame dimensions
	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]

	# Check to see if we are currently tracking an object
	if initBB is not None:
		# Grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		# Check to see if the tracking was a success
		# if success:
		(x, y, w, h) = [int(v) for v in box]
		rect = cv2.rectangle(frame, (x, y), (x + w, y + h),
			(0, 255, 0), 2)
			
		frame_number = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
		
		count = 0
		for frm in range (frame_number, frame_count):
			# correcImage = illumCorrection(croppedRect(rect, x, x + w, y, y + h))
			correcImage = croppedRect(rect, x, x + w, y, y + h)
			cv2.imwrite('./savedFrames/%d.jpg' % count, correcImage)
			# cv2.imwrite('./detectedPlates/%d.jpg' % count, detectPlateRough(correcImage,correcImage.shape[0],top_bottom_padding_rate=0.1))
			count += 1
		
	

		# Update the FPS counter
		fps.update()
		fps.stop()

		# Initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]

		# Loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# Show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	# count = 0
	# while success:
	# 	cv2.imwrite('./savedFrames/frame%d.jpg' % count, croppedRect(rect, x, x + w, y, y + h))
	# 	(success, box) = tracker.update(frame)
	# 	count += 1 

	# If the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("s"):
		# Select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker.init(frame, initBB)
		fps = FPS().start()

    	# If the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

    # If the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# If we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# Otherwise, release the file pointer
else:
	vs.release()

# Close all windows
cv2.destroyAllWindows()
