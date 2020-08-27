import cv2
import imutils
from tkinter import *
from tkinter import messagebox

class staticROI(object):

    def __init__(self):
        self.capture = cv2.VideoCapture('front_coming.mp4') # Capture video

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False

        self.update()

    def update(self):
        while True:
            if self.capture.isOpened(): 

                # Read frame
                (self.status, self.frame) = self.capture.read()
                # Resize video
                self.frame = imutils.resize(self.frame, width=500)
                # Show video
                cv2.imshow('video', self.frame)
                key = cv2.waitKey(25)

                # Crop image
                if key == ord('c'):
                    self.clone = self.frame.copy()
                    cv2.namedWindow('video')
                    cv2.setMouseCallback('video', self.extract_coordinates)
                    while True:
                        key = cv2.waitKey(2)
                        cv2.imshow('video', self.clone)

                        # Crop and display cropped image
                        if key == ord('c'):
                            self.crop_ROI()
                            self.show_cropped_ROI()

                        # Save cropped image
                        try:
                            if key == ord('s') and self.show_cropped_ROI() == "cropped":
                                cv2.imwrite('cropped_video_img.jpg', self.cropped_image)
                            pass
                        # Handle AttributeError exception
                        except AttributeError:
                            messagebox.showerror("Error", "Select a region")
                            pass

                        # Resume video
                        if key == ord('r'):
                            break

                        # Close program with keyboard 'q'
                        if key == ord('q'):
                            cv2.destroyAllWindows()
                            exit(1)

                # Close program with keyboard 'q'
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(1)
            else:
                pass

    def extract_coordinates(self, event, x, y, flags, parameters):

        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True

        # Record ending (x,y) coordinates on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False

            self.selected_ROI = True

            # Draw rectangle around ROI
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.frame.copy()
            self.selected_ROI = False

    def crop_ROI(self):

        if self.selected_ROI:
            self.cropped_image = self.frame.copy()

            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]

            self.cropped_image = self.cropped_image[y1:y2, x1:x2]

            print('Cropped image: {} {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
        else:
            print('Select ROI to crop before cropping')

    def show_cropped_ROI(self):

        cv2.imshow('cropped image', self.cropped_image)
        return "cropped"

if __name__ == '__main__':
    App = Tk()
    App.withdraw()

    static_ROI = staticROI()
    
    App.mainloop()
