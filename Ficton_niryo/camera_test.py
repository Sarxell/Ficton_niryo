#!/usr/bin/python3

#!/usr/bin/env python
import cv2
from imutils.video import FPS

def main():
    # initial setup
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 30)
    window_name = 'niryo camera'
    # cv2.namedWindow(window_name,  cv2.WINDOW_AUTOSIZE)

    while 1:
        _, image = capture.read()   # get an image from the camera
        image = cv2.resize(image, (960, 700))
        cv2.imshow(window_name, image)    # add code to show acquired image
        if cv2.waitKey(1) & 0xFF == 27:   # add code to wait for a key press
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()