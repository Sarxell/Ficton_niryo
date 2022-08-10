#!/usr/bin/python3

import copy
import cv2
import numpy as np


def main():
    # initial setup
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 25)
    window_name = 'niryo camera'
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)

    # RED MASK
    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])

    # cv2.namedWindow(window_name,  cv2.WINDOW_AUTOSIZE)

    while 1:
        _, image = capture.read()   # get an image from the camera
        # image = cv2.resize(image, (960, 700))
        cv2.imshow(window_name, image)    # add code to show acquired image
        if cv2.waitKey(1) & 0xFF == 27:   # add code to wait for a key press
            break
        red_mask = image.copy()
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0 + mask1
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # mask = cv2.dilate(mask , kernel, iterations=2)

        # if we want to see the mask in red
        # red_mask = cv2.bitwise_and(red_mask, red_mask, mask=mask)

        cv2.imshow('mask', mask)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()