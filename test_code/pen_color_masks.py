#!/usr/bin/python3

import copy
import cv2
import numpy as np


def main():
    # initial setup

    window_name = 'Canetas'
    image = cv2.imread('Canetas.png')
    # image = cv2.resize(image, (960, 700))
    cv2.imshow(window_name, image)    # add code to show acquired image
    red_mask = image.copy()

    # MASKS
    lower_orange = np.array([10, 70, 70])
    upper_orange = np.array([20, 255, 255])

    lower_green = np.array([40, 70, 70])
    upper_green = np.array([70, 255, 255])

    lower_red = np.array([170, 70, 70])
    upper_red = np.array([180, 255, 255])

    lower_pink = np.array([145, 70, 70])
    upper_pink = np.array([170, 255, 255])

    lower_aqua = np.array([75, 70, 70])
    upper_aqua = np.array([100, 255, 255])

    lower_blue = np.array([105, 40, 40])
    upper_blue = np.array([145, 255, 255])


    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)
    mask_pink = cv2.inRange(img_hsv, lower_pink, upper_pink)
    mask_aqua = cv2.inRange(img_hsv, lower_aqua, upper_aqua)
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    cv2.imshow('mask_red', mask_red)
    cv2.imshow('mask_green', mask_green)
    cv2.imshow('mask_orange', mask_orange)
    cv2.imshow('mask_pink', mask_pink)
    cv2.imshow('mask_aqua', mask_aqua)
    cv2.imshow('mask_blue', mask_blue)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()