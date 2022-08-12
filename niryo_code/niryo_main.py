#!/usr/bin/env python

import numpy as np
from niryo_one_python_api.niryo_one_api import *
import rospy
import cv2
import time

rospy.init_node('niryo_one_example_python_api')

n = NiryoOne()

# You should replace these 3 lines with the output in calibration step
DIM = (1600, 1200)
K = np.array([[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
D = np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])
font = cv2.FONT_HERSHEY_SIMPLEX

def main():
    try:

        # CAMERA SETTINGS AND MASKS
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FPS, 25)

        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((5, 5), np.uint8)

        # RED MASK

        # upper mask (170-180)
        # lower_red = np.array([170, 50, 50])
        lower_red = np.array([170, 90, 90])
        upper_red = np.array([180, 255, 255])
        # --------------------------------------

        # INITIAL ROBOT POSE
        # Move
        n.set_arm_max_velocity(30)

        initial_joints = [0, 0, -0.465, -0.078, -0.969, -0.05]
        n.move_joints(initial_joints)
        # --------------------

        while 1:
            _, frame = capture.read()  # get an image from the camera
            image = cv2.rotate(frame, cv2.ROTATE_180)
            image = cv2.resize(image, (1600, 1200))

            #  UNDISTORTING IMAGE
            h, w = image.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imshow("fisheye image", image)
            cv2.imshow("undistorted", undistorted_img)
            # ----------------------------

            if cv2.waitKey(1) & 0xFF == 27:   # add code to wait for a key press
                break

            # CREATING THE MASKS
            img_hsv = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(img_hsv, lower_red, upper_red)

            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)
            # mask = cv2.morphologyEx(mask ,cv2.MORPH_OPEN,kernel)
            mask, centroids = removeSmallAndBigComponents(mask, 4000, 20000, (0, 0, 255))


        n.activate_learning_mode(True)

    except NiryoOneException as e:
        print e
        n.activate_learning_mode(True)
        # Handle errors here


def removeSmallAndBigComponents(image, threshold_min, threshold_max, color):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    h = 90
    w = 90
    x = None
    y = None
    img2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold_min:
            if sizes[i] <= threshold_max:
                # to use the biggest
                x, y = centroids[i+1]
                y, x = int(y), int(x)
                # only needed if we want the biggest one
                # threshold = sizes[i]
                img2[output == i + 1] = (255, 255, 255)
                # img2[int(x):int(x)+10, int(y):int(y)+10] = (0, 0, 255)
                cv2.circle(img2, (x, y), 5, color, -1)
                cv2.rectangle(img2, (x-h, y-w), (x + h, y + w), (0, 0, 255), 2)
                cv2.putText(img2, '(' + str(x) + ',' + str(y) + ')', (x + 50, y ), font, 3, (0, 255, 0), 2, cv2.LINE_AA)


    return img2, centroids

if __name__ == '__main__':
    main()