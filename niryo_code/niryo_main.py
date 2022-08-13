#!/usr/bin/env python

import numpy as np
from niryo_one_python_api.niryo_one_api import *
import rospy
import cv2
import time

rospy.init_node('niryo_one_example_python_api')

n = NiryoOne()

# You should replace these 3 lines with the output in calibration step
DIM = (1920, 1080)
K = np.array([[868.759915571659, 0.0, 903.564038832843], [0.0, 875.558509058017, 529.922725992597], [0.0, 0.0, 1.0]])
D = np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])
# D = np.array([[-0.2633], [0.0397], [0], [0]])
font = cv2.FONT_HERSHEY_SIMPLEX
h = 50
w = 50
M_ext = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

def main():
    try:

        # CAMERA SETTINGS AND MASKS
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FPS, 25)

        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((5, 5), np.uint8)

        # RED MASK

        # RED MASK
        # lower mask (0-10)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        # upper mask (170-180)
        # lower_red = np.array([170, 50, 50])
        lower_red = np.array([150, 90, 90])
        upper_red = np.array([180, 255, 255])
        # --------------------------------------

        # INITIAL ROBOT POSE
        # Move
        n.set_arm_max_velocity(30)
        n.change_tool(TOOL_GRIPPER_1_ID)
        initial_joints = [0, 0, -0.465, -0.078, -0.969, -0.05]
        n.move_joints(initial_joints)
        arm_pose= n.get_arm_pose()
        # --------------------

        while 1:
            _, frame = capture.read()  # get an image from the camera
            image = cv2.rotate(frame, cv2.ROTATE_180)
            image = cv2.resize(image, (1920, 1080))

            #  UNDISTORTING IMAGE
            h, w = image.shape[:2]
            camera_intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3))
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
            undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # cv2.imshow("fisheye image", image)
            # cv2.imshow("undistorted", undistorted_img)
            # ----------------------------
            #undistorted_img = cv2.resize(undistorted_img, (640, 480))
            if cv2.waitKey(1) & 0xFF == 27:   # add code to wait for a key press
                break

            # red mask creation of the undistorted img!!
            img_hsv = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2HSV)

            mask0 = cv2.inRange(img_hsv, lower_red1, upper_red1)

            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            # join my masks
            mask = mask0 + mask1
            # mask = mask1
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)
            # mask = cv2.morphologyEx(mask ,cv2.MORPH_OPEN,kernel)
            mask, x, y = removeSmallAndBigComponents(mask, 1300, 10000, (0, 0, 255))
            cv2.imshow('mask', mask)
            if x is not None:
                pixel = np.array([x, y, 1]).transpose()
                A = np.dot(np.linalg.inv(camera_intrinsic), pixel).transpose()
                world_points = np.dot(M_ext, A - [-0.99933767, - 0.69325907, -0.095])
                world_coord = [world_points[0]/5.1, world_points[1] / 3.218]
                n.open_gripper(TOOL_GRIPPER_1_ID, 500)
                n.move_pose(0.093 + world_coord[1], -0.323 + world_coord[0], arm_pose.position.z - 0.08, arm_pose.rpy.roll, arm_pose.rpy.pitch, arm_pose.rpy.yaw)
                n.close_gripper(TOOL_GRIPPER_1_ID, 500)
                print(world_coord)
                n.move_pose(-0.1, -0.2, arm_pose.position.z , arm_pose.rpy.roll, arm_pose.rpy.pitch, arm_pose.rpy.yaw)
                n.open_gripper(TOOL_GRIPPER_1_ID, 500)
                n.move_joints(initial_joints)
                n.open_gripper(TOOL_GRIPPER_1_ID, 500)
                break




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
    h = 50
    w = 50
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

    return img2, x,y

if __name__ == '__main__':
    main()