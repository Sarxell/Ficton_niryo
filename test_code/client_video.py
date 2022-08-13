import cv2
import numpy as np
import socket
import sys
import pickle
import struct

# You should replace these 3 lines with the output in calibration step
DIM = (1600, 1200)
K = np.array([[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
D = np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])
font = cv2.FONT_HERSHEY_SIMPLEX

def main():
    cap=cv2.VideoCapture(0)
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect(('localhost', 8080))

    while True:
        _, image = cap.read()  # get an image from the camera
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.resize(image, (1600, 1200))

        #  Undistorting image!!
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # cv2.imshow("fisheye image", image)
        # cv2.imshow("undistorted", undistorted_img)
        # ----------------------------
        undistorted_img = cv2.resize(undistorted_img, (640, 480))
        if cv2.waitKey(1) & 0xFF == 27:  # add code to wait for a key press
            break

        # Serialize frame
        data = pickle.dumps(undistorted_img)

        # Send message length first
        message_size = struct.pack("L", len(data)) ### CHANGED

        # Then data
        clientsocket.sendall(message_size + data)


if __name__ == '__main__':
    main()