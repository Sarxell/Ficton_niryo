import numpy as np
import glob
import cv2
import os

# You should replace these 3 lines with the output in calibration step
DIM = (1600, 1200)
K = np.array([[781.3524863867165, 0.0, 794.7118000552183], [0.0, 779.5071163774452, 561.3314451453386], [0.0, 0.0, 1.0]])
D = np.array([[-0.042595202508066574], [0.031307765215775184], [-0.04104704724832258], [0.015343014605793324]])

def main():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FPS, 25)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)

    # RED MASK
    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([50, 255, 255])

    # upper mask (170-180)
    # lower_red = np.array([170, 50, 50])
    lower_red = np.array([170, 90, 90])
    upper_red = np.array([180, 255, 255])

    while 1:
        _, image = capture.read()   # get an image from the camera
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.resize(image, (1600, 1200))

        #  Undistorting image!!
        h,w = image.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow("fisheye image", image)
        cv2.imshow("undistorted", undistorted_img)
        # ----------------------------
        if cv2.waitKey(1) & 0xFF == 27:   # add code to wait for a key press
            break

        # red mask creation of the undistorted img!!
        img_hsv = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2HSV)

        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0 + mask1
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        # mask = cv2.morphologyEx(mask ,cv2.MORPH_OPEN,kernel)
        mask, _, _ = removeSmallAndBigComponents(mask, 4000, 20000)

        # if we want to see the mask in red
        # red_mask = cv2.bitwise_and(red_mask, red_mask, mask=mask)

        cv2.imshow('mask', mask)

    capture.release()
    cv2.destroyAllWindows()


def removeSmallAndBigComponents(image, threshold_min, threshold_max):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    x = None
    y = None
    img2 = np.zeros(output.shape, dtype=np.uint8)

    # for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        print([sizes[i]])
        if sizes[i] >= threshold_min:
            if sizes[i] <= threshold_max:
                # to use the biggest
                x, y = centroids[i + 1]
                # only needed if we want the biggest one
                # threshold = sizes[i]
                img2[output == i + 1] = 255

    return img2, x, y


if __name__ == '__main__':
    main()