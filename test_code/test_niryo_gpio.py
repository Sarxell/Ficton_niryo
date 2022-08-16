#!/usr/bin/env python
from niryo_one_python_api.niryo_one_api import *
import rospy
import time

rospy.init_node('niryo_one_example_python_api')

n = NiryoOne()

try:
    # Your code here
    pin = GPIO_2C
    n.pin_mode(pin, PIN_MODE_INPUT)
    while 1:
        print("\nRead pin GPIO 1_A 1: " + str(n.digital_read(pin)))


except NiryoOneException as e:
    print e
    # Handle errors here