#!/usr/bin/env python

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
from std_msgs.msg import (
    UInt16,
)
import baxter_interface


class calibrator(object):
    
    def __init__(self):
        self.level = 122
    
    def new_image(self, data):
        self.frame = data

    def threshold(self, calib_level):
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.ret, self.mask = cv2.threshold(self.gray, calib_level, 255,cv2.THRESH_BINARY)

    def find_contours(self):
        self.contours, self.hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.frame, self.contours, -1, (0, 255, 0), 3)


rospy.init_node('calibrator', anonymous = True)

def callback(data):
    
    try:
        cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    calib.new_image(cv_image)
    calib.threshold(calib.level)
    calib.find_contours()

    cv2.imshow('frame', calib.frame)
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('q'):
        print calib.level
        rospy.signal_shutdown("q key pressed!")
    elif k == ord('d') and calib.level < 255:
        calib.level = calib.level + 1
    elif k == ord('a') and calib.level > 0:
        calib.level = calib.level - 1

if __name__ == "__main__":
    
    calib = calibrator()
    image_sub = rospy.Subscriber("/cameras/left_hand_camera/image",
                                    Image, callback)
    
    while not rospy.is_shutdown():
        rospy.spin()
