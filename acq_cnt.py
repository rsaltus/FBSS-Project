#!/usr/bin/env python 

import cv2
import numpy as np
import rospy
from matplotlib import pyplot as plt
from math import sin, cos, pi

import sys
import copy
import geometry_msgs.msg
from std_msgs.msg import String

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
from std_msgs.msg import (
    UInt16,
)

def rebuild_contour(shape_array, sample_no):
    points = np.zeros((sample_no,1, 2))
    harmonics = (shape_array.shape[0] - 2)/4
    j1, i = 1, 0
    for rho in np.linspace(0,2*pi,sample_no):
        G_bp = np.array([[cos(j1*rho), sin(j1*rho), 0, 0], 
                            [0, 0, cos(j1*rho), sin(j1*rho)]])
        for j in range(2,harmonics+1):
            G_app = np.array([[cos(j*rho), sin(j*rho), 0, 0], 
                            [0, 0, cos(j*rho), sin(j*rho)]])
            G_bp = np.append(G_bp, G_app, axis=1)
        G_bp = np.append(G_bp, np.identity(2), axis=1)
        points[i,0,0] = np.matmul(G_bp[0,:],shape_array)
        points[i,0,1] = np.matmul(G_bp[1,:],shape_array)
        i += 1
    points = points.astype(int) 
    return points

class image(object):

    def __init__(self,data):

        self.frame = data 
       
    def threshold(self, level):
        
        #Threshold the image, must be done before finding contours
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.ret, self.mask = cv2.threshold(self.gray, level, 255, cv2.THRESH_BINARY)

    def build_contour(self):

        # Find Contours using OpenCV
        self.contours, self.hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
        final = self.contours[0]
        for i in range(len(self.contours)):
            iterator = self.contours[i]
            if cv2.contourArea(iterator) > cv2.contourArea(final):
                final = iterator
        
        self.cnt = final 

        # Calculate total perimeter length, paramterize x and y based on rho
        self.perimeter = cv2.arcLength(self.cnt, True)
        self.reference = self.cnt[:,:,0].argmin()       
        a,b = self.cnt[0:self.reference-1,:,:], self.cnt[self.reference:-1,:,:]
        self.cnt = np.append(b, a, 0)
        rho_array = np.zeros([self.cnt.shape[0],1,1])

        for i in range(1, self.cnt.shape[0]):
            rho_array[i,0,0] = ((cv2.arcLength(self.cnt[0:i,:,:], False) * pi * 2) 
                                / self.perimeter)

        self.cnt = np.append(self.cnt, rho_array, 2)
        self.cnt = np.squeeze(self.cnt)

    def convert_contour_to_shape_params(self, array, harmonics):
       
        # Construct c vector
        long_c_vector = np.expand_dims(array[:,[0,1]].flatten(), axis=1)

        # Construct G array
        j1, flag = 1, 1
        for rho in np.nditer(array[:,2]):
            G_bp = np.array([[cos(j1*rho), sin(j1*rho), 0, 0], 
                             [0, 0, cos(j1*rho), sin(j1*rho)]])
            for j in range(2,harmonics+1):
                G_app = np.array([[cos(j*rho), sin(j*rho), 0, 0], 
                                  [0, 0, cos(j*rho), sin(j*rho)]])
                G_bp = np.append(G_bp, G_app, axis=1)
            G_bp = np.append(G_bp, np.identity(2), axis=1)
            if (flag == 1):
                G_bp_final = G_bp
                flag = 0
            else:
                G_bp_final = np.append(G_bp_final, G_bp, axis=0)

        G_bp_plus = np.linalg.pinv(G_bp_final)

        s = np.matmul(G_bp_plus, long_c_vector)
        return s

    def draw_ref(self):
        # Should draw the reference point ( where rho is init )
        size = 5 
        b = int(self.cnt[0,0])
        a = int(self.cnt[0,1])
        self.frame[a-size:a+size, b-size:b+size] = [255, 0, 0] 

    def show_image(self):
        cv2.imshow('frame', self.frame)


rospy.init_node('contour_control', anonymous = True)

def callback(data):

    try:
        cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    # Initial thresholding and contour building
    handler = image(cv_image)
    handler.threshold(210)
    handler.build_contour()
    
    # Try to convert the contour into an S array
    try:
        S = handler.convert_contour_to_shape_params(handler.cnt,15)
        rebuilt_contour = rebuild_contour(S,100)
    except:
        pass

    # Try to draw the contours and the reference point on the existing image
    try:
        cv2.drawContours(handler.frame, handler.contours, -1, (0,255,0), 3)
        cv2.drawContours(handler.frame, rebuilt_contour, -1, (0,0,255), 3)
        cv2.drawContours(handler.frame, d_c, -1, (255,0,0), 3)
        handler.draw_ref()
    except:
        pass 

    # Display the image and contours using OpenCV protocol
    handler.show_image() 
    k = cv2.waitKey(1) & 0xFF

    # Program interaction
    # Press q to quit program
    if k == ord('q'):
        cv2.destroyAllWindows() 
        rospy.signal_shutdown("Q pressed!")
   
    # Press y to grab a contour and save it into the working directory
    # Automatically overrites the old contour
    elif k == ord('y'): 
        np.save('/home/baxter/ryan_ws/ros_ws/src/python_package/fbss/desired_contour.npy', S)
        print S

if __name__ == "__main__":

    image_sub = rospy.Subscriber("/cameras/left_hand_camera/image",
                                    Image, callback)
    
    while not rospy.is_shutdown():
        rospy.spin()
