#!/usr/bin/env python

import numpy as np
import sys
import rospy
from std_msgs.msg import (
    UInt16,
)
import baxter_interface
from baxter_pykdl import baxter_kinematics
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import csv
import tf

class Controller(object):

    def __init__(self, limb):
        self._control_limb = limb
        self._control_arm = baxter_interface.limb.Limb(self._control_limb)
        self._kin = baxter_kinematics('right')

    def _reset_control_modes(self):
        rate = rospy.Rate(100)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._control_arm.exit_control_mode()
            rate.sleep()
        return True

    def set_neutral(self):
        """
        Sets both arms back into a neutral pose.
        """
        print("Moving to neutral pose...")
        self._control_arm.move_to_neutral()

    def clean_shutdown(self):
        print("\nExiting example...")
        #return to normal
        self._reset_control_modes()
        self.set_neutral()

    def control(self):
        
        rate = rospy.Rate(100)
        control_joint_names = self._control_arm.joint_names()
        
        
        jacob_i = self._kin.jacobian_pseudo_inverse()
        self._vel_command = np.transpose(np.array([0,0.03,0,0,0,0]))
        vel = np.matmul(jacob_i, self._vel_command) 

        self._joint_command = {'right_s0':vel[0,0], 'right_s1':vel[0,1], 'right_e0':vel[0,2], 'right_e1':vel[0,3]
                            , 'right_w0':vel[0,4], 'right_w1':vel[0,5], 'right_w2':vel[0,6]}
        self._control_arm.set_joint_velocities(self._joint_command)
        rate.sleep()

def main():
    
    rospy.init_node("velocity_controller")
    print("Initializing node... ")
    controller = Controller('right')
    rospy.on_shutdown(controller.clean_shutdown)
    kin = baxter_kinematics('right')
    while not rospy.is_shutdown():
        controller.control()        

if __name__ == '__main__':
    sys.exit(main())
