#!/usr/bin/env python

# Author: Ryan Saltus

import rospy

import geometry_msgs.msg
from sensor_msgs.msg import Image
import numpy as np
import baxter_interface
from baxter_pykdl import baxter_kinematics


class velocity_converter(object):
    
    def __init__(self):

        self._control_arm = baxter_interface.limb.Limb('right')
        self._kin = baxter_kinematics('right')

        self.image_sub = rospy.Subscriber("/end_effector_velocities",
                                            geometry_msgs.msg.Twist, self.callback)
    
    def callback(self, data):

        rate = rospy.Rate(100)
        
        # Compute joint velocities based on desired end effector velocity 
        control_joint_names = self._control_arm.joint_names()
        jacob_i = self._kin.jacobian_pseudo_inverse()

        if self.command[0,0] > .02:
            self.command[0,0] = .02
        if self.command[0,0] < -.02:
            self.command[0,0] = -.02
        if self.command[1,0] > .02:
            self.command[1,0] = .02
        if self.command[1,0] < -.02:
            self.command[1,0] = -.02
        
        self._vel_command = np.transpose(np.array([self.command[0,0],self.command[1,0],0,0,0,0]))
        vel = np.matmul(jacob_i, self._vel_command) 
       
        # Format and send computed velocities to the control arm 
        self._joint_command = {'right_s0':vel[0,0], 'right_s1':vel[0,1], 'right_e0':vel[0,2],
                               'right_e1':vel[0,3], 'right_w0':vel[0,4], 'right_w1':vel[0,5],
                               'right_w2':vel[0,6]}
        self._control_arm.set_joint_velocities(self._joint_command)
        rate.sleep()
        

# Define function to be called on shutdown
def rel():
    print 'Shutting down'

if __name__ == '__main__':
    rospy.init_node('vel_converter', anonymous=True)
    vel = velocity_converter()
    rospy.on_shutdown(rel)
    while not rospy.is_shutdown():	
    	rospy.spin()
