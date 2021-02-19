#!/usr/bin/env python 

import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

from math import sin, cos, pi
import sys
import copy

import scipy.io as sio

import geometry_msgs.msg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import UInt16

from cv_bridge import CvBridge, CvBridgeError

import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics

state = 0
command = 0
desk = 0

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


###############################################
###############################################

class writer(object):
    
    def __init__(self):
        self.out = cv2.VideoWriter("out.avi" ,cv2.cv.CV_FOURCC(*'XVID'), 24, (640,400))
    
    def rel(self):
        print("Shutting down")
        self.out.release()

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



###############################################
###############################################



class deformation_model(object):

    def __init__(self, points, omega):
        self.R = np.zeros((1,2))
        self.T = points
        self.num_points = 0
        self.omega = omega
        self.init = 0
        
        self.dsn_array = np.zeros((1,1))
        self.energy_array = np.zeros((1,1))

    def algorithm_initialization(self, s_new, robot_p):
        self.size = s_new.shape[0]
        self.R = np.zeros((1,2))
        self.sigma = np.zeros((1,1,self.size))
        self.r_old = robot_p
        self.s_old = s_new
        self.origin = robot_p
        self.init = 1
        self.Q_hat = np.zeros((self.size,2))

    def add_point(self, d_s, d_r):
        self.sigma = np.append(self.sigma, d_s, axis = 0)
        self.R = np.append(self.R, d_r, axis = 0)
        self.r_old = self.r_new
        self.s_old = self.s_new
        self.num_points = self.num_points + 1
    
    def replace_point(self, d_s, d_r):
        self.sigma[self.num_points, 0, :] = d_s 
        self.R[self.num_points, :] = d_r
        self.r_old = self.r_new
        self.s_old = self.s_new
        self.num_points = self.num_points + 1

    def compute_relative(self, s_new, r_new):
        # Calculate the deltas, reshape for appending
        self.r_new = r_new
        self.s_new = s_new
        delta_s = np.subtract(s_new, self.s_old)
        self.delta_r = np.subtract(r_new, self.r_old)
        self.delta_s = np.reshape(delta_s, (1,1,s_new.shape[0]))

    def compute_Ji(self, i):
        past_term = np.matmul(self.R, np.transpose(self.Q_hat[i,:]))
        past_term = np.subtract(past_term, self.sigma[:,0,i])
        current_term = np.matmul(self.delta_r, np.transpose(self.Q_hat[i,:]))
        current_term = np.subtract(current_term, self.delta_s[0,0,i])
        Ji = ((np.linalg.norm(current_term) ** 2) + (np.linalg.norm(past_term) ** 2))/2
        return Ji        

    def compute_total_J(self):
        total_J = 0
        for x in range(self.size):
            total_Ji = total_J + self.compute_Ji(x)
        self.J = total_Ji

    def compute_qi_hat_dot(self, gamma, i):
        past_term = np.matmul(self.R, np.transpose(self.Q_hat[i, :]))
        past_term = np.subtract(past_term, self.sigma[:, 0, i])
        current_term = np.matmul(self.delta_r, np.transpose(self.Q_hat[i, :]))
        current_term = np.subtract(current_term, self.delta_s[0, 0, i])
        cl_matrix = np.append(past_term, current_term, axis = 0)
        H_matrix = np.append(self.R, self.delta_r, axis = 0)
        qi_hat = -gamma * np.matmul(np.transpose(H_matrix), cl_matrix)
        return qi_hat
        
    def update_Q_matrix(self, gamma, epsilon):
        for x in range(self.size):
            Ji = self.compute_Ji(x)
            if Ji > epsilon:
                dq = self.compute_qi_hat_dot(gamma,x)
                self.Q_hat[x,:] = np.add(self.Q_hat[x,:], dq)

    def print_test(self):
        print("\r R = \r")
        print self.R
        print self.R.shape
        print("\r delta r = \r")
        print self.delta_r
        print self.delta_r.shape
        print("\r sigma = \r")
        print self.sigma
        print self.sigma.shape
        print("\r delta s = \r")
        print self.delta_s
        print self.delta_s.shape
        print("\r Q hat = \r")
        print self.Q_hat
        print self.Q_hat.shape




###############################################
###############################################




class Controller(object):
    
    def __init__(self, limb, desired_shape_params):
        self.desired_s = desired_shape_params
        self._control_arm = baxter_interface.limb.Limb(limb)
        self._kin = baxter_kinematics(limb)
        self.command = np.array([[-.02],[0]])
        self.state = 0

    def compute_sat_delta_s(self, current_s, min_s, max_s):
        self.delta_s = current_s - self.desired_s
        self.sat_delta_s = np.clip(self.delta_s, min_s, max_s)

    def compute_velocity_controller(self, q_hat, lamda):
        r_dot = -lamda * np.matmul(np.linalg.pinv(q_hat), self.sat_delta_s)
        return r_dot 

    def _reset_control_modes(self):
        rate = rospy.Rate(100)
        for _ in xrange(100):
            if rospy.is_shutdown():
                return False
            self._control_arm.exit_control_mode()
            rate.sleep()
        return True

    def set_neutral(self):
        
        print("Moving to neutral pose...")
        self._control_arm.move_to_neutral()

    def clean_shutdown(self):
        print("\nExiting example...")
        #return to normal
        self._reset_control_modes()
        self.set_neutral()

    def command_velocity(self):
        
        rate = rospy.Rate(100)
        
        # Compute joint velocities based on desired end effector velocity 
        control_joint_names = self._control_arm.joint_names()
        jacob_i = self._kin.jacobian_pseudo_inverse()
        
        lim = .003
        if self.command[0,0] > lim:
            self.command[0,0] = lim 
        if self.command[0,0] < -lim:
            self.command[0,0] = -lim
        if self.command[1,0] > lim:
            self.command[1,0] = lim 
        if self.command[1,0] < -lim:
            self.command[1,0] = -lim
        
        self._vel_command = np.transpose(np.array([self.command[0,0],self.command[1,0],0,0,0,0]))
        vel = np.matmul(jacob_i, self._vel_command) 
       
        # Format and send computed velocities to the control arm 
        self._joint_command = {'right_s0':vel[0,0], 'right_s1':vel[0,1], 'right_e0':vel[0,2],
                               'right_e1':vel[0,3], 'right_w0':vel[0,4], 'right_w1':vel[0,5],
                               'right_w2':vel[0,6]}
        self._control_arm.set_joint_velocities(self._joint_command)
        rate.sleep()

    def rotate_vector(self, vel, delta_theta):
        rotation_matrix = np.array([[cos(delta_theta), -sin(delta_theta)],
                                    [sin(delta_theta), cos(delta_theta)]])
        new_vel = np.matmul(rotation_matrix, vel) 
        return new_vel

    def get_endpoint_position(self):
        R = self._control_arm.endpoint_pose()['position']
        robot_array = np.array([R[0], R[1]], ndmin=2)
        return robot_array
    


###############################################
###############################################



# Inits to be done before the callback
rospy.init_node('cnt_ctrl', anonymous = True)
dc_path = '/home/baxter/ryan_ws/ros_ws/src/python_package/fbss/desired_contour.npy'
desired_contour = np.load(dc_path)
controller = Controller('right', desired_contour)
Writer = writer()
old_time = time.time()


def callback(data):
 
    global old_time

    # Convert image to from ROS to OpenCV for processing
    try:
        cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    # Initial thresholding and contour building
    handler = image(cv_image)
    handler.threshold(120)
    handler.build_contour()
    
    # Try to convert the contour into an S array
    try:
        S = handler.convert_contour_to_shape_params(handler.cnt,15)
        rebuilt_contour = rebuild_contour(S,100)
    except:
        pass


    robot_new = controller.get_endpoint_position()



    # Initialization state
    # Perform small testing deformations in the shape of a star around the origin point
    if controller.state == 0:
        # Perform model inits
        if model.init == 0:
            controller.command = np.array([[0],[.003]])
            model.algorithm_initialization(S, robot_new) 
            controller.command_velocity() 
            model.init = 1
       
        model.compute_relative(S, robot_new)

        if np.linalg.norm(model.delta_r) > (float(model.omega)/2) and model.num_points < 5:
            model.add_point(model.delta_s, model.delta_r)
            controller.command = controller.rotate_vector(controller.command, 
                                                        (float(-126)*2*pi)/360)
        if model.num_points >= 5:
            print("state is now cruise")
            model.origin = robot_new
            controller.state = 1 

        controller.command_velocity() 

    # Cruising state
    elif controller.state == 1:
        model.compute_relative(S, robot_new)
        model.update_Q_matrix(.00001, .000000001)
        controller.compute_sat_delta_s(S, -.5, .5)
        r_dot = controller.compute_velocity_controller(model.Q_hat, lamda)
        controller.command = r_dot
        print r_dot
        controller.command_velocity()
        print("state is 1")
        if np.linalg.norm(model.r_new - model.origin) > (model.omega):
            model.num_points = 0
            model.init = 0 
            controller.state = 2

    # Recalibration state
    elif controller.state == 2:
        model.compute_relative(S, robot_new)
        model.update_Q_matrix(.00001, .000000001)
        controller.compute_sat_delta_s(S, -.5, .5)
        r_dot = controller.compute_velocity_controller(model.Q_hat, lamda)
        controller.command = r_dot
        print r_dot
        controller.command_velocity()
        print("state is 1")
        
        if np.linalg.norm(model.delta_r) > .007 and model.num_points < 5:
            model.replace_point(model.delta_s, model.delta_r)
        
        if model.num_points >= 5:
            model.origin = robot_new
            model.num_points = 0
            controller.state = 1 

        print("state is 2")

    else:
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

    Writer.out.write(handler.frame)

    norm = np.linalg.norm(controller.delta_s)
    model.dsn_array = np.append(model.dsn_array, norm)
    model.compute_total_J()
    model.energy_array = np.append(model.energy_array, model.J)

    # Program interaction
    # Press q to quit program
    if k == ord('q'):
        cv2.destroyAllWindows() 

        fig = plt.figure() 
        ax1 = fig.add_subplot(111)
        ax1.plot(model.dsn_array[:])
        
        fig2 = plt.figure() 
        ax2 = fig2.add_subplot(111)
        ax2.plot(model.energy_array[:])
        
        plt.show()

        sio.savemat('dsn_array', {'dsn':model.dsn_array})
        sio.savemat('energy_array', {'energy':model.energy_array})

        rospy.signal_shutdown("Exit on q press")
    if k == ord('a'):
        controller.command = .001
    if k == ord('s'):
        controller.command = 0 
    if k == ord('d'):
        controller.command = -.001 
    if k == ord('r'):
        model.origin = robot_new
        model.num_points = 0
        model.init = 0 
        controller.state = 0

    w = time.time()
    t = old_time - w
    old_time = w
    
if __name__ == "__main__":

    image_sub = rospy.Subscriber("/cameras/left_hand_camera/image",
                                    Image, callback)
    model = deformation_model(15, .015)
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame',1920,1080)
    R = np.zeros((1,3))
   
    d_c = rebuild_contour(desired_contour, 100)
    
    lamda = .0001
    
    # Use this as a notification that a move has been completed
    # This way the program calculates a new model point only after a move is complete
    min_point_dist = .05
    move = 0
    
    rospy.on_shutdown(Writer.rel)
    while not rospy.is_shutdown():
        rospy.spin()
