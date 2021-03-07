#!/usr/bin/env python 

import cv2
import numpy as np
import rospy
from matplotlib import pyplot as plt
from math import sin, cos, pi
import os

import sys
import copy
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
import moveit_commander

cap = cv2.VideoCapture(0)
desk = 0

class image(object):

    def __init__(self):

        self.ret, self.frame = cap.read()
       
    def threshold(self, level):
        
        #Threshold the image, must be done before finding contours
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.ret, self.mask = cv2.threshold(self.gray, level, 255, cv2.THRESH_BINARY)

    def build_contour(self):

        # Find Contours using OpenCV
        self.contours, self.hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
        self.cnt = self.contours[0]

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

    def rebuild_contour(self, shape_array, sample_no):
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

    def draw_ref(self):
        # Should draw the reference point ( where rho is init )
        size = 5
        b = int(self.cnt[0,0])
        a = int(self.cnt[0,1])
        self.frame[a-size:a+size, b-size:b+size] = [255, 0, 0] 
        pass

    def show_image(self):
        cv2.imshow('frame', self.frame)

class robot_controller(object):
    
    def __init__(self):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")
        self.group.set_planner_id("RRTConnectkConfigDefault");
        self.group.set_goal_position_tolerance(.0001)
        self.group.set_goal_orientation_tolerance(.0001)
        self.pose_target = geometry_msgs.msg.Pose()
        self.group.set_planning_time(.1)
        self.move_completed = 0
        
    def update_current_angles(self):
        current_angles = self.group.get_current_joint_values()
        self.a_1 = current_angles[0]
        self.a_2 = current_angles[1]
        self.a_3 = current_angles[2]
        self.a_4 = current_angles[3]
        self.a_5 = current_angles[4]
        self.a_6 = current_angles[5]
    
    def update_current_pose(self):
        current_pose = self.group.get_current_pose()
        self.p_x = current_pose.pose.position.x
        self.p_y = current_pose.pose.position.y
        self.p_z = current_pose.pose.position.z
        self.o_x = current_pose.pose.orientation.x
        self.o_y = current_pose.pose.orientation.y
        self.o_z = current_pose.pose.orientation.z
        self.o_w = current_pose.pose.orientation.w

    def change_pose(self, dx, dy, dz):
        self.update_current_pose()
        self.update_current_angles()
        
        # Setup the pose target based on the difference from the current pose
        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.x = self.o_x
        pose_target.orientation.y = self.o_y
        pose_target.orientation.z = self.o_z
        pose_target.orientation.w = self.o_w
        pose_target.position.x = self.p_x + dx
        pose_target.position.y = self.p_y + dy
        pose_target.position.z = self.p_z + dz
       
        # Compute an initial plan
        self.group.set_pose_target(pose_target)
        plan1 = self.group.plan()
        goal = np.array((plan1.joint_trajectory.points[-1].positions))
        current = np.array((self.a_1, self.a_2, self.a_3, self.a_4, self.a_5, self.a_6))
       
        # Failsafe for flipped manipulator, if test fails, replan (5 attempts)
        attempts = 0
        while attempts < 5:
            if np.linalg.norm(np.subtract(goal, current)) > 1:
                plan1 = self.group.plan()
                goal = np.array((plan1.joint_trajectory.points[-1].positions))
                current = np.array((self.a_1, self.a_2, self.a_3, 
                                    self.a_4, self.a_5, self.a_6))
                print("\r Invalid trajectory, replanning... \r")
                attempts = attempts + 1
            else: break

        self.group.execute(plan1)
        self.move_completed = 1
  

class deformation_model(object):

    def __init__(self, points):
        self.R = np.zeros((1,3))
        self.T = points
        self.num_points = 0
        self.init = 0

    def add_point(self, d_s, d_r):
        self.sigma = np.append(self.sigma, d_s, axis = 0)
        self.R = np.append(self.R, d_r, axis = 0)

    def compute_relative_and_add_point(self, s_new, r_new):
        
        # Inits
        if self.init == 0:
            self.size = s_new.shape[0]
            self.sigma = np.zeros((1,1,s_new.shape[0]))
            self.r_old = r_new
            self.s_old = s_new
            self.Q_hat = np.zeros((self.size,3))
            self.init = 1

        # Calculate the deltas, reshape for appending
        delta_s = np.subtract(s_new, self.s_old)
        self.delta_r = np.subtract(r_new, self.r_old)
        self.delta_s = np.reshape(delta_s, (1,1,s_new.shape[0]))
        
        # Add points to deformation model if still not full
        if self.num_points < self.T:
            self.add_point(self.delta_s, self.delta_r)
        
        self.r_old = r_new
        self.s_old = s_new
        self.num_points = self.num_points + 1


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

    def compute_qi_hat_dot(self,gamma,i):
        past_term = np.matmul(self.R, np.transpose(self.Q_hat[i,:]))
        past_term = np.subtract(past_term, self.sigma[:,0,i])
        current_term = np.matmul(self.delta_r, np.transpose(self.Q_hat[i,:]))
        current_term = np.subtract(current_term, self.delta_s[0,0,i])
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





class controller(object):
    
    def __init__(self):
        pass

    def function(self):
        pass




if __name__ == "__main__":

    model = deformation_model(5)
    # Below are all robot inits 
    
    R = np.zeros((1,3))

    robot = robot_controller()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('contour_control', anonymous = True)
    
    # Use this as a notification that a move has been completed
    # This way the program calculates a new model point only after a move is complete
    robot.move_completed = 0 
    min_point_dist = .05
    move = 0

    while(True):

        # Initial thresholding and contour building
        handler = image()
        handler.threshold(120)
        handler.build_contour()
        
        # Try to convert the contour into an S array
        try:
            S = handler.convert_contour_to_shape_params(handler.cnt,5)
            handler.rebuilt_contour = handler.rebuild_contour(S,100)
        except:
            pass

        # Try to draw the contours and the reference point on the existing image
        try:
            cv2.drawContours(handler.frame, handler.contours, -1, (0,255,0), 3)
            cv2.drawContours(handler.frame, handler.rebuilt_contour, -1, (0,0,255), 3)
            handler.draw_ref()
        except:
            pass

        # Display the image and contours using OpenCV protocol
        handler.show_image() 
        k = cv2.waitKey(1) & 0xFF
        
        # Program interaction
        # Press q to quit program
        if k == ord('q'):
            break
        
        elif k == ord('y') and desk == 0:
            np.save('/home/ryan/abb_ws/src/testing/scripts/desired_contour.npy', S)
            print S
            break

        elif k == ord('w') and desk == 0:
            robot.change_pose(.001,0,0)
        
        elif k == ord('a') and desk == 0:
            robot.change_pose(0,.001,0)
        
        elif k == ord('s') and desk == 0:
            robot.change_pose(-.001,0,0)
        
        elif k == ord('d') and desk == 0:
            robot.change_pose(0,-.001,0)
        
        elif k == ord('g') and desk == 0:
            x = float(raw_input())
            y = float(raw_input())
            z = float(raw_input())
            robot.change_pose(x,y,z)

    # Shutdown Protocol
    cap.release()
    cv2.destroyAllWindows()
