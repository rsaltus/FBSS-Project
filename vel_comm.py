import sys
import copy


class commander(object):
    




    def command_velocity(self):
        
        rate = rospy.Rate(100)
        
        # Compute joint velocities based on desired end effector velocity 
        control_joint_names = self._control_arm.joint_names()
        jacob_i = self._kin.jacobian_pseudo_inverse()

        if self.command[0,0] > .002:
            self.command[0,0] = .002
        if self.command[0,0] < -.002:
            self.command[0,0] = -.002
        if self.command[1,0] > .002:
            self.command[1,0] = .002
        if self.command[1,0] < -.002:
            self.command[1,0] = -.002
        
        self._vel_command = np.transpose(np.array([self.command[0,0],self.command[1,0],0,0,0,0]))
        vel = np.matmul(jacob_i, self._vel_command) 
       
        # Format and send computed velocities to the control arm 
        self._joint_command = {'right_s0':vel[0,0], 'right_s1':vel[0,1], 'right_e0':vel[0,2],
                               'right_e1':vel[0,3], 'right_w0':vel[0,4], 'right_w1':vel[0,5],
                               'right_w2':vel[0,6]}
        self._control_arm.set_joint_velocities(self._joint_command)
        rate.sleep()
