#! /usr/bin/env python

import rospy
from std_msgs.msg import Empty
    
if __name__ == "__main__":
    pub1 = rospy.Publisher('/robot/limb/left/suppress_collision_avoidance', Empty, queue_size=10)
    pub2 = rospy.Publisher('/robot/limb/right/suppress_collision_avoidance', Empty, queue_size=10)
    rospy.init_node('collision_suppresor')
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub1.publish()
        pub2.publish()
        r.sleep()
