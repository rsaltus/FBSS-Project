#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import csv
import tf

class observer(object):
    
    def __init__(self):

        # Create tf interface, wait for buffer to register data
        self.tl = tf.TransformListener()
        self.tl.waitForTransform('right_hand_camera', 'base', rospy.Time(0),rospy.Duration(10))

        # Create subscriber for hand camera
        self.image_sub = rospy.Subscriber("/cameras/right_hand_camera/image",
                                            Image, self.callback_image)
        # Build CV-ROS bridge
        self.bridge = CvBridge()
        self.out = cv2.VideoWriter("out.avi" ,cv2.cv.CV_FOURCC(*'XVID'), 24, (640,400))

        # Create file to record data
        self.new_file = open("velocity.csv","w")
        self.writer = csv.writer(self.new_file)
        self.init_time = rospy.Time.now()
    
    # Call this function whenever an image is received from hand camera
    def callback_image(self, data):
        
        # Try to convert to OpenCV, if failure, throw exception
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # Write to frame to output avi file
        self.out.write(cv_image) 

        # Lookup linear and angular velocity of right hand camera
        twist = self.tl.lookupTwistFull('right_hand_camera', 'base', 'right_hand_camera',
                            (0,0,0),'right_hand_camera', rospy.Time(0),rospy.Duration(.01))

        # Get current time and convert to seconds
        time = (rospy.Time.now()-self.init_time)
        time = time.to_sec()
        
        # Create row to be written to CSV file, and write it
        row =[time, twist[0][0], twist[0][1], twist[0][2], twist[1][0], twist[1][1], twist[1][2]]
        self.writer.writerow(row)

# Define function to be called on shutdown
def rel():
    print 'Shutting down'
    obs.out.release()

if __name__ == '__main__':
    rospy.init_node('observer', anonymous=True)
    obs = observer() 
    rospy.on_shutdown(rel)
    while not rospy.is_shutdown():	
    	rospy.spin()
