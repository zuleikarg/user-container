import os
import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Int32
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
from cv_bridge import CvBridge

class MotRem(Node):
    def __init__(self):
        super().__init__('motion_removal')
        self.rig_obj_removed_publisher_ = self.create_publisher(Image, 'rigid_opt_comb', 10)

        self.bridge = CvBridge()
        self.rigid_object_image_ = np.empty(0)
        self.nonrigid_object_image_ = np.empty(0)
        self.optical_flow_image_ = np.empty(0)
        self.camera_frame_ = np.empty(0)

        cv2.namedWindow('ORBs', cv2.WINDOW_AUTOSIZE)

        self.camera = self.create_subscription(
            Image,
            'camera_corr',
            self.camera_callback,
            10)

        self.nonrig_obj_subscriber_ = self.create_subscription(
            Image,
            'nonrigid_segmentation',
            self.nonrig_obj_callback,
            10)
        self.rig_obj_subscriber_ = self.create_subscription(
            Image,
            'rigid_segmentation',
            self.rig_obj_callback,
            10)

        self.optical_flow_subscriber = self.create_subscription(
            Image,
            'opticalflow',
            self.opt_flow_callback,
            10)
        
        #self.timer = self.create_timer(0.1, self.timer_callback)

    #def timer_callback(self):
    #    res, frame = self.cap.read()
    #    if (self.rigid_object_image_.size > 0 and self.nonrigid_object_image_ is not None and self.nonrigid_object_image_.size > 0):
    #        self.pub_combination()

    def camera_callback(self,msg):
        if (msg.height != 0):
            self.camera_frame_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if (self.rigid_object_image_.size >0 and self.nonrigid_object_image_.size >0 and self.optical_flow_image_.size >0):
                self.pub_combination()


    def rig_obj_callback(self, msg):
        if (msg.height != 0):
            self.rigid_object_image_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            #self.get_logger().info("Rigid segmentation arrived.")

    def nonrig_obj_callback(self, msg):
        if (msg.height != 0):
            self.nonrigid_object_image_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            #self.get_logger().info("NonRigid segmentation arrived.")
            
    def opt_flow_callback(self, msg):
        if (msg.height != 0):
            self.optical_flow_image_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            #self.get_logger().info("Optical flow image arrived.")


    def pub_combination(self):
        self.get_logger().info("Both images are ready.")
        
        # Adding ORB points to grayscale image
        img_orb_gray = cv2.cvtColor(self.camera_frame_, cv2.COLOR_BGR2GRAY)
        # Initiate ORB detector
        orb = cv2.ORB_create()


        # # Create a large kernel (e.g., 15x15)
        kernel = np.ones((7, 7), np.uint8)

        opt_image = np.copy(self.optical_flow_image_)

        # # Apply dilation with multiple iterations
        dilated_nr = cv2.dilate(self.nonrigid_object_image_, kernel, iterations=2)
        dilated_r = cv2.dilate(self.rigid_object_image_, kernel, iterations=2)

        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_r, 0, cv2.CV_32S)

        moving_objects = []

        for label_id in range(1, numLabels):  # skip background
            object_mask = (labels == label_id).astype(np.uint8) * 255

            # Resize opt_image if needed
            if object_mask.shape != opt_image.shape:
                opt_image = cv2.resize(opt_image, (object_mask.shape[1], object_mask.shape[0]))

            # Check overlap
            overlap = cv2.bitwise_and(object_mask, opt_image)

            if np.count_nonzero(overlap) > 0:
                moving_objects.append(object_mask)

        # Combine masks
        combined_mask = np.zeros_like(opt_image, dtype=np.uint8)
        for obj in moving_objects:
            combined_mask = cv2.bitwise_or(combined_mask, obj)

        # Resize opt_image if needed
        if combined_mask.shape != dilated_nr.shape:
            combined_mask = cv2.resize(combined_mask, (dilated_nr.shape[1], dilated_nr.shape[0]))


        kp_out_nonrigid = []
        # find the keypoints with ORB
        kp = orb.detect(img_orb_gray,None)
        if len(kp) != 0:
            for k in kp:
                if dilated_nr[int(k.pt[1]), int(k.pt[0])] != 255 and combined_mask[int(k.pt[1]), int(k.pt[0])] != 255:
                    kp_out_nonrigid.append(k)

        # compute the descriptors with ORB
        kp_out_nonrigid, des = orb.compute(img_orb_gray, kp_out_nonrigid)

        # draw only keypoints location,not size and orientation
        img_orb_gray = cv2.drawKeypoints(img_orb_gray, kp_out_nonrigid, None, color=(0,255,0), flags=0)
        cv2.imshow("ORBs",  img_orb_gray)
        cv2.waitKey(1)


def main(args=None):
    # Start the fusion node
    try:
        rclpy.init(args=args)
        motion_removal_node = MotRem()

        rclpy.spin(motion_removal_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if motion_removal_node is not None:
            motion_removal_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
            cv2.destroyAllWindows()
            
        motion_removal_node.get_logger().info("Fusion node has been shut down.")

if __name__ == "__main__":
    # Call Main Function
    main()