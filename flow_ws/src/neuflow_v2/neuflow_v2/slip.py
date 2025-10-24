import open3d as o3d
import torch
import os
import numpy as np
import cv2
from datetime import datetime
from neuflow_v2.NeuFlow.neuflow import NeuFlow
from neuflow_v2.NeuFlow.backbone_v7 import ConvBlock
from neuflow_v2.data_utils import flow_viz
import os
import cv2
import numpy as np
import rclpy
import message_filters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
from cv_bridge import CvBridge

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

from scipy.spatial.transform import Rotation as R


image_width = 640
image_height = 480
vis_path = 'camera_results/'

class SLIP(Node):

    def __init__(self):
        # Initialize node
        super().__init__('slip')

        # Initialize variables
        self.odom_count = 0
        self.odom_no_count = 0
        self.odom_prev_mov = False
        self.bridge = CvBridge()
        self.camera_frame_ = np.empty(0)
        self.depth_frame_ = np.empty(0)
        self.prev_frame_ = None
        self.curr_frame_ = None

        self.prev_depth_ = None
        self.curr_depth_ = None

        self.prev_quat_ = None
        self.prev_trans_ = None
        self.curr_quat_ = None
        self.curr_trans_ = None

        self.prev_floor_seg_ = None
        self.curr_floor_seg_ = None


        self.prev_T_ = None
        self.curr_T_ = None
        
        self.T_odom_prev = None
        self.T_odom_curr = None

        self.T_est_total = np.eye(4)

        # Transformation matrices between coordinate systems
        self.T_base_to_body = np.diag([1., 1., 1., 1.])  # ROS to camera frame
        self.T_base_to_body[:3, 3] = [0.0, 0.0, 0.064]  # Adjust if needed

        self.T_body_to_cam = np.diag([1., 1., 1., 1.])  # ROS to camera frame
        self.T_body_to_cam[:3, 3] = [0.3156, 0.017, 0.096]  # Adjust if needed

        quat_cam_to_frame = np.array([0.0111, 0.0047, -0.00207, 1.0])  # x,y,z,w
        self.T_cam_to_frame = np.diag([1., 1., 1., 1.])  # ROS to camera frame
        self.T_cam_to_frame[:3, :3] = R.from_quat(quat_cam_to_frame).as_matrix()
        self.T_cam_to_frame[:3, 3] = [-0.0003, 0.014886, 0.00008]  # Adjust if needed

        quat_frame_to_opt = np.array([-0.5, 0.5, -0.5, 0.5])  # x,y,z,w
        self.T_frame_to_opt = np.diag([1., 1., 1., 1.])  # ROS to camera frame
        self.T_frame_to_opt[:3, :3] = R.from_quat(quat_frame_to_opt).as_matrix()
        self.T_frame_to_opt[:3, 3] = [0.0, 0.0, 0.0]  # Adjust if needed

        # Final static transformation betwen base_link and camera frame
        self.T_to_cam = self.T_base_to_body @ self.T_body_to_cam @ self.T_cam_to_frame @ self.T_frame_to_opt

        self.bool_first_frame_ = False
        
        # Intrinsic camera parameters
        self.fx = 605.639
        self.fy = 605.663
        self.cx = 325.493
        self.cy = 248.010

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Start camera
        #cap = cv2.VideoCapture('/dev/video6')  # Use 0 for default webcam


        self.sub_tf = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10)

        # self.sub_odom = self.create_subscription(
        #     Odometry,
        #     '/robot/odom',
        #     self.odom_callback,
        #     10)

        # self.sub_depth = self.create_subscription(
        #     Image,
        #     '/robot/camera/aligned_depth_to_color/image_raw',
        #     self.depth_callback,
        #     10)

        # self.sub_rgb = self.create_subscription(
        #     Image,
        #     '/robot/camera/color/image_raw',
        #     self.camera_callback,
        #     10)
        
        # Create subscribers
        self.sub_rgb   = message_filters.Subscriber(self, Image, '/robot/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/robot/camera/aligned_depth_to_color/image_raw')
        self.sub_odom  = message_filters.Subscriber(self, Odometry, '/robot/odom')
        self.local_map_pcd = o3d.geometry.PointCloud()
        self.frame_count = 0
        self.local_map_size = 20  # number of frames per local map
        self.time_interval = 0

        # Approximate time sync
        ats = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_odom], 
            queue_size=5, 
            slop=0.02   # tolerance in seconds (e.g., 50ms)
        )

        # Start synchronized callback
        ats.registerCallback(self.synced_callback)

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        self.depth_callback(depth_msg)
        self.odom_callback(odom_msg)
        self.camera_callback(rgb_msg)

    # CAMERA CALLBACK - Get previous frame at first, then current frame  and call the process function
    def camera_callback(self,msg):
        if (msg.height != 0):
            self.bool_first_frame_ = True
            self.camera_frame_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if (msg.height != 0):

            if(self.prev_frame_ is None and self.curr_frame_ is None):
                self.prev_frame_ = np.copy(self.camera_frame_)
                self.prev_floor_seg_ = self.process_slic()
            else:
                self.curr_frame_ = np.copy(self.camera_frame_)
                self.curr_floor_seg_ = self.process_slic()
                if(self.prev_depth_ is not None and self.curr_depth_ is not None and self.prev_quat_ is not None and self.curr_quat_ is not None):
                    self.estimate_odom()


    # DEPTH CALLBACK - Get previous depth at first, then current depth
    def depth_callback(self,msg):
        if (msg.height != 0):
            self.depth_frame_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            # print("Received depth frame")
            if(self.prev_depth_ is None and self.curr_depth_ is None):
                self.prev_depth_ = np.where((self.depth_frame_/ 1000.0 == 0) | (self.depth_frame_/ 1000.0 >= 65000), np.nan, self.depth_frame_ / 1000.0)

            else:
                self.curr_depth_ = np.where((self.depth_frame_/ 1000.0 == 0) | (self.depth_frame_/ 1000.0 >= 65000), np.nan, self.depth_frame_ / 1000.0)


    # ODOMETRY CALLBACK - Get previous odom at first, then current odom and compute T_odom with velocity and time interval
    def odom_callback(self,msg):
        if (msg.twist.twist.linear.x is not None or msg.twist.twist.linear.y is not None or msg.twist.twist.linear.z is not None or msg.twist.twist.angular.z is not None):
            if(self.prev_trans_ is None and self.curr_trans_ is None and self.prev_quat_ is None and self.curr_quat_ is None):
                self.prev_trans_ = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                self.prev_quat_ = msg.twist.twist.angular.z
                self.time_prev = datetime.now()

            else:
                self.curr_trans_ = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                self.curr_quat_ = msg.twist.twist.angular.z
                self.time_curr = datetime.now()
                diff_time = (self.time_curr - self.time_prev).total_seconds()
                self.T_odom_transform(diff_time)
            
    # TRANSFORMATION ODOM - Get the transformation between frames using the velocity and time interval.
    def T_odom_transform(self,d_t):
    
        if(self.T_odom_prev is None):
            # prev pose (4x4)
            R_prev = np.array([
            [np.cos(self.prev_quat_*d_t), -np.sin(self.prev_quat_*d_t), 0],
            [np.sin(self.prev_quat_*d_t),  np.cos(self.prev_quat_*d_t), 0],
            [0,           0,          1]])
            self.T_odom_prev = np.eye(4)
            self.T_odom_prev[:3, :3] = R_prev
            self.T_odom_prev[:3, 3] = self.prev_trans_*d_t
                    
        else:
            R_curr = np.array([
            [np.cos(self.prev_quat_*d_t), -np.sin(self.prev_quat_*d_t), 0],
            [np.sin(self.prev_quat_*d_t),  np.cos(self.prev_quat_*d_t), 0],
            [0,           0,          1]])
            self.T_odom_curr = np.eye(4)
            self.T_odom_curr[:3, :3] = R_curr
            self.T_odom_curr[:3, 3] = self.prev_trans_*d_t

    # TF CALLBACK - Get transformation from odom to base_link, first prev_T_, then curr_T_
    def tf_callback(self,msg):

        if (msg.transforms[0].child_frame_id == "base_link"):
            trans = np.array([msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y, msg.transforms[0].transform.translation.z])
            quat = [msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y, msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w]
            
            # Convert quaternion to rotation matrix
            rotation_matrix = R.from_quat(quat).as_matrix()
            # Translation vector
            translation = trans

            # Construct transformation matrix
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = translation

            if(self.prev_T_ is None and self.curr_T_ is None):
                self.prev_T_ = T
            else:
                self.curr_T_ = T


    # KEYPOINTS TO POINTCLOUD FUNCTION - Convert matched keypoints to 3D using depth and camera intrinsics
    def keypoints_to_pointcloud(self,keypoints, depth_image):
        points = []
        for kp in keypoints:
            u, v = int(kp.pt[0]), int(kp.pt[1])
            z = depth_image[v, u]  # depth in meters
            if z == 0: continue  # skip invalid depth
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy
            points.append([x, y, z])
        return np.array(points)

    # ESTIMATE ODOM FUNCTION - Determine if the robot is possibly slipping with:
    #  - Estimating displacement with tranformation between floor keypoints of prev and curr frames.
    #  - Calculating displacement with given odometry velocity and time interval.
    def estimate_odom(self):
            # Keep prev and curr frames and curoff top part
            frame_p = np.copy(self.prev_frame_)
            frame_c = np.copy(self.curr_frame_)
            
            # Get top part as black segment and stack it with the floor segmentation as a white segment
            top_part = np.zeros_like(frame_p[:240,:,0], dtype=np.uint8)

            total_img_floor_p = np.vstack((top_part, self.prev_floor_seg_))
            total_img_floor_c = np.vstack((top_part, self.curr_floor_seg_))

            # Anything that is not floor, will be black in the original images
            frame_p[total_img_floor_p!=255] = [0,0,0]
            frame_c[total_img_floor_c!=255] = [0,0,0]

            # Enhance the texture of the floor. Especially interesting for smooth floors
            gray_p = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)
            enhanced_p = cv2.equalizeHist(gray_p)

            gray_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
            enhanced_c = cv2.equalizeHist(gray_c)

            # Detect keypoints from floor with SHIFT
            sift = cv2.SIFT_create()
            kp_p, des_p = sift.detectAndCompute(enhanced_p, None)
            kp_c, des_c = sift.detectAndCompute(enhanced_c, None)

            if des_p is None or des_c is None:
                print("No descriptors found in one of the frames.")
                return
            des_p = des_p.astype(np.float32)
            des_c = des_c.astype(np.float32)

            # Get matches between keypoints of prev and curr frames
            bf = cv2.BFMatcher()
            knn = bf.knnMatch(des_p, des_c, k=2)
            matches = []
            if matches is None:
                print("No matches found between frames.")
                return
            if len(knn) <2:
                print("Not enough matches found between frames.")
                return
            for m,n in knn:
                if m is None or n is None:
                    break
                # Threshold to consider it a good correspondence has to be pretty high
                if m.distance < 0.85 * n.distance:
                    matches.append(m)

            # Build an image showing matches where both have valid depth
            # match_img = cv2.drawMatches(enhanced_p, kp_p, enhanced_c, kp_c, matches, None, flags=2)

            # Create 3D poitnclouds base on keypoints from prev and curr frames
            points_p = self.keypoints_to_pointcloud(kp_p, self.prev_depth_)
            points_c = self.keypoints_to_pointcloud(kp_c, self.curr_depth_)

            points_p = points_p[~np.isnan(points_p).any(axis=1)]
            points_c = points_c[~np.isnan(points_c).any(axis=1)]

            # Create Open3D point clouds and set them up with the keypoint pointcloud created before
            pcd_p = o3d.geometry.PointCloud()
            pcd_c = o3d.geometry.PointCloud()
            pcd_p.points = o3d.utility.Vector3dVector(points_p)  # from previous frame
            pcd_c.points = o3d.utility.Vector3dVector(points_c)  # from current frame

            # Build correspondences from matches
            correspondences = np.array([[m.queryIdx, m.trainIdx] for m in matches])
            correspondence_set = o3d.utility.Vector2iVector(correspondences)

            # Run RANSAC registration in order to estimate the transformation betwen keypoints of prev and curr frames
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                pcd_p, pcd_c, correspondence_set,
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
            )

            # print("Estimated transformation:\n", result.transformation)


            # Convert odometry motion into camera frame
            T_prev = self.T_odom_prev @ self.T_to_cam
            if self.T_odom_curr is None:
                T_curr = np.eye(4) @ self.T_to_cam
            else:
                T_curr = self.T_odom_curr @ self.T_to_cam
            T_rel_cam = np.linalg.inv(T_prev) @ T_curr

            # print("Current Odometry:\n", T_rel_cam)


            t_est = result.transformation[:3, 3]
            t_gt = T_rel_cam[:3, 3]
            R_est = result.transformation[:3, :3]
            R_gt = T_rel_cam[:3, :3]

            # Translation error vector
            trans_error_vec = t_gt - t_est
            translation_error = np.linalg.norm(trans_error_vec)

            # Rotation error using relative rotation
            T_diff = T_rel_cam[:3, :3] @ result.transformation[:3, :3].T
            R_diff = T_diff[:3,:3]
            rotation_error_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
            rotation_error_deg = np.degrees(rotation_error_rad)
            print("Traslation error:\n", translation_error)
            print("Rotation error:\n", rotation_error_deg)

            # Rotation in degrees of estimation and actual odometry
            rot_est = np.arccos(np.clip((np.trace(R_est) - 1) / 2, -1.0, 1.0))
            rot_est_deg = np.degrees(rot_est)

            rot_gt = np.arccos(np.clip((np.trace(R_gt) - 1) / 2, -1.0, 1.0))
            rot_gt_deg = np.degrees(rot_gt)
            
            # SLIP DETECTION PIPELINE
            # It is divided in traslation slip and rotational slip
            #  - At the same time it is divided in:
            #    - When the camera detects a movement and the odometry does not:
            #      - If the estimated movement is bigger than the actual movement.
            #      - And the movement on the previous step is near 0 to avoid noisy errors due to delay.
            #    - When the odometry detects a movement and the camera does not.
            #      - There will have to be 3 of these detections in order to consider that there is a 
            #        slip since the robot will be moving the wheels but not of the same spot. The 3
            #        consecutive detections are introduced aiming to avoid noisy errors due to delay.
            if translation_error > 1:  # meters

                #self.get_logger().warning("POSSIBLE TRANS SLIP.")
                if(np.linalg.norm(t_est) > np.linalg.norm(t_gt)):# Camera detects movement but not odom
                    if(self.odom_prev_mov == False ):
                        self.get_logger().warning("FINAL TRANS SLIP. - NOT ODOM")

                else:
                    self.odom_count +=1
                    if(self.odom_count >= 3):
                        self.get_logger().warning("FINAL TRANS SLIP.")
            else:
                self.odom_count=0
                if(np.linalg.norm(t_gt) > 0.01):
                    self.odom_prev_mov = True
                else:
                    self.odom_prev_mov = False

            if rotation_error_deg > 70:  # degrees
                #self.get_logger().warning("POSSIBLE ROT SLIP.")
                if(rot_est_deg > rot_gt_deg):# Camera detects movement but not odom
                    if(self.odom_prev_mov == False):
                        self.get_logger().warning("FINAL ROT SLIP. - NOT ODOM")
                else: 
                    self.odom_count +=1
                    if(self.odom_count >= 3):
                        self.get_logger().warning("FINAL ROT SLIP.")
            else: 
                self.odom_count = 0
                # print(rot_gt_deg)
                if(rot_gt_deg > 1):
                    self.odom_prev_mov = True
                else:
                    self.odom_prev_mov = False

            # cv2.imshow('ORBS SIFT',enhanced_p)
            # # cv2.imshow('ORBS AKAZE',image_with_keypoints_a)
            # cv2.imshow("Original Frame", self.prev_frame_)
            print("___________________________________")
            # cv2.waitKey(1)
            
            self.frame_count += 1
            self.prev_frame_ = self.curr_frame_
            self.prev_floor_seg_ = self.curr_floor_seg_
            self.prev_depth_ = self.curr_depth_

            self.prev_quat_ = self.curr_quat_
            self.prev_trans_ = self.curr_trans_

            self.time_prev = self.time_curr


            self.T_odom_prev = self.T_odom_curr

            self.prev_T_ = self.curr_T_


    def process_slic(self):
        # print("Processing SLIC")
        img = self.camera_frame_
        img = img[int(image_height/2):,:]

        ## FIRST PART - Line detection

        # Create a default parametrization of the LSD detector
        lsd = cv2.createLineSegmentDetector(0)

        line_img = img
        line_img_gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

        # Detect lines in the image
        lines = lsd.detect(line_img_gray)[0]  # Position 0 of the returned tuple are the detected lines
        
        if lines is not None:
            # Reshape to (N, 4) for easier math
            lines = lines.reshape(-1, 4)


            # Compute lengths of all lines at once
            x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
            lengths = np.hypot(x2 - x1, y2 - y1)
            degrees = np.arctan2(y2 - y1, x2 - x1) * (180.0 / np.pi)
            
            # Filter lines by length and angle
            mask = (lengths >= 60) & (((degrees >= 150) & (degrees <= 180)) | ((degrees <= -150) & (degrees >= -180)) | ((degrees <= 0) & (degrees >= 30)) | ((degrees <= 0) & (degrees >= -30)))
            long_lines = lines[mask]

            extended_lines = []
            for line in long_lines:
                extended = self.extend_line_to_image(line, line_img_gray.shape)
                if extended is not None:
                    extended_lines.append(extended)
            extended_lines = np.array(extended_lines)

            drawn_img = line_img.copy()

            for line in extended_lines:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        else:
            drawn_img = line_img.copy()

        # Get separated segments with lines going from one side of the image to the other
        #  - Lines are black, the background is white
        drawn_img = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)
        cv2.threshold(drawn_img, 1, 255, cv2.THRESH_BINARY, dst=drawn_img)

        ## SECOND PART- SLIC method

        slic = segmentation.slic(img, n_segments=120, start_label=1)
        recolored = np.zeros_like(img)

        # Iterate over each superpixel
        for seg_val in np.unique(slic):
            # Mask for the current segment
            mask = slic == seg_val
            # Compute mean color for that region
            mean_color = img[mask].mean(axis=0)
            # Assign mean color to all pixels in that region
            recolored[mask] = mean_color

        # Make it blurry in order to get less sharp edges
        gray = cv2.cvtColor(recolored, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3 , 3), 0)
        # Apply threshold to get the contrasting borders among the superpixels created
        #   and invert in order to match the previous part solution
        mid = cv2.Canny(blurred, 30, 150)
        mid_invert = cv2.bitwise_not(mid)

        # Morphological clean-up
        kernel = np.ones((5, 5), np.uint8)
        edges_closed = cv2.morphologyEx(mid_invert, cv2.MORPH_ERODE, kernel)

        # Make a border around the image in order to create closed segments
        border_size = 5
        border = cv2.copyMakeBorder(
            edges_closed,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Find closed contours 
        found = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[-2]  

        painted = img.copy()

        # Loop through contours and fill only the big ones
        min_area = 60000  # adjust this threshold depending on image size

        for cnt in contours:
            if cnt is None or len(cnt) < 3:
                continue  # skip small contours
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue  # skip small regions

            color = [255, 255, 255]  # White color for filling
            cv2.drawContours(painted, [cnt], -1, color, thickness=cv2.FILLED)

        # Keep in black all image but the contours drawn before
        painted = cv2.cvtColor(painted, cv2.COLOR_BGR2GRAY)
        cv2.threshold(painted, 254, 255, cv2.THRESH_BINARY, dst=painted)        

        # THIRD PART - Connection
        # Find connected components
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(drawn_img, 0, cv2.CV_32S)

        floor_parts = []

        # Find intersection of all components found after drawing the lines in the first part and
        #  the broad segmentation done in the second part
        for label_id in range(1, numLabels):  # skip background
            object_mask = (labels == label_id).astype(np.uint8) * 255

            max_pixels = np.count_nonzero(object_mask)

            # Check overlap
            overlap = cv2.bitwise_and(object_mask, painted)

            if (np.count_nonzero(overlap)*100)/max_pixels > 60:
                floor_parts.append(overlap)

        # Combine masks
        combined_mask = np.zeros_like(drawn_img, dtype=np.uint8)
        for f in floor_parts:
            combined_mask = cv2.bitwise_or(combined_mask, f)

        # Morphologycal clean-up
        kernel = np.ones((7, 7), np.uint8)
        combined_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        #segmentation.mark_boundaries(img, slic)
        # cv2.imshow("Large Closed Segments Painted", painted)
        # cv2.imshow('Original',img)
        # cv2.imshow('frame',border)
        # cv2.imshow('Lines',drawn_img)
        # cv2.imshow('Floor Segementation',combined_closed)
        # cv2.waitKey(1)
        return combined_closed


    def extend_line_to_image(self,line, img_shape):
        """Extend a line segment to the borders of the image."""
        x1, y1, x2, y2 = line
        h, w = img_shape[:2]

        # Line coefficients (Ax + By + C = 0)
        A = y2 - y1 + 1e-5
        B = x1 - x2 + 1e-5
        C = x2*y1 - x1*y2

        # Possible intersections with the image borders
        points = []
        borders = [
            (0, -C / B),            # x = 0
            (w - 1, -(A*(w - 1) + C) / B),  # x = w-1
            (-C / A, 0),            # y = 0
            (-(B*(h - 1) + C) / A, h - 1)   # y = h-1
        ]

        # Keep only the points within image bounds
        for (x, y) in borders:
            if 0 <= x < w and 0 <= y < h:
                points.append((x, y))

        # Should have 2 valid intersection points
        if len(points) >= 2:
            return [points[0][0], points[0][1], points[1][0], points[1][1]]
        else:
            return None

def main(args=None):
    # Start the fusion node
    try:
        rclpy.init(args=args)
        slic_node = SLIP()

        rclpy.spin(slic_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if slic_node is not None:
            slic_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
            cv2.destroyAllWindows()
            
        slic_node.get_logger().info("Slip estimation node has been shut down.")

if __name__ == "__main__":
    # Call Main Function
    main()