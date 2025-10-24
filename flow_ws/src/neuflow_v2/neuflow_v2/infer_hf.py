import torch
import os
import numpy as np
import cv2
import time
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

from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
from cv_bridge import CvBridge

from skimage import segmentation

# Size of RGB-D images
image_width = 640
image_height = 480
# vis_path = 'camera_results/'

class OpticalFlow(Node):

    def __init__(self):
        # Initialize ROS2 node
        super().__init__('infer_hf')

        torch.backends.cudnn.benchmark = True
        # Create publisher for Optical Flow mask
        self.opt_flow_pub_ = self.create_publisher(Image, 'opticalflow', 10)

        # Initialize variables
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

        self.prev_T_ = None
        self.curr_T_ = None
        
        self.T_odom_prev = None
        self.T_odom_curr = None

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

        # Intrinsic camera parameters
        self.fx = 605.639
        self.fy = 605.663
        self.cx = 325.493
        self.cy = 248.010

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuFlow.from_pretrained("Study-is-happy/neuflow-v2").to(self.device)

        for m in self.model.modules():
            if isinstance(m, ConvBlock):
                m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
                m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
                delattr(m, "norm1")
                delattr(m, "norm2")
                m.forward = m.forward_fuse

        self.model.eval()
        self.model.half()
        self.model.init_bhwd(1, image_height, image_width, 'cuda')

        # if not os.path.exists(vis_path):
        #     os.makedirs(vis_path)

        # Start camera
        #cap = cv2.VideoCapture('/dev/video6')  # Use 0 for default webcam

        # Create subscribers
        #  - TF: to get robot pose. Does not have _stamp_ in topic name, therefore message_filters cannot be used
        #  - Message filters:
        #    - Odometry: to get robot velocity
        #    - Depth image
        #    - RGB image
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
            self.camera_frame_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if(self.prev_frame_ is None and self.curr_frame_ is None):
                self.prev_frame_ = self.camera_frame_
            else:
                self.curr_frame_ = self.camera_frame_
                if(self.prev_frame_ is not None and self.curr_frame_ is not None and self.prev_depth_ is not None and self.curr_depth_ is not None and self.prev_quat_ is not None and self.curr_quat_ is not None and self.curr_T_ is not None):
                    self.process_optflow()

    # DEPTH CALLBACK - Get previous depth at first, then current depth
    def depth_callback(self,msg):
        if (msg.height != 0):
            self.depth_frame_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

            if(self.prev_depth_ is None and self.curr_depth_ is None):
                self.prev_depth_ = self.depth_frame_ / 1000.0  # Convert to meters
            else:
                self.curr_depth_ = self.depth_frame_ / 1000.0

    # ODOMETRY CALLBACK - Get previous odom at first, then current odom and compute T_odom with velocity and time interval
    def odom_callback(self,msg):
        if (msg.twist.twist.linear.x is not None or msg.twist.twist.linear.y is not None or msg.twist.twist.linear.z is not None or msg.twist.twist.angular.z is not None):
            if(self.prev_trans_ is None and self.curr_trans_ is None and self.prev_quat_ is None and self.curr_quat_ is None):
                self.prev_trans_ = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                self.prev_quat_ = msg.twist.twist.angular.z
                self.time_prev = time.time()

            else:
                self.curr_trans_ = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
                self.curr_quat_ = msg.twist.twist.angular.z
                self.time_curr = time.time()
                diff_time = (self.time_curr - self.time_prev)
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
        # If the transformation is from odom to base_link, process it
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

    # PROCESS OPTICAL FLOW - Main function to compute optical flow, ego-motion compensation, residual flow, and moving object mask
    def process_optflow(self):
        # Get both previous and current frames
        image_0 = preprocess_frame(self.prev_frame_)
        image_1 = preprocess_frame(self.curr_frame_)

        # Keep depth of previous frame
        Z = self.prev_depth_
        H_d, W_d = Z.shape

        # Pixel grid 
        u, v = np.meshgrid(np.arange(W_d), np.arange(H_d))  # (H, W)
        # Smooth depth to reduce noise
        Z = cv2.medianBlur((Z * 1000).astype(np.uint16), 3).astype(np.float32) / 1000.0
        
        # 3D points in camera frame (meters) using pixel grid and depth
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        points_3d = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)  # (H, W, 4)

        # Get total transformations from odom coordinate frame to camera frame at previous and current time steps
        T_prev = self.T_odom_prev @ self.prev_T_ @ self.T_to_cam
        if self.T_odom_curr is None:
            T_curr = np.eye(4) @ self.curr_T_ @ self.T_to_cam
        else:
            T_curr = self.T_odom_curr @ self.curr_T_ @ self.T_to_cam

        # Transform points from previous camera frame to world (odom) frame, then to current camera frame
        points_world = points_3d @ np.linalg.inv(T_prev)
        points_curr = points_world @ T_curr

        # Get position in x,y,z of each 3D point in current time step
        X_new = points_curr[..., 0]
        Y_new = points_curr[..., 1]
        Z_new = points_curr[..., 2]

        # avoid division by zero / negative depth and non-valid regions for depth
        valid_mask = (Z > 0.5) & (Z_new > 0.5) &  (Z < 4.5) & (Z_new < 4.5) & np.isfinite(Z_new) & np.isfinite(Z)

        # print(valid_mask.min(), valid_mask.max(), Z_new.min(), Z_new.max(), np.count_nonzero(valid_mask), valid_mask.size)
        
        # initialize u_new, v_new with original grid so invalid pixels map to zero motion
        u_new = np.zeros_like(X_new, dtype=np.float32)
        v_new = np.zeros_like(Y_new, dtype=np.float32)
        # safe reprojection only where valid
        u_new[valid_mask] = (self.fx * X_new[valid_mask] / Z_new[valid_mask]) + self.cx
        v_new[valid_mask] = (self.fy * Y_new[valid_mask] / Z_new[valid_mask]) + self.cy

        # ego flow in pixel units
        flow_ego = np.stack([u - u_new, v - v_new], axis=-1).astype(np.float32)

        with torch.no_grad():
            flow = self.model(image_0, image_1)[-1][0]         # (2, H, W)
            flow = flow.permute(1, 2, 0).cpu().numpy()         # (H, W, 2)

        # ensure same dtype
        flow = flow.astype(np.float32)

        # residual flow and magnitudes
        flow_res = flow - flow_ego                        # (H, W, 2)
        flow_residual = flow_res
        flow_res_mag = np.linalg.norm(flow_residual, axis=-1)  # (H, W)
        ego_flow_mag = np.linalg.norm(flow_ego, axis=-1)
        flow_mag = np.linalg.norm(flow, axis=-1)


        # mask invalid depth pixels (avoid contaminating results)
        flow_res_mag[~valid_mask] = 0.0
        ego_flow_mag[~valid_mask] = 0.0
        flow_mag[~valid_mask] = 0.0

        # flow color visual and cut it off base on magnitude on flow - to avoid noise introduction when stoped
        flow_color = flow_viz.flow_to_image(flow_residual)  # (H,W,3) uint8
        flow_color[flow_mag < 0.5] = 0  # mask low-magnitude residuals for better visualization

        # Transform into a binary segmentation to join it with residual mask
        flow_color = cv2.cvtColor(flow_color, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        _, flow_color = cv2.threshold(flow_color, 1, 255, cv2.THRESH_BINARY)

        # normalize and threshold residual magnitude for detection
        flow_res_norm = cv2.normalize(flow_res_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # choose sensible threshold (tuneable).
        _, moving_mask = cv2.threshold(flow_res_norm, 80, 255, cv2.THRESH_BINARY)

        # Remove the noise when robot is static joinin residual mask with flow_color binary mask
        moving_mask = cv2.bitwise_and(flow_color, moving_mask, mask = None)

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_OPEN, kernel)
        moving_mask = cv2.morphologyEx(moving_mask, cv2.MORPH_DILATE, kernel)

        # Visual debug windows 
        # ego_vis = cv2.normalize(ego_flow_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # flow_vis = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # res_vis = flow_res_norm

        # Get floor segmentation
        floor_seg = self.process_slic()

        # Remove floor segmentaiton from the resulted mask
        floor_seg_not = cv2.bitwise_not(floor_seg)
        bottom_flow = cv2.bitwise_and(moving_mask[int(image_height/2):,:], floor_seg_not)
        # Split the top part
        top_part = moving_mask[:int(image_height/2), :]
        # Combine top and processed bottom
        combined_vis = np.vstack((top_part, bottom_flow))

        # publish mask as single-channel 8-bit image
        combined_vis_msg = self.bridge.cv2_to_imgmsg(combined_vis, encoding="mono8")
        self.opt_flow_pub_.publish(combined_vis_msg)
        self.get_logger().info("OpticalFlow mask published.")

        # depth_display = cv2.normalize(Z_new, None, 0, 255, cv2.NORM_MINMAX)
        # depth_display = depth_display.astype(np.uint8)

        # Update prev variables
        self.prev_frame_ = self.curr_frame_
        self.prev_depth_ = self.curr_depth_

        self.prev_quat_ = self.curr_quat_
        self.prev_trans_ = self.curr_trans_
        self.time_prev = self.time_curr
        self.T_odom_prev = self.T_odom_curr

        self.prev_T_ = self.curr_T_

        # Visualize results
        # cv2.imshow("Depth (resized)", depth_vis)
        # cv2.imshow("Ego flow magnitude", ego_vis)
        # cv2.imshow("Flow magnitude", flow_vis)
        # cv2.imshow("Residual magnitude (norm)", res_vis)
        # # cv2.imshow("Residual flow (color)", flow_color)
        # cv2.imshow("Moving mask", moving_mask)
        cv2.imshow("Final OpticalFlow", combined_vis)
        cv2.imshow("Original image", self.prev_frame_)
        # cv2.imshow("Depth image", depth_display)
        cv2.waitKey(1)


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
        A = y2 - y1
        B = x1 - x2
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

def preprocess_frame(frame):
    tensor = torch.from_numpy(frame).permute(2, 0, 1).half()
    return tensor[None].cuda()

def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def main(args=None):
    # Start the fusion node
    try:
        rclpy.init(args=args)
        opt_flow_node = OpticalFlow()

        rclpy.spin(opt_flow_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if opt_flow_node is not None:
            opt_flow_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
            cv2.destroyAllWindows()
            
        opt_flow_node.get_logger().info("OpticalFlow node has been shut down.")

if __name__ == "__main__":
    # Call Main Function
    main()