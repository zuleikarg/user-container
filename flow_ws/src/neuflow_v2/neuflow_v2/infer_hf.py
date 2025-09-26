import torch
import os
import numpy as np
import cv2
from neuflow_v2.NeuFlow.neuflow import NeuFlow
from neuflow_v2.NeuFlow.backbone_v7 import ConvBlock
from neuflow_v2.data_utils import flow_viz
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


image_width = 512
image_height = 288
vis_path = 'camera_results/'

class OpticalFlow(Node):

    def __init__(self):
        super().__init__('infer_hf')

        self.opt_flow_pub_ = self.create_publisher(Image, 'opticalflow', 10)
        self.bridge = CvBridge()
        self.camera_frame_ = np.empty(0)
        self.prev_frame_ = None
        
        self.curr_frame_ = None

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

        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        # Start camera
        #cap = cv2.VideoCapture('/dev/video6')  # Use 0 for default webcam

        self.frame_count_ = 0


        self.subscription = self.create_subscription(
            Image,
            'camera',
            self.camera_callback,
            10)

    def camera_callback(self,msg):
        if (msg.height != 0):
            self.camera_frame_ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            if(self.prev_frame_ is None and self.curr_frame_ is None):
                self.prev_frame_ = np.copy(self.camera_frame_)
            elif(self.curr_frame_ is None):
                self.curr_frame_ = np.copy(self.camera_frame_)

            if(self.prev_frame_ is not None and self.curr_frame_ is not None):
                self.process_optflow()
            

    def process_optflow(self):
        image_0 = preprocess_frame(self.prev_frame_)
        image_1 = preprocess_frame(self.curr_frame_)

        with torch.no_grad():
            flow = self.model(image_0, image_1)[-1][0]
            flow = flow.permute(1, 2, 0).cpu().numpy()
            flow_img = flow_viz.flow_to_image(flow)
            flow_magnitude = np.linalg.norm(flow, axis=2)
            #print(np.max(flow_magnitude))
            flow_img[flow_magnitude < 2.0] = [0, 0, 0]  # Suppress low-motion areas
                    

            # stacked = np.vstack([
            #     cv2.resize(prev_frame, (image_width, image_height)),
            #     flow_img
            # ])

            # convert the input image to grayscale
        kernel = np.ones((5, 5), np.uint8)
        gray = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)

            # apply thresholding to convert grayscale to binary image
        ret,thresh = cv2.threshold(gray,1,255,0)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        msg = Image()
        msg = self.bridge.cv2_to_imgmsg(thresh,encoding="mono8")
        self.opt_flow_pub_.publish(msg)
        self.get_logger().info("OpticalFlow image sent.")

        cv2.imshow("Optical Flow", thresh)
            # cv2.imwrite(f"{vis_path}/frame_{frame_count:04d}.jpg", stacked)

        self.frame_count_ += 1
        self.prev_frame_ = np.copy(self.curr_frame_)
        self.curr_frame_ = None

        cv2.waitKey(1)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (image_width, image_height))
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