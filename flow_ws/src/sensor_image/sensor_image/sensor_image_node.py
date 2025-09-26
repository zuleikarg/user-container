import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):  

    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera', 10)
        timer_period = 0.1  # seconds
        
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture('/dev/video6')
        
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Image()
        
        res,frame = self.cap.read()
        if res:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
            self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    camera_publisher = CameraPublisher()

    rclpy.spin(camera_publisher)

    camera_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()