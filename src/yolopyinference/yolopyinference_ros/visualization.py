#!/usr/bin/env python3

import cv2
import cv_bridge
import message_filters
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class ImageVisualizer(Node):
    QUEUE_SIZE = 100
    color = (0, 255, 0)
    bbox_thickness = 3

    def __init__(self):
        super().__init__('image_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._processed_image_pub = self.create_publisher(
            Image, 'processed_image',  self.QUEUE_SIZE)

        self._image_subscription = message_filters.Subscriber(
            self,
            Image,
            'camera/color/image_raw')

        self.time_synchronizer = message_filters.TimeSynchronizer(
            [self._image_subscription],
            self.QUEUE_SIZE)

        self.time_synchronizer.registerCallback(self.detections_callback)

    def detections_callback(self, img_msg):
        txt_color=(255, 0, 255)
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg)
        print(cv2_img.shape)#(720, 1280, 3) from realsense

        processed_img = self._bridge.cv2_to_imgmsg(
            cv2_img, encoding=img_msg.encoding)
        self._processed_image_pub.publish(processed_img)


def main():
    rclpy.init()
    rclpy.spin(ImageVisualizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()