#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion


class PoseEvery5s(Node):

    def __init__(self):
        super().__init__('pose_every_5s')

        self.latest_odom = None

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Timer: every 5 seconds (ROS time aware)
        self.create_timer(5.0, self.timer_callback)

        self.get_logger().info("📍 Pose printer every 5 seconds started")

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def timer_callback(self):
        if self.latest_odom is None:
            self.get_logger().warn("No odom received yet")
            return

        pos = self.latest_odom.pose.pose.position
        ori = self.latest_odom.pose.pose.orientation

        _, _, yaw = euler_from_quaternion(
            [ori.x, ori.y, ori.z, ori.w]
        )

        self.get_logger().info(
            f"x={pos.x:.2f}, y={pos.y:.2f}, yaw={yaw:.2f} rad"
        )


def main():
    rclpy.init()
    node = PoseEvery5s()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
