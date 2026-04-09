#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
import time


class JointLogger(Node):
    def __init__(self):
        super().__init__('joint_logger')

        self.latest_msg = None

        # Subscribe to joint states
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_cb,
            10
        )

        # Timer: log every 2 seconds
        self.timer = self.create_timer(2.0, self.log_joint_state)

        # CSV file setup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f'joint_log_{timestamp}.csv'
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        self.header_written = False

        self.get_logger().info(
            f"📄 Logging joint angles every 2s to {self.filename} (CSV + terminal)"
        )

    def joint_cb(self, msg: JointState):
        self.latest_msg = msg

    def log_joint_state(self):
        if self.latest_msg is None:
            return

        msg = self.latest_msg

        # Timestamp (ROS time if available)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Write CSV header once
        if not self.header_written:
            header = ['time'] + list(msg.name)
            self.writer.writerow(header)
            self.header_written = True

        # CSV row
        row = [f"{t:.3f}"] + [f"{p:.6f}" for p in msg.position]
        self.writer.writerow(row)
        self.file.flush()

        # Terminal log (human readable)
        joint_str = " | ".join(
            f"{name}={pos:.3f}"
            for name, pos in zip(msg.name, msg.position)
        )

        self.get_logger().info(f"[t={t:.3f}] {joint_str}")

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = JointLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
