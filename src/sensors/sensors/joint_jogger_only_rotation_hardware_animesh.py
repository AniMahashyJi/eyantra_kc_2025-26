#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointJog
import math


class WaypointJointReplay(Node):
    def __init__(self):
        super().__init__('waypoint_joint_replay')

        # -------- PARAMETERS (SET SAFE FOR HARDWARE) --------
        self.joint_tol = 0.            # rad
        self.delta_step = 0.9           # rad
        self.publish_period = 0.1        # seconds
        self.print_period = 3.0          # seconds
        # ---------------------------------------------------

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # -------- WAYPOINTS (FROM YOUR CSV) --------
        self.trajectory = [
            [-3.139994, -0.589978, -2.490028, -0.057027, 1.569993, 3.150000],
            [-3.356007, -0.433283, -2.498856, -0.208168, 1.442954, 3.149216],
            [-3.925166, -0.436361, -2.498231, -0.203740, 1.360202, 3.148498],
            [-4.142793, -0.432856, -2.497828, -0.206701, 1.260283, 3.148249],
            [-4.209146, -0.438970, -2.497461, -0.220501, 1.187313, 3.148026],
            [-4.183461, -0.436494, -2.496472, -0.253613, 1.096558, 3.147606],
            [-4.360000, -0.436000, -2.496000, -0.253000, 1.100000, 3.148000],
        ]
        # ------------------------------------------

        self.current_joint_pos = None
        self.last_vel_cmd = [0.0] * 6

        self.target_index = 0
        self.active_joint = 0

        # Subscriber
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            10
        )

        # Hardware publisher
        self.pub = self.create_publisher(
            JointJog,
            '/delta_joint_cmds',
            10
        )

        self.control_timer = self.create_timer(
            self.publish_period,
            self.control_loop
        )

        self.print_timer = self.create_timer(
            self.print_period,
            self.print_status
        )

        self.effective_velocity = self.delta_step / self.publish_period

        self.get_logger().info(
            f"Loaded {len(self.trajectory)} waypoints | "
            f"Joint velocity = {self.effective_velocity:.3f} rad/s"
        )

    # ---------------- CALLBACK ----------------

    def joint_state_cb(self, msg):
        pos_map = dict(zip(msg.name, msg.position))
        self.current_joint_pos = [pos_map[j] for j in self.joint_names]

    # ---------------- CONTROL ----------------

    def control_loop(self):
        if self.current_joint_pos is None:
            return

        if self.target_index >= len(self.trajectory):
            self.send_zero()
            self.get_logger().info("✅ Waypoint execution completed")
            rclpy.shutdown()
            return

        q_des = self.trajectory[self.target_index]
        q_cur = self.current_joint_pos

        j = self.active_joint
        err = q_des[j] - q_cur[j]

        # Joint reached target
        if abs(err) < self.joint_tol:
            self.send_zero()
            self.active_joint += 1

            if self.active_joint >= 6:
                self.active_joint = 0
                self.target_index += 1
                self.get_logger().info(f"➡ Moving to waypoint {self.target_index}")
            return

        # Convert delta to velocity
        vel = math.copysign(self.delta_step / self.publish_period, err)

        joint_cmd = JointJog()
        joint_cmd.joint_names = self.joint_names
        joint_cmd.velocities = [0.0] * 6
        joint_cmd.velocities[j] = vel

        self.last_vel_cmd = joint_cmd.velocities.copy()
        self.pub.publish(joint_cmd)

    # ---------------- LOGGING ----------------

    def print_status(self):
        self.get_logger().info(
            f"[VEL={self.effective_velocity:.3f} rad/s] "
            f"Active joint: J{self.active_joint + 1} | "
            f"Vel: {['%.3f' % v for v in self.last_vel_cmd]}"
        )

    # ---------------- UTIL ----------------

    def send_zero(self):
        joint_cmd = JointJog()
        joint_cmd.joint_names = self.joint_names
        joint_cmd.velocities = [0.0] * 6
        self.last_vel_cmd = [0.0] * 6
        self.pub.publish(joint_cmd)


def main():
    rclpy.init()
    node = WaypointJointReplay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()