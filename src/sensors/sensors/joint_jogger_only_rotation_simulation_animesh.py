#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math


class WaypointJointReplaySim(Node):
    def __init__(self):
        super().__init__('waypoint_joint_replay_sim')

        # -------- PARAMETERS (SAFE FOR SIM) --------
        self.joint_tol = 0.17          # rad
        self.delta_step = 0.5          # rad per publish
        self.publish_period = 0.1      # seconds (10 Hz)
        self.print_period = 3.0        # seconds
        # ------------------------------------------

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # -------- BUILT-IN WAYPOINTS --------
        self.trajectory = [
           [-5.131,-1.436,-1.535,-0.405,2.403,2.487],
           [-5.177,-1.289,-1.255,-2.069,1.768,1.168],
           [-4.579,-1.101,-1.346,-2.221,1.648,1.223],
           [-3.806,-1.145,-1.326,-2.202,1.652,1.148],
           [-2.852,-0.980,-1.021,-2.685,1.569,2.899],
           [-2.983,-1.444,-1.473,-1.789,1.570,3.185],
           [-2.991,-1.897,-1.210,-1.605,1.571,3.278],
        ]
        # -----------------------------------

        self.current_joint_pos = None
        self.last_cmd = [0.0] * 6

        self.target_index = 0
        self.active_joint = 0

        # Subscriber
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            10
        )

        # Simulation publisher
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/delta_joint_cmds',
            10
        )

        # Timers
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
            f"Effective joint velocity = {self.effective_velocity:.3f} rad/s"
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

        # Delta jogging (SIMULATION)
        delta = math.copysign(self.delta_step, err)

        cmd = Float64MultiArray()
        cmd.data = [0.0] * 6
        cmd.data[j] = delta

        self.last_cmd = list(cmd.data)
        self.pub.publish(cmd)

    # ---------------- LOGGING ----------------

    def print_status(self):
        self.get_logger().info(
            f"[VEL={self.effective_velocity:.3f} rad/s] "
            f"Active joint: J{self.active_joint + 1} | "
            f"Cmd: {['%.3f' % v for v in self.last_cmd]}"
        )

    # ---------------- UTIL ----------------

    def send_zero(self):
        cmd = Float64MultiArray()
        cmd.data = [0.0] * 6
        self.last_cmd = list(cmd.data)
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = WaypointJointReplaySim()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
