#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointJog
import csv
import math


class CSVJointReplay(Node):
    def __init__(self):
        super().__init__('csv_joint_replay')

        # -------- PARAMETERS --------
        self.csv_path = '/home/ee1240935/fullrun.csv'
        self.downsample = 2
        self.joint_tol = 0.17
        self.delta_step = 5.0          # rad
        self.publish_period = 0.1      # seconds
        self.print_period = 3.0
        # ----------------------------

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.current_joint_pos = None
        self.last_vel_cmd = [0.0] * 6

        # Subscriber
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            10
        )

        # ✅ HARDWARE publisher
        self.pub = self.create_publisher(
            JointJog,
            '/delta_joint_cmds',
            10
        )

        # Load CSV
        self.trajectory = self.load_csv()
        self.target_index = 0
        self.active_joint = 0

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
            f"Loaded {len(self.trajectory)} poses | "
            f"Joint velocity = {self.effective_velocity:.3f} rad/s"
        )

    # ---------------- CSV ----------------

    def load_csv(self):
        traj = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for i in range(0, len(rows), self.downsample):
            row = rows[i]
            traj.append([float(row[j]) for j in self.joint_names])

        return traj

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
            self.get_logger().info("✅ Trajectory completed")
            rclpy.shutdown()
            return

        q_des = self.trajectory[self.target_index]
        q_cur = self.current_joint_pos

        j = self.active_joint
        err = q_des[j] - q_cur[j]

        if abs(err) < self.joint_tol:
            self.send_zero()
            self.active_joint += 1
            if self.active_joint >= 6:
                self.active_joint = 0
                self.target_index += 1
            return

        # Δq → velocity
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
            f"Velocities: {['%.3f' % v for v in self.last_vel_cmd]}"
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
    node = CSVJointReplay()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
