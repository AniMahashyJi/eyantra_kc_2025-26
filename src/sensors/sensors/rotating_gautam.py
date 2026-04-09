#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from linkattacher_msgs.srv import AttachLink, DetachLink


class JointServoPID(Node):
    def __init__(self):
        super().__init__('joint_servo_node')

        self.pub = self.create_publisher(
            Float64MultiArray, '/delta_joint_cmds', 10
        )

        self.sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10
        )

        self.attach_cli = self.create_client(AttachLink, '/attach_link')
        self.detach_cli = self.create_client(DetachLink, '/detach_link')

        self.attached = False
        self.detached = False

        self.attach_future = None
        self.detach_future = None

        self.waiting_for_attach = False
        self.attach_wait_start = None


        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.waypoints = [
            (10.77, [-3.140,-0.590,-2.490,-0.057,1.570,3.150]), #23
            (10.77, [-3.140,-0.590,-2.490,-0.057,1.570,3.150]), #23
            (10.77, [-3.140,-0.590,-2.490,-0.057,1.570,3.150]), #23
            (10.77, [-3.140,-0.590,-2.490,-0.057,1.570,3.150]), #23
            (10.77, [-3.140,-0.590,-2.490,-0.057,1.570,3.150]), #23
            (10.77, [-3.140,-0.590,-2.490,-0.057,1.570,3.150]), #23
            (10.77, [-3.140,-0.590,-2.490,-0.057,1.570,3.150]), #23

        ]

        self.current = [0.0] * 6
        self.have_state = False

        self.kp = 3
        self.kd = 0.5
        self.prev_error = [0.0] * 6
        self.max_step = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        self.wp_idx = 0

        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info(
            "Joint Servo PID started. Call /servo_node/start_servo first!"
        )

    def joint_state_cb(self, msg):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                self.current[i] = msg.position[msg.name.index(name)]
        self.have_state = True

    # ---------------- ATTACH / DETACH ----------------

    def attach_can(self):
        if not self.attach_cli.service_is_ready():
            self.get_logger().warn("Attach service not ready")
            return

        req = AttachLink.Request()
        req.model1_name = 'fertiliser_can'
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'

        self.attach_future = self.attach_cli.call_async(req)

    def detach_can(self):
        if not self.detach_cli.service_is_ready():
            self.get_logger().warn("Detach service not ready")
            return

        req = DetachLink.Request()
        req.model1_name = 'fertiliser_can'
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'

        self.detach_future = self.detach_cli.call_async(req)

    # ---------------- CONTROL LOOP ----------------

    def control_loop(self):
        if not self.have_state:
            return
        
        # ---- Wait before attach ----
        if self.waiting_for_attach and not self.attached:
            elapsed = (
                self.get_clock().now() - self.attach_wait_start
            ).nanoseconds * 1e-9

            if elapsed >= 1.5:
                self.get_logger().info("⏱ Attaching fertilizer can now")
                self.attach_can()
                self.waiting_for_attach = False

            return


        # ---- Attach response ----
        if self.attach_future and self.attach_future.done():
            if self.attach_future.result().success:
                self.get_logger().info("✅ Fertilizer can ATTACHED")
                self.attached = True
            else:
                self.get_logger().warn("❌ Failed to attach")
            self.attach_future = None

        # ---- Detach response ----
        if self.detach_future and self.detach_future.done():
            if self.detach_future.result().success:
                self.get_logger().info("✅ Fertilizer can DETACHED")
                self.detached = True
                self.timer.cancel()          # <<< STOP ONLY AFTER DETACH
            else:
                self.get_logger().warn("❌ Failed to detach")
            self.detach_future = None
            return

        if self.wp_idx >= len(self.waypoints) - 1:
            if self.attached and not self.detached and self.detach_future is None:
                self.get_logger().info("All waypoints completed → detaching")
                self.detach_can()
            return

        _, pos1 = self.waypoints[self.wp_idx + 1]

        deltas = []
        for i in range(6):
            error = pos1[i] - self.current[i]
            derr = error - self.prev_error[i]
            step = self.kp * error + self.kd * derr
            step = max(min(step, self.max_step[i]), -self.max_step[i])
            deltas.append(step)
            self.prev_error[i] = error

        msg = Float64MultiArray()
        msg.data = deltas
        self.pub.publish(msg)

        if all(abs(c - p1) < 0.04 for c, p1 in zip(self.current, pos1)):
            self.wp_idx += 1


            self.get_logger().info(
                f"🎯 Reached waypoint {self.wp_idx} / {len(self.waypoints)-1}"
            )



            if self.wp_idx == 5 and not self.attached and not self.waiting_for_attach:
                self.waiting_for_attach = True
                self.attach_wait_start = self.get_clock().now()
                self.get_logger().info("⏸ Waiting 1.5s before attaching fertilizer can")



def main():
    rclpy.init()
    node = JointServoPID()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()