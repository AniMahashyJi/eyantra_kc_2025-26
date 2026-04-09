#!/usr/bin/env python3
"""
===============================================================================
TEAM ID: 2635
TEAM MEMBERS: Animesh, Swayam, Harshit, Gautam
===============================================================================

FILE NAME: ebot_nav_task3b.py
DESCRIPTION:
    ROS2 Navigation node for the EBot robot with shape detection integration.
    Task 3B version with /arm_flag coordination - USES GAZEBO SIMULATION TIME
    MODIFIED: Shape detection is stored and published when waypoint is reached
===============================================================================
"""

# ===================== IMPORTS =====================
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf_transformations import euler_from_quaternion
import math


# ===================== MAIN NODE CLASS =====================
class EBotNav(Node):
    """
    ROS2 Node that handles:
        - Waypoint navigation
        - Obstacle avoidance
        - Shape detection and response (published at waypoint arrival)
        - Orientation alignment
        - Waiting at dock for CAN_RELEASED from arm
        - USES GAZEBO SIMULATION TIME
    """

    def __init__(self):
        """Initialize subscriptions, publishers, state variables, and timers."""
        super().__init__('ebot_nav_task3b_sim_time')
        self.align_direction = None

        # ----------------- Arm flag state -----------------
        self.waiting_for_can_release = False
        self.can_released = True
        self.can_released_time: Time = None
        self.post_release_delay = Duration(seconds=2.0)

        # -------- Arm Flag Subscriber --------
        self.sub_arm_flag = self.create_subscription(
            String, '/arm_flag', self.arm_flag_cb, 10
        )

        # ----------------- Subscribers -----------------
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.sub_shape = self.create_subscription(
            String, '/detected_shape', self.shape_callback, 10
        )

        # ----------------- Publishers -------------------
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_enable_detection = self.create_publisher(Bool, '/enable_detection', 10)
        self.pub_filter_negative = self.create_publisher(Bool, '/filter_negative_theta', 10)
        self.pub_detection_status = self.create_publisher(String, '/detection_status', 10)

        # ----------------- Robot State -----------------
        self.robot_x: float = None
        self.robot_y: float = None
        self.robot_theta: float = None
        self.scan_data: LaserScan = None

        # ----------------- Shape Detection State (MODIFIED) -------
        # Store pending detections to publish at waypoint arrival
        self.pending_detection = None  # Will store: {'shape': str, 'local_x': float, 'local_y': float, 'waypoint': int}
        self.waypoint_stop_triggered = False  # Flag for stopping at waypoint with detection
        self.waypoint_stop_start_time: Time = None  # Time when stop started
        
        # ----------------- Dock Detection State (using ROS Time) --------
        self.dock_stop_triggered = False
        self.dock_stop_start_time: Time = None
        self.dock_announced = False

        # ----------------- Waypoints -------------------
        self.waypoints = [
            [-0.15, -5.6, 0.0],              # 0: waypoint1
            [0.32, -4.094+0.20, 1.57],      # 1: plant1
            [0.32, -2.77+0.20, 1.57],       # 2: plant2
            [0.477, -1.78, 1.75],           # 3: dock
            [0.32, -1.4044+0.20, 1.57],     # 4: plant3
            [0.32, -0.046+0.20, 1.57],      # 5: plant4
            [0.26, 0.9, 1.57],              # 6: waypoint2
            [-2.8, 1.37, -3.1],             # 7: waypoint3
            [-3.3, -0.046-0.20, -1.57],     # 8: plant8
            [-3.3, -1.4044-0.20, -1.57],    # 9: plant7
            [-3.3, -2.77-0.20, -1.57],      # 10: plant6
            [-3.3, -4.094-0.20, -1.57],     # 11: plant5
            [-3.3, -5.4, -1.57],            # 12: waypoint4
           # [-3.3, -5.4, 0.0],            # same but change orientation
            [-1.5, -5.4, 0.0],             # 13: waypoint5
            [-1.5, -5.4, 1.57],             # same but change orientation
            [-1.48, -4.094+0.20, 1.57],     # 14: plant1,5
            [-1.48, -2.77+0.20, 1.57],      # 15: plant2,6
            [-1.48, -1.4044+0.20, 1.57],    # 16: plant3,7
            [-1.48, -0.046+0.20, 1.57],     # 17: plant4,8
            [-1.48, 1.5, 1.57],             # 18
            [-1.53, -6.61, 1.57]            # 19
        ]
        self.current_wp = 0

        # ----------------- Control Gains ----------------
        self.k_lin = 0.5
        self.k_ang = 1.5
        self.k_theta = 1.0

        # ----------------- Tolerances -------------------
        self.pos_tol_x = 0.2
        self.pos_tol_y = 0.2
        self.theta_tol = math.radians(10)

        # ----------------- Timer (Control Loop) ---------
        self.timer = self.create_timer(0.1, self.control_loop)
        self.debug_counter = 0

        self.get_logger().info("EBot navigation node (Task3B - Publish at Waypoint) started.")

    # ===================== CALLBACKS =====================
    def arm_flag_cb(self, msg: String):
        """Handle CAN_RELEASED from arm."""
        if msg.data == "CAN_RELEASED":
            self.get_logger().info("Arm: CAN_RELEASED received, waiting 2 sec before resuming nav")
            self.can_released = True
            self.can_released_time = self.get_clock().now()
            self.waiting_for_can_release = False

    def odom_callback(self, odom_msg: Odometry):
        """Update robot pose from odometry."""
        self.robot_x = odom_msg.pose.pose.position.x
        self.robot_y = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        (_, _, self.robot_theta) = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def scan_callback(self, scan_msg: LaserScan):
        """Update LiDAR scan data."""
        self.scan_data = scan_msg

    def shape_callback(self, msg):
        
        if self.waiting_for_can_release:
            return

        if self.pending_detection is not None:
            return

        if self.current_wp >= len(self.waypoints):
            return

        try:
            shape, lx, ly = msg.data.split('|')
            lx = float(lx)
            ly = float(ly)

            # Assign waypoint
            if self.current_wp == 3:      # dock
                assigned_wp = self.current_wp + 1
            else:
                assigned_wp = self.current_wp

            assigned_wp = min(assigned_wp, len(self.waypoints) - 1)

            self.pending_detection = {
                'shape': shape,
                'local_x': lx,
                'local_y': ly,
                'waypoint': assigned_wp
            }

            self.get_logger().info(
                f"Shape detected: {shape} → assigned WP {assigned_wp}"
            )

        except Exception as e:
            self.get_logger().error(f"Shape parse error: {e}")

    # ===================== HELPER METHODS =====================
    def calculate_shape_position(self, local_y):
        """Approximate global position of detected shape from local offsets."""
        if self.robot_theta > 0:
            global_x = self.robot_x
            global_y = self.robot_y
        else:
            global_y = self.robot_y
            if local_y < 0:
                global_x = self.robot_x
            else:
                global_x = self.robot_x

        return global_x, global_y

    def publish_pending_detection(self):
        """Publish the pending shape detection."""
        if self.pending_detection is None:
            return
        
        shape_type = self.pending_detection['shape']
        local_y = self.pending_detection['local_y']
        
        # Determine status based on shape
        if shape_type == "TRIANGLE":
            shape_status = "FERTILIZER_REQUIRED"
        elif shape_type == "SQUARE":
            shape_status = "BAD_HEALTH"
        elif shape_type == "PENTAGON":
            shape_status = "DOCK_STATION"
        else:
            shape_status = "UNKNOWN"
        
        # Calculate position
        global_x, global_y = self.calculate_shape_position(local_y)
        plant_id = self.get_plant_id(global_x, global_y, local_y)
        
        # Publish detection
        shape_msg = String()
        shape_msg.data = f"{shape_status},{global_x:.2f},{global_y:.2f},{plant_id}"
        self.pub_detection_status.publish(shape_msg)
        self.get_logger().info(f"✅ Published at waypoint arrival: {shape_msg.data}")
        
        # Clear pending detection
        self.pending_detection = None

    # ===================== CONTROL LOOP =====================
    def control_loop(self):
        """Main navigation + detection control loop."""

        # Get current simulation time
        current_time = self.get_clock().now()

        # ---------- While arm is working at dock, keep robot stopped ----------
        if self.waiting_for_can_release and not self.can_released:
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.pub_cmd.publish(stop)
            enable_msg = Bool()
            enable_msg.data = False
            self.pub_enable_detection.publish(enable_msg)
            return
            
        # ---------- Wait 2 seconds after CAN_RELEASED before moving ----------
        if self.can_released and self.can_released_time is not None:
            elapsed = current_time - self.can_released_time
            if elapsed < self.post_release_delay:
                stop = Twist()
                stop.linear.x = 0.0
                stop.angular.z = 0.0
                self.pub_cmd.publish(stop)

                enable_msg = Bool()
                enable_msg.data = True
                self.pub_enable_detection.publish(enable_msg)
                return
            else:
                self.get_logger().info("2-second post-release wait done → navigation resumed")
                self.can_released_time = None

        if None in [self.robot_x, self.robot_y, self.robot_theta, self.scan_data]:
            return

        if self.current_wp >= len(self.waypoints):
            twist_msg = Twist()
            self.pub_cmd.publish(twist_msg)
            self.get_logger().info("All waypoints reached!")
            return

        goal_x, goal_y, goal_theta = self.waypoints[self.current_wp]

        # Set filter based on waypoint
        if self.current_wp in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            filter_msg = Bool()
            filter_msg.data = True
            self.pub_filter_negative.publish(filter_msg)
        else:
            filter_msg = Bool()
            filter_msg.data = False
            self.pub_filter_negative.publish(filter_msg)

        dx = goal_x - self.robot_x
        dy = goal_y - self.robot_y
        distance_error = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(desired_heading - self.robot_theta)

        use_reverse = False
        if self.current_wp == len(self.waypoints)-1:
            heading_error_forward = heading_error
            heading_error_backward = self.normalize_angle(heading_error_forward + math.pi)
            if abs(heading_error_backward) < abs(heading_error_forward):
                use_reverse = True
                heading_error = heading_error_backward

        twist_msg = Twist()

        # Enable shape detection (except at last waypoint)
        enable_msg = Bool()
        enable_msg.data = self.current_wp < len(self.waypoints)-1
        self.pub_enable_detection.publish(enable_msg)

        # ----------- Dock stop logic -----------
        if self.dock_stop_triggered:
            elapsed = current_time - self.dock_stop_start_time
            elapsed_seconds = elapsed.nanoseconds / 1e9

            if elapsed_seconds >= 2.0:
                self.dock_stop_triggered = False
                self.dock_announced = True
                self.get_logger().info("Dock stop complete")
            else:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.pub_cmd.publish(twist_msg)
                return

        # ----------- Waypoint stop logic (for shape detection) -----------
        if self.waypoint_stop_triggered:
            elapsed = current_time - self.waypoint_stop_start_time
            elapsed_seconds = elapsed.nanoseconds / 1e9

            if elapsed_seconds >= 2.0:
                # 2 seconds elapsed - publish detection and move to next waypoint
                if self.pending_detection is not None and self.pending_detection['waypoint'] == self.current_wp:
                    self.publish_pending_detection()
                
                self.get_logger().info(
                    f"Waypoint {self.current_wp + 1} stop complete, moving to next waypoint"
                )
                
                self.waypoint_stop_triggered = False
                self.current_wp += 1
                self.dock_announced = False
                self.get_logger().info(f"Moving to waypoint {self.current_wp + 1}")
                
                # IMPORTANT: Return here to let next cycle handle the new waypoint
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.pub_cmd.publish(twist_msg)
                return
            else:
                # Still waiting - keep robot stopped
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.pub_cmd.publish(twist_msg)
                return

        # ---------- DOCK STOP + WAIT FOR ARM ----------
        if self.current_wp == 3 and not self.dock_announced:
            at_position = (abs(dx) <= self.pos_tol_x) and (abs(dy) <= self.pos_tol_y)

            if at_position and not self.dock_stop_triggered:
                orientation_error = self.normalize_angle(goal_theta - self.robot_theta)
                if abs(orientation_error) > self.theta_tol:
                    twist_align = Twist()
                    twist_align.linear.x = 0.0
                    max_ang = 1.0
                    twist_align.angular.z = max(min(self.k_theta * orientation_error, max_ang), -max_ang)
                    self.get_logger().info(
                        f"Aligning at dock before wait | Error: {math.degrees(orientation_error):.2f}°"
                    )
                    self.pub_cmd.publish(twist_align)
                    return

                global_x = self.robot_x
                global_y = self.robot_y
                status_msg = String()
                status_msg.data = f"DOCK_STATION,{global_x:.2f},{global_y:.2f},0"
                self.pub_detection_status.publish(status_msg)
                self.get_logger().info(f"Dock reached & aligned → published: {status_msg.data}")

                self.waiting_for_can_release = True
                self.can_released = False

                self.dock_stop_triggered = True
                self.dock_stop_start_time = current_time
                return

        # ----------- Obstacle avoidance -----------
        scan_msg = self.scan_data
        front_angle_rad = math.radians(60)
        start_idx = int((0 - front_angle_rad - scan_msg.angle_min) /
                        scan_msg.angle_increment)
        end_idx = int((0 + front_angle_rad - scan_msg.angle_min) /
                      scan_msg.angle_increment)

        front_ranges = scan_msg.ranges[start_idx:end_idx]
        front_ranges = [r for r in front_ranges
                        if scan_msg.range_min < r < scan_msg.range_max]
        min_front_distance = min(front_ranges) if front_ranges else float('inf')

        safety_distance = 0.3

        if min_front_distance < safety_distance:
            twist_msg.linear.x = 0.0
            mid_idx = len(front_ranges) // 2
            left_avg = sum(front_ranges[:mid_idx]) / max(1, mid_idx)
            right_avg = sum(front_ranges[mid_idx:]) / max(
                1, len(front_ranges) - mid_idx)
            twist_msg.angular.z = 0.5 if left_avg < right_avg else -0.5
            self.get_logger().info("Obstacle detected! Turning...")
            self.pub_cmd.publish(twist_msg)
            return

        # ----------- Normal navigation -----------
        at_position = (abs(dx) <= self.pos_tol_x) and (abs(dy) <= self.pos_tol_y)
        if not at_position:
            speed = min(self.k_lin * distance_error, 0.5)
            twist_msg.angular.z = max(min(self.k_ang * heading_error, 1.0), -1.0)

            if use_reverse:
                twist_msg.linear.x = -speed
            else:
                twist_msg.linear.x = speed
        else:
            orientation_error = self.normalize_angle(goal_theta - self.robot_theta)
            if abs(orientation_error) > self.theta_tol:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = max(min(self.k_theta * orientation_error, 1.0), -1.0)
                self.get_logger().info(
                    f"Aligning to final orientation | Error: {math.degrees(orientation_error):.2f}°"
                )
            else:
                # WAYPOINT REACHED - Check if we have a pending detection
                if self.pending_detection is not None and self.pending_detection['waypoint'] == self.current_wp:
                    # We have a detection for this waypoint - start 2 second stop
                    if not self.waypoint_stop_triggered:
                        self.get_logger().info(f"Waypoint {self.current_wp + 1} reached with detection - stopping for 2 seconds")
                        self.waypoint_stop_triggered = True
                        self.waypoint_stop_start_time = current_time
                        twist_msg.linear.x = 0.0
                        twist_msg.angular.z = 0.0
                else:
                    # No detection - move to next waypoint immediately
                    self.get_logger().info(
                        f"Waypoint {self.current_wp + 1} reached (no detection) | Distance error: {distance_error:.4f} m"
                    )
                    twist_msg.linear.x = 0.0
                    twist_msg.angular.z = 0.0
                    self.current_wp += 1
                    self.dock_announced = False
                    self.get_logger().info(f"Moving to waypoint {self.current_wp + 1}")

        self.pub_cmd.publish(twist_msg)

    # ===================== HELPER METHODS =====================
    def get_plant_id(self, x, y, local_y):
        """Determine plant_ID based on orientation + x,y region."""
        theta = self.robot_theta

        # LANE GROUP A (θ ≈ -1.57) - facing downward
        if -1.9 < theta < -1.2:
            if 0.0 < x < 1.0:
                if -4.7 <= y <= -3.5:
                    return 1
                elif -3.5 <= y <= -2.1:
                    return 2
                elif -2.1 <= y <= -0.6:
                    return 3
                elif -0.6 <= y <= 0.5:
                    return 4
            elif -2.0 < x < -0.6:
                if -4.7 <= y <= -3.5:
                    return 5 if local_y < 0 else 1
                elif -3.5 <= y <= -2.1:
                    return 6 if local_y < 0 else 2
                elif -2.1 <= y <= -0.6:
                    return 7 if local_y < 0 else 3
                elif -0.6 <= y <= 0.5:
                    return 8 if local_y < 0 else 4
            elif -4.0 <= x <= -2.5:
                if -4.7 <= y <= -3.5:
                    return 5
                elif -3.5 <= y <= -2.1:
                    return 6
                elif -2.4 <= y <= -0.6:
                    return 7
                elif -0.6 <= y <= 0.5:
                    return 8

        # LANE GROUP B (θ ≈ +1.57) - facing upward
        elif 1.2 < theta < 1.8:
            if 0.0 < x < 1.0:
                if -4.7 <= y <= -3.5:
                    return 1
                elif -3.5 <= y <= -2.1:
                    return 2
                elif -2.1 <= y <= -0.6:
                    return 3
                elif -0.6 <= y <= 0.5:
                    return 4
            elif -2.0 < x < -0.6:
                if -4.7 <= y <= -3.5:
                    return 5 if local_y > 0 else 1
                elif -3.5 <= y <= -2.1:
                    return 6 if local_y > 0 else 2
                elif -2.1 <= y <= -0.6:
                    return 7 if local_y > 0 else 3
                elif -0.6 <= y <= 0.5:
                    return 8 if local_y > 0 else 4
            elif -4.0 < x < -2.5:
                if -4.7 <= y <= -3.5:
                    return 5
                elif -3.5 <= y <= -2.1:
                    return 6
                elif -2.1 <= y <= -0.6:
                    return 7
                elif -0.6 <= y <= 0.5:
                    return 8

        return 0

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize any angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


# ===================== MAIN =====================
def main(args=None):
    """Entry point: Initialize ROS2, create node, and spin."""
    rclpy.init(args=args)
    node = EBotNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()