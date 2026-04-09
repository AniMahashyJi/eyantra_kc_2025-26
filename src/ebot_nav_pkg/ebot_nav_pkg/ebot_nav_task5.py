# ===================== IMPORTS =====================
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation as R
import math
def euler_from_quaternion(quaternion):
        """
        Convert quaternion (x, y, z, w) to euler angles (roll, pitch, yaw)
        """
        rot = R.from_quat(quaternion)
        euler = rot.as_euler('xyz', degrees=False)
        return euler[0], euler[1], euler[2]  # roll, pitch, yawimport math

# ===================== MAIN NODE CLASS =====================
class EBotNav(Node):
    """
    ROS2 Node that handles:
        - Waypoint navigation
        - Obstacle avoidance
        - Shape detection and response (published at waypoint arrival)
        - Orientation alignment
        - Optional: Waiting at dock for CAN_RELEASED from arm
    """

    def __init__(self):
        """Initialize subscriptions, publishers, state variables, and timers."""
        super().__init__('ebot_nav_task4b')
        self.align_direction = None

        # ========== SHAPE DETECTION STOP CONFIGURATION ==========
        # Configure which detection number should trigger a stop
        # Key: detection number (1st, 2nd, 3rd, etc.)
        # Value: number of waypoints to skip (0 = stop at next waypoint, 1 = skip 1 waypoint, etc.)
        self.detection_stop_config = {
            1: 1,  # 1st detection: stop at next waypoint (0 skips)
            2: 0,  # 2nd detection: stop at next waypoint
            3: 0,  # 3rd detection: skip 1 waypoint, stop at next-to-next
            4: 0,  # 4th detection: stop at next waypoint
            5: 0,  # 5th detection: skip 2 waypoints
            # Add more as needed...
        }
        
        # Shape detection tracking
        self.detection_count = 0  # Counter for total detections
        self.detection_history = []  # Track all detections with their info
        # ========================================================

        # ----------------- Arm flag state -----------------
        self.use_arm_coordination = False  # Set to True for Task 3B, False for Task 4B
        self.waiting_for_can_release = False
        self.can_released = False
        self.can_released_time = None
        self.post_release_delay = 2.0           # seconds
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
        self.pending_detection = None  # Will store: {'shape': str, 'local_x': float, 'local_y': float, 'waypoint': int, 'detection_num': int}
        self.waypoint_stop_triggered = False  # Flag for stopping at waypoint with detection
        self.waypoint_stop_start_time = None  # Time when stop started
        
        # ----------------- Dock Detection State (using ROS Time) --------
        self.dock_stop_triggered = False
        self.dock_stop_start_time = None
        self.dock_announced = False

        # ----------------- Waypoints -------------------
        self.waypoints = [
            [0.0, 0.0, -0.19],
            [0.0, 0.0, -0.88],
            [0.0, 0.0, -1.53],
            [0.0, -0.65, -1.61],
            [0.0, -1.27, -1.61],
            [0.0, -1.42, -1.2],
            [0.0, -1.42, -0.52],
            [0.0, -1.42, 0.01],
            [0.70, -1.64, 0.0],
            [1.71, -1.69, 0.0],      # 1: plant1
            [2.25, -1.69, 0.0],       # 2: dock
            [2.25, -1.69, 0.0],           # 3: plant2
            [3.06, -1.69, 0.0],     # 4: plant3
            [3.53, -1.69, 0.0],      # 5: plant4
            [4.43, -1.75, 0.0],
            [4.76, -1.67, 0.64],
            [4.76, -1.67, 1.32],
            [4.76, 1.4, 1.57],
            [4.76, 1.43, 1.85],
            [4.54, 1.7, 2.48],             
            [3.53, 1.7, 3.12],     # 8: plant8
            [3.06, 1.7, 3.12],    # 9: plant7
            [2.15, 1.7, 3.12],      # 10: plant6
            [1.71, 1.7, 3.12],     # 11: plant5
            [0.94, 1.66, -2.69],            
            [0.53, 1.21, -2.01],   
            [0.49, 0.53, -1.34], 
            [0.49, 0.0, 0.0],         
            [1.71, 0., 0.0],     # 14: plant1,5
            [2.25, 0.0, 0.0],      # 15: plant2,6
            [3.06, 0.0, 0.0],    # 16: plant3,7
            [3.53, 0.0, 0.0],     # 17: plant4,8
            [5.08, 0.0, 0.0],             # 18
            [0.0, 0.0, 0.0]            # home
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

        mode = "Task 4B (No Arm Coordination)" if not self.use_arm_coordination else "Task 3B (With Arm Coordination)"
        self.get_logger().info(f"EBot navigation node started - Mode: {mode}")
        self.get_logger().info(f"Detection stop config: {self.detection_stop_config}")

    # ===================== CALLBACKS =====================

    def arm_flag_cb(self, msg: String):
        """Handle CAN_RELEASED from arm."""
        if not self.use_arm_coordination:
            return  # Ignore arm flags if not using arm coordination
        
        if msg.data == "CAN_RELEASED":
            self.get_logger().info("Arm: CAN_RELEASED received, waiting 2 sec before resuming nav")
            self.can_released = True
            self.can_released_time = time.time()
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
        
        if self.use_arm_coordination and self.waiting_for_can_release:
            return

        if self.pending_detection is not None:
            return

        if self.current_wp >= len(self.waypoints):
            return

        try:
            shape, lx, ly = msg.data.split('|')
            lx = float(lx)
            ly = float(ly)

            # Increment detection counter
            self.detection_count += 1
            
            # Get skip count for this detection number
            skip_count = self.detection_stop_config.get(self.detection_count, 0)
            
            # Calculate target waypoint based on skip count
            if self.current_wp == 2:  # dock special case
                base_wp = self.current_wp + 1
            else:
                base_wp = self.current_wp
            
            # Add skip count to determine final waypoint
            assigned_wp = base_wp + skip_count
            assigned_wp = min(assigned_wp, len(self.waypoints) - 1)

            self.pending_detection = {
                'shape': shape,
                'local_x': lx,
                'local_y': ly,
                'waypoint': assigned_wp,
                'detection_num': self.detection_count
            }
            
            # Store in history
            self.detection_history.append({
                'num': self.detection_count,
                'shape': shape,
                'assigned_wp': assigned_wp,
                'skip_count': skip_count
            })

            self.get_logger().info(
                f"Detection #{self.detection_count}: {shape} → "
                f"Skip {skip_count} waypoint(s), assigned WP {assigned_wp}"
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
        detection_num = self.pending_detection['detection_num']
        
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
        self.get_logger().info(
            f"✅ Published detection #{detection_num} at waypoint {self.current_wp}: {shape_msg.data}"
        )
        
        # Clear pending detection
        self.pending_detection = None

    # ===================== CONTROL LOOP =====================
    def control_loop(self):
        """Main navigation + detection control loop."""
        # ---------- While arm is working at dock, keep robot stopped ----------
        if self.use_arm_coordination and self.waiting_for_can_release and not self.can_released:
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.pub_cmd.publish(stop)
            enable_msg = Bool()
            enable_msg.data = False
            self.pub_enable_detection.publish(enable_msg)
            return
            
        # ---------- Wait 2 seconds after CAN_RELEASED before moving ----------
        if self.use_arm_coordination and self.can_released and self.can_released_time is not None:
            elapsed = time.time() - self.can_released_time
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
        if self.current_wp in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15]:
            filter_msg = Bool()
            filter_msg.data = True
            self.pub_filter_negative.publish(filter_msg)
        elif self.current_wp in [16,17,18,19]:  # e.g., [14, 15, 16]
            # Disable detection completely at these waypoints
            enable_msg = Bool()
            enable_msg.data = False
            self.pub_enable_detection.publish(enable_msg)
            # Still publish filter status
            filter_msg = Bool()
            filter_msg.data = False
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
            elapsed = time.time() - self.dock_stop_start_time

            if elapsed >= 2.0:
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
            elapsed = time.time() - self.waypoint_stop_start_time

            if elapsed >= 2.0:
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

        # ---------- DOCK STOP (2 seconds) ----------
        if self.current_wp == 11 and not self.dock_announced:
            at_position = (abs(dx) <= self.pos_tol_x) and (abs(dy) <= self.pos_tol_y)

            if at_position and not self.dock_stop_triggered:
                orientation_error = self.normalize_angle(goal_theta - self.robot_theta)
                if abs(orientation_error) > self.theta_tol:
                    twist_align = Twist()
                    twist_align.linear.x = 0.0
                    max_ang = 1.0
                    twist_align.angular.z = max(min(self.k_theta * orientation_error, max_ang), -max_ang)
                    self.get_logger().info(
                        f"Aligning at dock before stop | Error: {math.degrees(orientation_error):.2f}°"
                    )
                    self.pub_cmd.publish(twist_align)
                    return

                # Publish dock detection
                global_x = self.robot_x
                global_y = self.robot_y
                status_msg = String()
                status_msg.data = f"DOCK_STATION,{global_x:.2f},{global_y:.2f},0"
                self.pub_detection_status.publish(status_msg)
                self.get_logger().info(f"✅ Dock reached & aligned → published: {status_msg.data}")

                # Start 2-second stop at dock
                if self.use_arm_coordination:
                    self.waiting_for_can_release = True
                    self.can_released = False
                    self.get_logger().info("Waiting for arm to release can...")

                self.dock_stop_triggered = True
                self.dock_stop_start_time = time.time()
                self.get_logger().info("Starting 2-second stop at dock...")
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
                        detection_num = self.pending_detection['detection_num']
                        self.get_logger().info(
                            f"Waypoint {self.current_wp + 1} reached with detection #{detection_num} - stopping for 2 seconds"
                        )
                        self.waypoint_stop_triggered = True
                        self.waypoint_stop_start_time = time.time()
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

        # LANE GROUP A (θ ≈ 3.14) - facing downward
        if 2.5 < theta < 3.14 or -3.14 < theta < -2.5:
            if -1.0 < y < -3.0:
                if 0 <= x <= 1.9:
                    return 1
                elif 1.9 < x <= 2.8:
                    return 2
                elif 2.8 < x <= 3.2:
                    return 3
                elif 3.2 < x <= 5.0:
                    return 4
            elif -1.0 < y < 1.0:
                if 0 <= x <= 1.9:
                    return 5 if local_y < 0 else 1
                elif 1.9 < x <= 2.8:
                    return 6 if local_y < 0 else 2
                elif 2.8 < x <= 3.2:
                    return 7 if local_y < 0 else 3
                elif 3.2 < x <= 5.0:
                    return 8 if local_y < 0 else 4
            elif 1.0 <= y <= 3.0:
                if 0.0 <= x <= 1.9:
                    return 5
                elif 1.9 < x <= 2.8:
                    return 6
                elif 2.8 < x <= 3.2:
                    return 7
                elif 3.2 < x <= 5.0:
                    return 8

        # LANE GROUP B (θ ≈ +0.0) - facing upward
        elif 0.0 <= theta < 1.0 or -1.0 < theta <= 0.0:
            if -1.0 < y < -3.0:
                if 0 <= x <= 1.9:
                    return 1
                elif 1.9 < x <= 2.8:
                    return 2
                elif 2.8 < x <= 3.2:
                    return 3
                elif 3.2 < x <= 5.0:
                    return 4
            elif -1.0 < y < 1.0:
                if 0 <= x <= 1.9:
                    return 5 if local_y > 0 else 1
                elif 1.9 < x <= 2.8:
                    return 6 if local_y > 0 else 2
                elif 2.8 < x <= 3.2:
                    return 7 if local_y > 0 else 3
                elif 3.2 < x <= 5.0:
                    return 8 if local_y > 0 else 4
            elif 1.0 <= y <= 3.0:
                if 0.0 <= x <= 1.9:
                    return 5
                elif 1.9 < x <= 2.8:
                    return 6
                elif 2.8 < x <= 3.2:
                    return 7
                elif 3.2 < x <= 5.0:
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