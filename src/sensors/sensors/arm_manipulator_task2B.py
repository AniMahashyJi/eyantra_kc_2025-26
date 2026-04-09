#!/usr/bin/env python3
'''
 Team ID:          2635
 Theme:            Krishi CoBot
 Author List:      Animesh,Swayam,Gautam,Harshit
 Filename:         task3b_manipulation.py
 Functions:        dock_cb, get_tf, get_eef_pose, is_valid_fruit_position,
                   find_next_fruit, calc_ee_orientation, servo_to_pos,
                   servo_pose, servo_pose_loose, attach, detach, stop_t,
                   wait, log_phase, upd
 Global variables: TEAM_ID, SUCTION_AXIS
'''

import rclpy, time
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from linkattacher_msgs.srv import AttachLink, DetachLink
from scipy.spatial.transform import Rotation as R
import numpy as np

TEAM_ID = "2635"
SUCTION_AXIS = 'Z'

class PickPlace(Node):
    """
    PURPOSE:
        - Controls manipulator arm and gripper
        - Moves arm to target positions using joint jogging and Cartesian servoing
        - Opens/closes gripper
        - Reads TFs from perception node
    """
    def __init__(self):
        
        super().__init__('pick_place')
        
        # Publishers
        self.joint_pub = self.create_publisher(
            Float64MultiArray, '/delta_joint_cmds', 10
        )
        self.arm_flag_pub = self.create_publisher(
            String, '/arm_flag', 10
        )  
        self.tp = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        # Subscribers
        self.dock_sub = self.create_subscription(
            String, '/detection_status', self.dock_cb, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10
        )
        
        # TF
        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)
        
        # Service clients
        self.attach_cli = self.create_client(AttachLink, '/attach_link')
        self.detach_cli = self.create_client(DetachLink, '/detach_link')
        
        # State
        self.dock_reached = True
        
        # Joint state tracking
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        self.current_joints = [0.0] * 6
        self.have_joint_state = False
        
        # Joint jogging waypoints for APPROACH (Phase 0)
        # Format: (time, [joint positions])
        self.approach_waypoints = [
            (1.0, [-3.140, -0.590, -2.490, -0.057, 1.570, 3.150]),
            (2.0, [-4.773, -1.413, -1.820, 0.140, 1.570, 3.198]),
            (3.0, [-4.773, -1.314, -1.408, -0.003, 1.570, 3.167]),
            (4.0, [-4.773, -1.371, -1.453, -0.015, 1.570, 3.169]),
            (5.0, [-4.773, -1.336, -1.521, -0.036, 1.570, 3.144]),
            (6.0, [-4.803, -1.304, -1.527, -0.051, 1.570, 3.142]),

        ]
        
        # Joint jogging waypoints for RETRACT (Phase 2)
        # Format: (time, [joint positions])
        self.retract_waypoints = [
            # TODO: Fill with your retract waypoints
            # Leave blank - you'll fill this
             (29.28, [-5.131,-1.436,-1.535,-0.405,2.403,2.487]),
            (30.35, [-5.177,-1.289,-1.255,-2.069,1.768,1.168]),
            (31.16, [-4.579,-1.101,-1.346,-2.221,1.648,1.223]),
            (32.01, [-3.806,-1.145,-1.326,-2.202,1.652,1.148]),
            (33.18, [-2.852,-1.101,-1.346,-2.221,1.648,1.223]),
            (34.03, [-2.983,-1.444,-1.473,-1.789,1.570,3.185]),
            (35.12, [-2.991,-1.897,-1.210,-1.605,1.571,3.278]),
        ]
        
        # Joint control parameters
        self.kp = 0.8
        self.kd = 0.05
        self.prev_error = [0.0] * 6
        self.max_step = [0.9, 0.9, 0.9, 0.5, 0.5, 0.5]
        self.wp_idx = 0
        
        # Cartesian motion parameters
        self.K_pos = 1.2
        self.K_rot = 1.8
        self.ml = 0.6
        self.ma = 1.5
        self.pt = 0.03
        self.rt = 0.10
        
        # Phase management
        self.ph = 0
        self.st = time.time()
        self.drop_timer = 0
        
        # Workspace
        self.picked_fruits = set()
        self.skipped_fruits = set()
        self.current_fruit_id = None
        self.max_fruits = 3
        self.fruit_positions = {}
        
        # Trash bin position
        self.p3 = np.array([-0.806, 0.010, 0.182]) 
        
        self.lift_offset = -0.1
        
        self.phase_logged = -1
        
        self.timer = self.create_timer(0.02, self.upd)
        
        self.get_logger().info("Pick and Place Started - Waiting for dock...")

    def dock_cb(self, msg: String):
        if msg.data.startswith("DOCK_STATION") and not self.dock_reached:
            self.get_logger().info("Dock reached → starting arm sequence (ONLY ONCE)")
            self.dock_reached = True
            self.ph = 0
            self.st = time.time()
            self.wp_idx = 0

    def joint_state_cb(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                self.current_joints[i] = msg.position[msg.name.index(name)]
        self.have_joint_state = True

    # ===================== TF & POSE UTILITIES =====================        

    def get_tf(self, target_frame):
        try:
            t = self.tf_buf.lookup_transform('base_link', target_frame, 
                                             rclpy.time.Time())
            pos = np.array([t.transform.translation.x, 
                            t.transform.translation.y, 
                            t.transform.translation.z])
            quat = np.array([t.transform.rotation.x,
                             t.transform.rotation.y,
                             t.transform.rotation.z,
                             t.transform.rotation.w])
            return pos, quat
        except:
            return None, None

    def get_eef_pose(self):
        try:
            t = self.tf_buf.lookup_transform('base_link', 'tool0', 
                                             rclpy.time.Time())
            pos = np.array([t.transform.translation.x, 
                            t.transform.translation.y, 
                            t.transform.translation.z])
            quat = np.array([t.transform.rotation.x,
                             t.transform.rotation.y,
                             t.transform.rotation.z,
                             t.transform.rotation.w])
            return pos, quat
        except:
            return None, None

    def is_valid_fruit_position(self, pos):
        false_pos = np.array([0.558, -0.005, -0.021])
        tolerance = 0.1
        
        distance = np.linalg.norm(pos - false_pos)
        if distance < tolerance:
            return False
        return True

    def find_next_fruit(self):
        for fruit_id in range(1, self.max_fruits + 1):
            if fruit_id not in self.picked_fruits and fruit_id not in self.skipped_fruits:
                if fruit_id in self.fruit_positions:
                    pos, quat = self.fruit_positions[fruit_id]
                    return fruit_id, pos, quat
        return None, None, None

    def calc_ee_orientation(self, z_axis, for_pick=True):
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        if for_pick:
            target_dir = z_axis
        else:
            target_dir = -z_axis
        
        if SUCTION_AXIS == 'X':
            ee_x = target_dir
            if abs(ee_x[2]) < 0.9:
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([0, 1, 0])
            ee_y = np.cross(ref, ee_x)
            if np.linalg.norm(ee_y) < 0.01:
                ref = np.array([1, 0, 0])
                ee_y = np.cross(ref, ee_x)
            ee_y = ee_y / np.linalg.norm(ee_y)
            ee_z = np.cross(ee_x, ee_y)
            
        elif SUCTION_AXIS == 'Y':
            ee_y = target_dir
            if abs(ee_y[2]) < 0.9:
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([1, 0, 0])
            ee_x = np.cross(ee_y, ref)
            if np.linalg.norm(ee_x) < 0.01:
                ref = np.array([0, 1, 0])
                ee_x = np.cross(ee_y, ref)
            ee_x = ee_x / np.linalg.norm(ee_x)
            ee_z = np.cross(ee_x, ee_y)
            
        elif SUCTION_AXIS == 'Z':
            ee_z = target_dir
            if abs(ee_z[2]) < 0.9:
                ref = np.array([0, 0, 1])
            else:
                ref = np.array([1, 0, 0])
            ee_x = np.cross(ref, ee_z)
            if np.linalg.norm(ee_x) < 0.01:
                ref = np.array([0, 1, 0])
                ee_x = np.cross(ref, ee_z)
            ee_x = ee_x / np.linalg.norm(ee_x)
            ee_y = np.cross(ee_z, ee_x)
        
        else:
            raise ValueError(f"Invalid SUCTION_AXIS: {SUCTION_AXIS}")
        
        ee_x = ee_x / np.linalg.norm(ee_x)
        ee_y = ee_y / np.linalg.norm(ee_y)
        ee_z = ee_z / np.linalg.norm(ee_z)
        
        rot_mat = np.column_stack([ee_x, ee_y, ee_z])
        quat = R.from_matrix(rot_mat).as_quat()
        return quat
    
    # ===================== JOINT JOGGING =====================

    def joint_jog_to_waypoint(self, waypoints):
        """
        Joint jogging using PID control to follow waypoints.
        Returns True when all waypoints completed.
        """
        if not self.have_joint_state:
            return False
        
        if self.wp_idx >= len(waypoints):
            self.stop_joints()
            return True
        
        _, target_positions = waypoints[self.wp_idx]
        
        # Compute delta commands with PID
        deltas = []
        for i in range(6):
            error = target_positions[i] - self.current_joints[i]
            derr = error - self.prev_error[i]
            step = self.kp * error + self.kd * derr
            step = max(min(step, self.max_step[i]), -self.max_step[i])
            deltas.append(step)
            self.prev_error[i] = error
        
        # Publish joint deltas
        msg = Float64MultiArray()
        msg.data = deltas
        self.joint_pub.publish(msg)
        
        # Check if reached current waypoint
        if all(abs(c - t) < 0.02 for c, t in zip(self.current_joints, target_positions)):
            self.wp_idx += 1
        
        return False

    def stop_joints(self):
        """Stop joint motion"""
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.joint_pub.publish(msg)

    # ===================== CARTESIAN SERVOING =====================

    def servo_to_pos(self, target):
        eef_pos, _ = self.get_eef_pose()
        if eef_pos is None:
            return False
        
        target = np.array(target)
        err = target - eef_pos
        d = np.linalg.norm(err)
        
        if d < self.pt:
            self.stop_t()
            return True
        
        vel = np.clip(self.K_pos * err, -self.ml, self.ml)
        
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = float(vel[0]), float(vel[1]), float(vel[2])
        self.tp.publish(tw)
        
        return False

    def servo_pose(self, target_pos, target_quat):
        eef_pos, eef_quat = self.get_eef_pose()
        if eef_pos is None:
            return False
        
        pos_err = target_pos - eef_pos
        pos_dist = np.linalg.norm(pos_err)
        
        eef_rot = R.from_quat(eef_quat)
        target_rot = R.from_quat(target_quat)
        rot_err = target_rot * eef_rot.inv()
        rot_vec = rot_err.as_rotvec()
        rot_dist = np.linalg.norm(rot_vec)
        
        if pos_dist < self.pt and rot_dist < self.rt:
            self.stop_t()
            return True
        
        lin_vel = np.clip(self.K_pos * pos_err, -self.ml, self.ml)
        ang_vel = np.clip(self.K_rot * rot_vec, -self.ma, self.ma)
        
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])
        tw.angular.x, tw.angular.y, tw.angular.z = float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])
        self.tp.publish(tw)
        
        return False

    def servo_pose_loose(self, target_pos, target_quat, timeout=2.0):
        eef_pos, eef_quat = self.get_eef_pose()
        if eef_pos is None:
            return False
        
        pos_err = target_pos - eef_pos
        pos_dist = np.linalg.norm(pos_err)
        
        eef_rot = R.from_quat(eef_quat)
        target_rot = R.from_quat(target_quat)
        rot_err = target_rot * eef_rot.inv()
        rot_vec = rot_err.as_rotvec()
        rot_dist = np.linalg.norm(rot_vec)
        
        elapsed = time.time() - self.drop_timer
        if (pos_dist < self.pt and rot_dist < self.rt) or elapsed > timeout:
            self.stop_t()
            return True
        
        lin_vel = np.clip(self.K_pos * pos_err, -self.ml, self.ml)
        ang_vel = np.clip(self.K_rot * rot_vec, -self.ma, self.ma)
        
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])
        tw.angular.x, tw.angular.y, tw.angular.z = float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])
        self.tp.publish(tw)
        
        return False

    # ===================== ATTACH/DETACH =====================

    def attach(self, obj_name):
        if not self.attach_cli.wait_for_service(timeout_sec=0.5):
            return False
        
        req = AttachLink.Request()
        req.model1_name = obj_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        
        future = self.attach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if future.result() is None:
            return False

        res = future.result()
        if hasattr(res, "success") and res.success:
            return True
        else:
            return False

    def detach(self, obj_name):
        if not self.detach_cli.wait_for_service(timeout_sec=0.5):
            return False
        
        req = DetachLink.Request()
        req.model1_name = obj_name
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        
        future = self.detach_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        if future.result() is None:
            return False

        res = future.result()
        if hasattr(res, "success") and res.success:
            return True
        else:
            return False

    def stop_t(self):
        self.tp.publish(Twist())

    def wait(self, s):
        return (time.time() - self.st) > s

    def log_phase(self, message):
        if self.ph != self.phase_logged:
            self.get_logger().info(message)
            self.phase_logged = self.ph

    # ===================== MAIN UPDATE LOOP =====================

    def upd(self):
        if not self.dock_reached:
            return
        
        p = self.ph
        
        # ========== PHASE 0: Joint Jogging to Approach ==========
        if p == 0:
            self.log_phase("Phase 0: Joint jogging to approach fertilizer can")
            
            if len(self.approach_waypoints) == 0:
                self.get_logger().warn("No approach waypoints defined! Skipping to Phase 1")
                self.ph = 1
                self.st = time.time()
                self.wp_idx = 0
                return
            
            if self.joint_jog_to_waypoint(self.approach_waypoints):
                self.get_logger().info("✅ Approach complete")
                self.ph = 0.2
                self.st = time.time()
                self.wp_idx = 0  # Reset for next jogging phase

        elif p == 0.2:
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')

            
            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                offset = 0.15
                target_pos = fert_pos - offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=True)
                self.get_logger().info(
                f"  Position  -> x={target_pos[0]:.3f}, "
                f"y={target_pos[1]:.3f}, z={target_pos[2]:.3f}\n"
                f"  Quaternion-> x={target_quat[0]:.3f}, "
                f"y={target_quat[1]:.3f}, z={target_quat[2]:.3f}, "
                f"w={target_quat[3]:.3f}"
                )
                self.log_phase("Phase 1: Rotating for fertilizer")
                if self.servo_pose(target_pos, target_quat):
                    self.fert_pos = fert_pos
                    self.fert_z = z_axis
                    self.ph = 0.3
                    self.st = time.time()
        
        elif p == 0.3:
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')

            
            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                offset = 0.05
                target_pos = fert_pos - offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=True)
                self.get_logger().info(
                f"  Position  -> x={target_pos[0]:.3f}, "
                f"y={target_pos[1]:.3f}, z={target_pos[2]:.3f}\n"
                f"  Quaternion-> x={target_quat[0]:.3f}, "
                f"y={target_quat[1]:.3f}, z={target_quat[2]:.3f}, "
                f"w={target_quat[3]:.3f}"
                )
                self.log_phase("Phase 2: Grasping fertilizer")
                
                if self.servo_pose(target_pos, target_quat):
                    self.fert_pos = fert_pos
                    self.fert_z = z_axis
                    self.ph = 1
                    self.st = time.time()
        
        
        # ========== PHASE 1: Attach Can ==========
        elif p == 1:
            self.log_phase("Phase 1: Attaching fertilizer can")
            if self.wait(0.3):
                self.attach('fertiliser_can')
                self.get_logger().info("✅ Fertilizer can ATTACHED")
                self.ph = 2
                self.st = time.time()
                self.wp_idx = 0  # Reset for retract waypoints
        
        # ========== PHASE 2: Joint Jogging to Retract ==========
        elif p == 2:
            self.log_phase("Phase 2: Joint jogging to retract with can")
            
            if len(self.retract_waypoints) == 0:
                self.get_logger().warn("No retract waypoints defined! Skipping to Phase 3")
                self.ph = 3
                self.st = time.time()
                return
            
            if self.joint_jog_to_waypoint(self.retract_waypoints):
                self.get_logger().info("✅ Retract complete")
                self.ph = 3
                self.st = time.time()
        
        # ========== PHASE 3: Detach Can ==========
        elif p == 3:
            self.log_phase("Phase 3: Detaching fertilizer can")
            if self.wait(0.3):
                self.detach('fertiliser_can')
                msg = String()
                msg.data = "CAN_RELEASED"
                self.arm_flag_pub.publish(msg)
                self.get_logger().info("✅ Fertilizer can DETACHED - Published CAN_RELEASED on /arm_flag")
                self.ph = 4
                self.st = time.time()
        
        # ========== PHASE 4: Scan Fruits ==========
        elif p == 4:
            self.log_phase("Phase 4: Scanning fruits")
            if self.wait(0.5):
                detected_fruits = []
                
                for fruit_id in range(1, self.max_fruits + 1):
                    frame_name = f"{TEAM_ID}_bad_fruit_{fruit_id}"
                    pos, quat = self.get_tf(frame_name)
                    
                    if pos is not None:
                        if self.is_valid_fruit_position(pos):
                            detected_fruits.append((fruit_id, pos))
                            self.fruit_positions[fruit_id] = (pos, quat)
                        else:
                            self.skipped_fruits.add(fruit_id)
                
                self.get_logger().info(f"Detected {len(detected_fruits)} valid fruits")
                self.ph = 5
                self.st = time.time()
        
        # ========== PHASE 5: Find Next Fruit ==========
        elif p == 5:
            if self.wait(0.3):
                fruit_id, fruit_pos, fruit_quat = self.find_next_fruit()
                
                if fruit_id is not None:
                    self.current_fruit_id = fruit_id
                    self.fruit_pos = fruit_pos
                    
                    rot = R.from_quat(fruit_quat)
                    self.fruit_z = rot.as_matrix()[:, 2]
                    
                    self.log_phase(f"Phase 5: Found fruit {fruit_id}")
                    
                    self.ph = 6
                    self.st = time.time()
                else:
                    self.get_logger().info("All fruits processed")
                    self.ph = 99
        
        # ========== PHASE 6: Move Above Fruit ==========
        elif p == 6:
            current_pos, current_quat = self.get_eef_pose()
            if current_quat is not None:
                rot = R.from_quat(current_quat)
                ee_z_axis = rot.as_matrix()[:, 2]
                
                target_pos = self.fruit_pos + self.lift_offset * ee_z_axis
                
                self.log_phase(f"Phase 6: Moving above fruit {self.current_fruit_id}")
                
                if self.servo_to_pos(target_pos):
                    self.ph = 7
                    self.st = time.time()
        
        # ========== PHASE 7: Approach Fruit ==========
        elif p == 7:
            if not hasattr(self, 'phase7_target'):
                current_pos, _ = self.get_eef_pose()
                if current_pos is not None:
                    approach_offset = 0.08
                    self.phase7_target = np.array([current_pos[0], current_pos[1], current_pos[2] - approach_offset])
            
            if hasattr(self, 'phase7_target'):
                self.log_phase(f"Phase 7: Approaching fruit {self.current_fruit_id}")
                if self.servo_to_pos(self.phase7_target):
                    delattr(self, 'phase7_target')
                    self.ph = 8
                    self.st = time.time()
        
        # ========== PHASE 8: Grasp Fruit ==========
        elif p == 8:
            current_pos, _ = self.get_eef_pose()
            if current_pos is not None:
                grasp_offset = 0.03
                target_pos = np.array([current_pos[0], current_pos[1], current_pos[2] - grasp_offset])
                
                self.log_phase(f"Phase 8: Grasping fruit {self.current_fruit_id}")
                
                if self.servo_to_pos(target_pos):
                    self.ph = 9
                    self.st = time.time()
        
        # ========== PHASE 9: Attach Fruit ==========
        elif p == 9:
            self.log_phase(f"Phase 9: Attaching fruit {self.current_fruit_id}")
            if self.wait(0.3):
                self.attach('bad_fruit')
                self.picked_fruits.add(self.current_fruit_id)
                self.ph = 10
                self.st = time.time()
        
        # ========== PHASE 10: Lift Fruit ==========
        elif p == 10:
            if not hasattr(self, 'phase10_target'):
                current_pos, current_quat = self.get_eef_pose()
                if current_quat is not None:
                    rot = R.from_quat(current_quat)
                    ee_z_axis = rot.as_matrix()[:, 2]
                    
                    self.phase10_target = current_pos - 0.17 * ee_z_axis
            
            if hasattr(self, 'phase10_target'):
                self.log_phase(f"Phase 10: Lifting fruit {self.current_fruit_id}")
                if self.servo_to_pos(self.phase10_target):
                    delattr(self, 'phase10_target')
                    self.ph = 11
                    self.st = time.time()
        
        # ========== PHASE 11: Move to Trash Bin ==========
        elif p == 11:
            target_pos = self.p3
            
            self.log_phase(f"Phase 11: Moving to trash bin with fruit {self.current_fruit_id}")
            
            if self.servo_to_pos(target_pos):
                self.ph = 12
                self.st = time.time()
        
        # ========== PHASE 12: Drop Fruit ==========
        elif p == 12:
            self.log_phase(f"Phase 12: Dropping fruit {self.current_fruit_id}")
            if self.wait(0.3):
                self.detach('bad_fruit')
                
                total_actual_fruits = len(self.fruit_positions)
                if len(self.picked_fruits) < total_actual_fruits:
                    self.ph = 5  # Back to find next fruit
                    self.st = time.time()
                else:
                    self.get_logger().info("All fruits processed")
                    self.ph = 99
        
        # ========== PHASE 99: Complete ==========
        elif p == 99:
            self.stop_t()
            self.stop_joints()
            self.log_phase("Phase 99: All tasks complete, holding position")


def main(a=None):
    rclpy.init(args=a)
    n = PickPlace()
    
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()