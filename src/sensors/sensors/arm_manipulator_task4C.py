#!/usr/bin/env python3
'''
 Team ID:          2635
 Theme:            Krishi CoBot
 Author List:      Animesh,Swayam,Gautam,Harshit
 Filename:         task3b_manipulation.py (IMPROVED VERSION)
 Functions:        dock_cb, get_tf, get_eef_pose, is_valid_fruit_position,
                   find_next_fruit, calc_ee_orientation, servo_to_pos,
                   servo_pose, servo_pose_loose, attach, detach, stop_t,
                   wait, log_phase, upd, check_attachment_success
 Global variables: TEAM_ID, SUCTION_AXIS
 
 IMPROVEMENTS:
 - Continuous approach with decreasing offset until contact
 - Attachment verification loop with service response checking
 - Retry mechanism for failed attachments
 - Applies to both fertilizer can and fruits
'''

import rclpy, time
from rclpy.node import Node
import math
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from linkattacher_msgs.srv import AttachLink, DetachLink
from scipy.spatial.transform import Rotation as R
import numpy as np

TEAM_ID = "2635"
SUCTION_AXIS = 'Z'

class PickPlace(Node):
    """
    PURPOSE:
        - Controls manipulator arm and gripper with improved attachment logic
        - Continuous approach with offset decrement
        - Attachment verification and retry mechanism
    """
    def __init__(self):
        # Phase 100: approach pose
        self.PHASE_100_POS = np.array([0.698, -0.004, 0.143])
        self.PHASE_100_QUAT = np.array([0.707, 0.707, 0.0, 0.0])

        # Phase 101: retreat / next pose
        self.PHASE_101_POS = np.array([0.698, -0.004, 0.143])
        self.PHASE_101_QUAT = np.array([0.707, 0.707, 0.0, 0.0])

        super().__init__('pick_place')
        
        # Second time ebot reaching
        self.second_dock_reached = False
        
        # Safe pose, quat
        self.safe_pos = np.array([-0.214, -0.474, 0.627])
        self.safe_quat = np.array([0.662, 0.749, -0.000, 0.000])
        
        # Home position
        self.home_pos = np.array([0.120, -0.109, 0.445])
        self.home_quat = np.array([0.501, 0.497, 0.503, 0.499])
        
        # Bin 
        self.bin_quat = np.array([0.662, 0.749, -0.000, 0.000])
        
        # State and timer
        self.dock_reached = False

        # Attachment control parameters (NEW)
        self.attach_offset = 0.15  # Starting offset (meters)
        self.attach_offset_step = 0.01  # Decrement step (10mm per iteration)
        self.attach_min_offset = 0.0  # Minimum offset (touch contact)
        self.attach_max_attempts = 20  # Max iterations before giving up
        self.attach_attempt = 0
        self.attachment_confirmed = False
        self.attach_retry_count = 0
        
        # Current object being attached
        self.current_attach_object = None

        # Around line 60-65, add a new publisher
        self.arm_flag_pub = self.create_publisher(
            String,
            '/arm_flag',
            10
        )
        self.dock_sub = self.create_subscription(
            String,
            '/detection_status',
            self.dock_cb,
            10
        )
        self.tp = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        
        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)
        
        self.attach_cli = self.create_client(AttachLink, '/attach_link')
        self.detach_cli = self.create_client(DetachLink, '/detach_link')
        
        # Motion parameters
        self.K_pos = 1.2
        self.K_rot = 1.8
        self.ml = 0.6
        self.ma = 1.5
        self.pt = 0.03
        self.rt = 0.10
        
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
        
        self.last_pose_log_time = 0.0
        self.pose_log_interval = 999.0
        
        self.timer = self.create_timer(0.02, self.upd)
        
        self.get_logger().info("Pick and Place Started (IMPROVED)")

    def dock_cb(self, msg: String):
        if not msg.data.startswith("DOCK_STATION"):
            return

        # First docking → normal sequence
        if not self.dock_reached:
            self.get_logger().info("Dock reached → starting arm sequence (FIRST TIME)")
            self.dock_reached = True
            self.ph = 0
            self.st = time.time()

        # Second docking → fertilizer disposal
        elif (
            self.dock_reached and
            self.ph == 99 and
            not self.second_dock_reached
        ):
            self.get_logger().info("Dock reached AGAIN → starting fertilizer disposal")
            self.second_dock_reached = True
            self.ph = 100
            self.st = time.time()

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
    
    def check_attachment_success(self, obj_name):
        """
        Check if attachment was successful by verifying the service response.
        For simulation, we rely on the service response.
        Returns True if object appears to be attached.
        """
        # In simulation, we trust the attach service response
        # This is called after attach() returns True
        self.get_logger().info(
            f"Attachment verification - {obj_name} attached via service response"
        )
        return True
    
    # ===================== MANIPULATION HELPERS =====================

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
            self.get_logger().info(f"✅ {obj_name} attached successfully")
            return True
        else:
            self.get_logger().warn(f"⚠️ {obj_name} attach service returned failure")
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
            self.get_logger().info(f"✅ {obj_name} detached successfully")
            return True
        else:
            self.get_logger().warn(f"⚠️ {obj_name} detach service returned failure")
            return False

    def stop_t(self):
        self.tp.publish(Twist())

    def wait(self, s):
        return (time.time() - self.st) > s

    def log_phase(self, message):
        if self.ph != self.phase_logged:
            self.get_logger().info(message)
            self.phase_logged = self.ph

    def upd(self):
        if not self.dock_reached:
            return
        p = self.ph
        
        # ========== PHASE 0: Initialize Fertilizer Approach ==========
        if p == 0:
            self.log_phase("Phase 0: Initializing fertilizer approach")
            
            # Reset attachment parameters
            self.attach_offset = 0.15
            self.attach_attempt = 0
            self.attachment_confirmed = False
            self.current_attach_object = 'fertiliser_can'
            
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')

            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                # Store fertilizer info
                self.fert_pos = fert_pos
                self.fert_z = z_axis
                
                target_pos = fert_pos - self.attach_offset * z_axis
                
                self.get_logger().info(
                    f"Starting approach - Offset: {self.attach_offset:.3f}m\n"
                    f"  Target pos: x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}"
                )
                
                if self.servo_to_pos(target_pos):
                    self.ph = 1
                    self.st = time.time()
            else:
                time.sleep(0.05)
        
        # ========== PHASE 1: Align Orientation ==========
        elif p == 1:
            self.log_phase("Phase 1: Aligning orientation for fertilizer")
            
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')
            
            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                self.fert_pos = fert_pos
                self.fert_z = z_axis
                
                target_pos = fert_pos - self.attach_offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=True)
                
                if self.servo_pose(target_pos, target_quat):
                    self.ph = 2
                    self.st = time.time()
        
        # ========== PHASE 2: Continuous Approach with Decreasing Offset ==========
        elif p == 2:
            if self.attach_attempt % 5 == 0:
                self.get_logger().info(
                    f"Phase 2: Approaching fertilizer - "
                    f"Offset: {self.attach_offset:.3f}m, "
                    f"Attempt: {self.attach_attempt + 1}/{self.attach_max_attempts}"
                )
            
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')
            
            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                self.fert_pos = fert_pos
                self.fert_z = z_axis
                
                # Calculate target with current offset
                target_pos = fert_pos - self.attach_offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=True)
                
                # Servo to current target
                if self.servo_pose(target_pos, target_quat):
                    # Reached current position
                    self.attach_attempt += 1
                    
                    if self.attach_offset <= self.attach_min_offset:
                        # Reached minimum offset
                        self.get_logger().info(
                            f"✓ Reached minimum offset ({self.attach_min_offset:.3f}m) - "
                            f"Proceeding to attachment"
                        )
                        self.ph = 3
                        self.st = time.time()
                    
                    elif self.attach_attempt >= self.attach_max_attempts:
                        # Max attempts reached
                        self.get_logger().warn(
                            f"⚠️  Max attempts ({self.attach_max_attempts}) reached - "
                            f"Proceeding to attachment anyway"
                        )
                        self.ph = 3
                        self.st = time.time()
                    
                    else:
                        # Decrease offset and continue
                        self.attach_offset = max(
                            self.attach_min_offset,
                            self.attach_offset - self.attach_offset_step
                        )
                        
                        self.get_logger().info(
                            f"➡ Decreasing offset to {self.attach_offset:.3f}m"
                        )
                        
                        self.st = time.time()
                        self.ph = 2.1  # Brief pause
            else:
                self.get_logger().warn("Fertilizer TF lost, retrying...")
        
        # ========== PHASE 2.1: Brief Pause ==========
        elif p == 2.1:
            if self.wait(0.1):
                self.ph = 2
                self.st = time.time()
        
        # ========== PHASE 3: Attach with Verification ==========
        elif p == 3:
            self.log_phase("Phase 3: Attempting to attach fertilizer")
            
            if not self.attachment_confirmed:
                if self.wait(0.3):
                    success = self.attach('fertiliser_can')
                    
                    if success:
                        # Service returned success
                        self.attachment_confirmed = True
                        self.get_logger().info("✅ Fertilizer can SECURELY ATTACHED")
                        
                        # Reset for next object
                        self.attach_offset = 0.15
                        self.attach_attempt = 0
                        
                        self.ph = 4
                        self.st = time.time()
                    else:
                        # Attachment failed
                        self.attach_retry_count += 1
                        
                        if self.attach_retry_count < 3:
                            self.get_logger().info(
                                f"Retrying attachment ({self.attach_retry_count}/3)"
                            )
                            time.sleep(0.3)
                            self.st = time.time()
                        else:
                            # Give up
                            self.get_logger().warn("Proceeding despite attachment failure")
                            self.attachment_confirmed = True
                            self.attach_retry_count = 0
                            self.ph = 4
                            self.st = time.time()
            else:
                self.ph = 4
                self.st = time.time()
        
        # ========== PHASE 4: Retract and Rotate ==========
        elif p == 4:
            ebot_pos, ebot_quat = self.get_tf(f'{TEAM_ID}_ebot_aruco')
            
            if ebot_pos is not None:
                rot_ebot = R.from_quat(ebot_quat)
                ebot_z = rot_ebot.as_matrix()[:, 2]
                
                offset = 0.35
                target_pos = self.fert_pos - offset * self.fert_z
                
                target_quat = self.calc_ee_orientation(ebot_z, for_pick=False)
                
                self.log_phase("Phase 4: Retracting and rotating")
                
                if self.servo_pose(target_pos, target_quat):
                    self.ebot_pos = ebot_pos
                    self.ebot_z = ebot_z
                    self.ph = 4.8
                    self.st = time.time()
            else:
                time.sleep(0.05)

        elif p == 4.8:
            # Virtual base rotation phase
            if not hasattr(self, 'rotation_target_computed'):
                p_cur, q_cur = self.get_eef_pose()
                if p_cur is None:
                    return
                
                R_cur = R.from_quat(q_cur)
                desired_rotation = np.pi / 2
                Rz = R.from_rotvec([0.0, 0.0, desired_rotation])
                
                self.rotation_p_des = Rz.apply(p_cur)
                self.rotation_R_des = Rz * R_cur
                
                self.rotation_target_computed = True
                self.log_phase("Phase 4.8: Starting virtual base rotation")
            
            p_cur, q_cur = self.get_eef_pose()
            if p_cur is None:
                return
            
            R_cur = R.from_quat(q_cur)
            
            pos_err = self.rotation_p_des - p_cur
            pos_dist = np.linalg.norm(pos_err)
            
            rot_err = self.rotation_R_des * R_cur.inv()
            rot_vec = rot_err.as_rotvec()
            rot_dist = np.linalg.norm(rot_vec)
            
            if pos_dist < 0.1 and rot_dist < 0.1:
                self.stop_t()
                delattr(self, 'rotation_target_computed')
                self.get_logger().info("✅ Virtual base rotation completed")
                self.ph = 5
                self.st = time.time()
                return
            
            lin_vel = np.clip(self.K_pos * pos_err, -self.ml, self.ml)
            ang_vel = np.clip(self.K_rot * rot_vec, -self.ma, self.ma)
            
            tw = Twist()
            tw.linear.x = float(lin_vel[0])
            tw.linear.y = float(lin_vel[1])
            tw.linear.z = float(lin_vel[2])
            tw.angular.x = float(ang_vel[0])
            tw.angular.y = float(ang_vel[1])
            tw.angular.z = float(ang_vel[2])
            self.tp.publish(tw)

        elif p == 5:
            ebot_pos, ebot_quat = self.get_tf(f'{TEAM_ID}_ebot_aruco')
            
            if ebot_pos is not None:
                rot = R.from_quat(ebot_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                offset = 0.25
                target_pos = ebot_pos + offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=False)
                
                self.log_phase("Phase 5: Moving to eBot approach")
                
                if self.servo_pose(target_pos, target_quat):
                    self.ebot_pos = ebot_pos
                    self.ebot_z = z_axis
                    self.ph = 6
                    self.st = time.time()
                    self.drop_timer = time.time()
        
        elif p == 6:
            ebot_pos, ebot_quat = self.get_tf(f'{TEAM_ID}_ebot_aruco')
            
            if ebot_pos is not None:
                rot = R.from_quat(ebot_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                offset = 0.25
                target_pos = ebot_pos + offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=False)
                
                self.log_phase("Phase 6: Moving to drop position")
                
                if self.servo_pose_loose(target_pos, target_quat, timeout=2.0):
                    self.ebot_pos = ebot_pos
                    self.ebot_z = z_axis
                    self.ph = 7
                    self.st = time.time()
        
        elif p == 7:
            self.log_phase("Phase 7: Detaching fertilizer")
            if self.wait(0.3):
                self.detach('fertiliser_can')
                msg = String()
                msg.data = "CAN_RELEASED"
                self.arm_flag_pub.publish(msg)
                self.get_logger().info("Published CAN_RELEASED on /arm_flag")

                self.ph = 8
                self.st = time.time()
        
        elif p == 8:
            offset = 0.25
            target_pos = self.ebot_pos + offset * self.ebot_z
            target_quat = self.calc_ee_orientation(self.ebot_z, for_pick=False)
            
            self.log_phase("Phase 8: Moving away from eBot")
            
            if self.servo_pose(target_pos, target_quat):
                self.ph = 9
                self.st = time.time()
        
        elif p == 9:
            self.log_phase("Phase 9: Scanning fruits")
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
                
                self.ph = 10
                self.st = time.time()
        
        elif p == 10:
            if self.wait(0.3):
                fruit_id, fruit_pos, fruit_quat = self.find_next_fruit()
                
                if fruit_id is not None:
                    self.current_fruit_id = fruit_id
                    self.fruit_pos = fruit_pos
                    
                    rot = R.from_quat(fruit_quat)
                    self.fruit_z = rot.as_matrix()[:, 2]
                    
                    # Reset attachment params for fruit
                    self.attach_offset = 0.10  # Smaller for fruits
                    self.attach_attempt = 0
                    self.attachment_confirmed = False
                    self.current_attach_object = 'bad_fruit'
                    
                    self.log_phase(f"Phase 10: Found fruit {fruit_id}")
                    
                    self.ph = 11
                    self.st = time.time()
                else:
                    self.get_logger().info("All tasks complete")
                    self.ph = 99
        
        elif p == 11:
            current_pos, current_quat = self.get_eef_pose()
            if current_quat is not None:
                rot = R.from_quat(current_quat)
                ee_z_axis = rot.as_matrix()[:, 2]
                
                target_pos = self.fruit_pos + self.lift_offset * ee_z_axis
                
                self.log_phase(f"Phase 11: Moving above fruit {self.current_fruit_id}")
                
                if self.servo_to_pos(target_pos):
                    self.ph = 12
                    self.st = time.time()
        
        # ========== PHASE 12: Continuous Approach to Fruit ==========
        elif p == 12:
            if self.attach_attempt % 5 == 0:
                self.get_logger().info(
                    f"Phase 12: Approaching fruit {self.current_fruit_id} - "
                    f"Offset: {self.attach_offset:.3f}m"
                )
            
            current_pos, current_quat = self.get_eef_pose()
            if current_quat is not None:
                rot = R.from_quat(current_quat)
                ee_z_axis = rot.as_matrix()[:, 2]
                
                # Target with decreasing offset
                target_pos = self.fruit_pos + (self.lift_offset + self.attach_offset) * ee_z_axis
                
                if self.servo_to_pos(target_pos):
                    self.attach_attempt += 1
                    
                    if self.attach_offset <= 0.0:
                        self.ph = 13
                        self.st = time.time()
                    elif self.attach_attempt >= 15:
                        self.ph = 13
                        self.st = time.time()
                    else:
                        self.attach_offset = max(0.0, self.attach_offset - 0.01)
                        self.ph = 12.1
                        self.st = time.time()
        
        elif p == 12.1:
            if self.wait(0.1):
                self.ph = 12
                self.st = time.time()
        
        # ========== PHASE 13: Attach Fruit with Verification ==========
        elif p == 13:
            self.log_phase(f"Phase 13: Attaching fruit {self.current_fruit_id}")
            
            if not self.attachment_confirmed:
                if self.wait(0.3):
                    success = self.attach('bad_fruit')
                    
                    if success:
                        self.attachment_confirmed = True
                        self.picked_fruits.add(self.current_fruit_id)
                        self.get_logger().info(f"✅ Fruit {self.current_fruit_id} ATTACHED")
                        self.ph = 14
                        self.st = time.time()
                    else:
                        self.attach_retry_count += 1
                        if self.attach_retry_count < 2:
                            time.sleep(0.2)
                            self.st = time.time()
                        else:
                            # Skip this fruit
                            self.get_logger().warn(f"Failed to grasp fruit {self.current_fruit_id}")
                            self.skipped_fruits.add(self.current_fruit_id)
                            self.attach_retry_count = 0
                            self.ph = 10
                            self.st = time.time()
        
        elif p == 14:
            if not hasattr(self, 'phase14_target'):
                current_pos, current_quat = self.get_eef_pose()
                if current_quat is not None:
                    rot = R.from_quat(current_quat)
                    ee_z_axis = rot.as_matrix()[:, 2]
                    
                    self.phase14_target = current_pos - 0.17 * ee_z_axis
            
            if hasattr(self, 'phase14_target'):
                self.log_phase(f"Phase 14: Lifting fruit {self.current_fruit_id}")
                if self.servo_to_pos(self.phase14_target):
                    delattr(self, 'phase14_target')
                    self.ph = 15
                    self.st = time.time()
        
        elif p == 15:
            self.log_phase(f"Phase 15: Moving to trash bin with fixed quaternion")

            if self.servo_pose(self.p3, self.bin_quat):
                self.ph = 16
                self.st = time.time()
        
        elif p == 16:
            self.log_phase(f"Phase 16: Detaching fruit {self.current_fruit_id}")
            if self.wait(0.3):
                self.detach('bad_fruit')
                
                total_actual_fruits = len(self.fruit_positions)
                if len(self.picked_fruits) < total_actual_fruits:
                    self.ph = 10
                    self.st = time.time()
                else:
                    self.get_logger().info("All fruits dropped → going to home position")
                    self.ph = 17
                    self.st = time.time()
        
        elif p == 17:
            self.log_phase("Phase 17: Moving to SAFE retreat pose")

            if self.servo_to_pos(self.safe_pos):
                self.ph = 18
                self.st = time.time()

        elif p == 18:
            self.log_phase("Phase 18: Moving to HOME position")

            if self.servo_pose(self.home_pos, self.home_quat):
                self.get_logger().info("Arm reached HOME position")
                self.ph = 99
                self.st = time.time()

        # ========== SECOND DOCKING PHASES ==========
        elif p == 100:
            self.log_phase("Phase 100: Initializing second fertilizer pickup")
            
            # Reset attachment parameters
            self.attach_offset = 0.15
            self.attach_attempt = 0
            self.attachment_confirmed = False
            
            fert_pos = self.PHASE_100_POS
            z_axis = np.array([0.0, 0.0, -1.0])
            
            self.fert_pos = fert_pos
            self.fert_z = z_axis
            
            target_pos = fert_pos - self.attach_offset * z_axis
            
            self.get_logger().info(
                f"Starting 2nd approach - Offset: {self.attach_offset:.3f}m\n"
                f"  Target pos: x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}"
            )

            if self.servo_to_pos(target_pos):
                self.ph = 101
                self.st = time.time()
            else:
                time.sleep(0.05)

        elif p == 101:
            self.log_phase("Phase 101: Aligning for 2nd fertilizer pickup")
            
            fert_pos = self.PHASE_100_POS
            z_axis = np.array([0.0, 0.0, -1.0])
            
            target_pos = fert_pos - self.attach_offset * z_axis
            target_quat = self.PHASE_101_QUAT
            
            if self.servo_pose(target_pos, target_quat):
                self.fert_pos = fert_pos
                self.fert_z = z_axis
                self.ph = 102
                self.st = time.time()

        # ========== PHASE 102: Continuous Approach (2nd fertilizer) ==========
        elif p == 102:
            if self.attach_attempt % 5 == 0:
                self.get_logger().info(
                    f"Phase 102: Approaching 2nd fertilizer - "
                    f"Offset: {self.attach_offset:.3f}m"
                )
            
            z_axis = np.array([0.0, 0.0, -1.0])
            
            target_pos = self.fert_pos - self.attach_offset * z_axis
            target_quat = self.calc_ee_orientation(z_axis, for_pick=True)
            
            if self.servo_pose(target_pos, target_quat):
                self.attach_attempt += 1
                
                if self.attach_offset <= self.attach_min_offset:
                    self.get_logger().info("✓ Reached minimum offset - attaching")
                    self.ph = 103
                    self.st = time.time()
                elif self.attach_attempt >= self.attach_max_attempts:
                    self.get_logger().warn("⚠️ Max attempts - proceeding")
                    self.ph = 103
                    self.st = time.time()
                else:
                    self.attach_offset = max(
                        self.attach_min_offset,
                        self.attach_offset - self.attach_offset_step
                    )
                    self.ph = 102.1
                    self.st = time.time()
        
        elif p == 102.1:
            if self.wait(0.1):
                self.ph = 102
                self.st = time.time()

        elif p == 103:
            self.log_phase("Phase 103: Attaching 2nd fertilizer")
            if self.wait(0.3):
                self.attach('fertiliser_can')

                msg = String()
                msg.data = "CAN_LIFTED"
                self.arm_flag_pub.publish(msg)
                self.get_logger().info("✅ Published CAN_LIFTED - eBot can resume navigation")
                
                self.ph = 104
                self.st = time.time()

        elif p == 104:
            offset = 0.25
            target_pos = self.ebot_pos + offset * self.ebot_z
            target_quat = self.calc_ee_orientation(self.ebot_z, for_pick=False)
            
            self.log_phase("Phase 104: Moving away from eBot")
            
            if self.servo_pose(target_pos, target_quat):
                self.ph = 104.1
                self.st = time.time()

        elif p == 104.1:
            self.log_phase("Phase 104.1: Moving to HOME position")

            if self.servo_pose(self.home_pos, self.home_quat):
                self.get_logger().info("Arm reached HOME position")
                self.ph = 104.5
                self.st = time.time()

        elif p == 104.5:
            if not hasattr(self, 'rotation_target_computed'):
                p_cur, q_cur = self.get_eef_pose()
                if p_cur is None:
                    return
                
                R_cur = R.from_quat(q_cur)
                desired_rotation = np.pi / 2
                Rz = R.from_rotvec([0.0, 0.0, desired_rotation])
                
                self.rotation_p_des = Rz.apply(p_cur)
                self.rotation_R_des = Rz * R_cur
                
                self.rotation_target_computed = True
                self.log_phase("Phase 104.5: Starting virtual base rotation")
            
            p_cur, q_cur = self.get_eef_pose()
            if p_cur is None:
                return
            
            R_cur = R.from_quat(q_cur)
            
            pos_err = self.rotation_p_des - p_cur
            pos_dist = np.linalg.norm(pos_err)
            
            rot_err = self.rotation_R_des * R_cur.inv()
            rot_vec = rot_err.as_rotvec()
            rot_dist = np.linalg.norm(rot_vec)
            
            if pos_dist < 0.01 and rot_dist < 0.02:
                self.stop_t()
                delattr(self, 'rotation_target_computed')
                self.get_logger().info("✅ Virtual base rotation completed")
                self.ph = 104.8
                self.st = time.time()
                return
            
            lin_vel = np.clip(self.K_pos * pos_err, -self.ml, self.ml)
            ang_vel = np.clip(self.K_rot * rot_vec, -self.ma, self.ma)
            
            tw = Twist()
            tw.linear.x = float(lin_vel[0])
            tw.linear.y = float(lin_vel[1])
            tw.linear.z = float(lin_vel[2])
            tw.angular.x = float(ang_vel[0])
            tw.angular.y = float(ang_vel[1])
            tw.angular.z = float(ang_vel[2])
            self.tp.publish(tw)
        
        elif p == 104.8:
            if not hasattr(self, 'rotation_target_computed'):
                p_cur, q_cur = self.get_eef_pose()
                if p_cur is None:
                    return
                
                R_cur = R.from_quat(q_cur)
                desired_rotation = np.pi / 2
                Rz = R.from_rotvec([0.0, 0.0, desired_rotation])
                
                self.rotation_p_des = Rz.apply(p_cur)
                self.rotation_R_des = Rz * R_cur
                
                self.rotation_target_computed = True
                self.log_phase("Phase 104.8: Starting virtual base rotation")
            
            p_cur, q_cur = self.get_eef_pose()
            if p_cur is None:
                return
            
            R_cur = R.from_quat(q_cur)
            
            pos_err = self.rotation_p_des - p_cur
            pos_dist = np.linalg.norm(pos_err)
            
            rot_err = self.rotation_R_des * R_cur.inv()
            rot_vec = rot_err.as_rotvec()
            rot_dist = np.linalg.norm(rot_vec)
            
            if pos_dist < 0.1 and rot_dist < 0.1:
                self.stop_t()
                delattr(self, 'rotation_target_computed')
                self.get_logger().info("✅ Virtual base rotation completed")
                self.ph = 105
                self.st = time.time()
                return
            
            lin_vel = np.clip(self.K_pos * pos_err, -self.ml, self.ml)
            ang_vel = np.clip(self.K_rot * rot_vec, -self.ma, self.ma)
            
            tw = Twist()
            tw.linear.x = float(lin_vel[0])
            tw.linear.y = float(lin_vel[1])
            tw.linear.z = float(lin_vel[2])
            tw.angular.x = float(ang_vel[0])
            tw.angular.y = float(ang_vel[1])
            tw.angular.z = float(ang_vel[2])
            self.tp.publish(tw)

        elif p == 105:
            self.log_phase("Phase 105: Moving fertilizer can to trash bin")
            
            if self.servo_pose(self.p3, self.bin_quat):
                self.ph = 106
                self.st = time.time()

        elif p == 106:
            self.log_phase("Phase 106: Dropping fertilizer can into bin")
            if self.wait(0.3):
                success = self.detach('fertiliser_can')
                if success:
                    self.get_logger().info("✅ Fertilizer can dropped into bin successfully")
                else:
                    self.get_logger().warn("⚠️ Detach may have failed")
                
                self.ph = 99
                self.st = time.time()
        
        elif p == 99:
            self.stop_t()
            self.log_phase("Phase 99: Sequence complete, holding position.")

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