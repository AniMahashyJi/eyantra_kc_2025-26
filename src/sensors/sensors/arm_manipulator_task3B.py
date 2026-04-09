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
        - Moves arm to target positions
        - Opens/closes gripper
        - Optionally reads TFs from perception node
    """
    def __init__(self):
        
        super().__init__('pick_place')
        # State and timer
        self.dock_reached = False

        self.dock_sub = self.create_subscription(
            String,
            '/detection_status',
            self.dock_cb,
            10
        )

        self.arm_flag_pub = self.create_publisher(
            String,
            '/arm_flag',
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
        
        self.get_logger().info("Pick and Place Started")

    def dock_cb(self, msg: String):
        if msg.data.startswith("DOCK_STATION") and not self.dock_reached:
            self.get_logger().info("Dock reached → starting arm sequence (ONLY ONCE)")
            self.dock_reached = True
            self.ph = 0            # start from beginning
            self.st = time.time()  # reset timer

     # ===================== TF & POSE UTILITIES =====================        

    def get_tf(self, target_frame):
        try:
            # NOTE: relying on default timeout; exceptions are caught below
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

    def upd(self):
        # Phase execution is exactly as in your previous code
        # Only changes:
        # - Removed all time.sleep() → use non-blocking wait()
        # - Logging structured
        # - Only stop_t() in phase 99

        if not self.dock_reached:
            return
        p = self.ph
        
        if p == 0:
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')

            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                offset = 0.15
                target_pos = fert_pos - offset * z_axis
                
                self.log_phase("Phase 0: Moving to fertilizer approach")
                
                if self.servo_to_pos(target_pos):
                    self.fert_pos = fert_pos
                    self.fert_z = z_axis
                    self.ph = 1
                    self.st = time.time()
            else:
                time.sleep(0.05)
        
        elif p == 1:
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')

            
            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                offset = 0.15
                target_pos = fert_pos - offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=True)
                
                self.log_phase("Phase 1: Rotating for fertilizer")
                
                if self.servo_pose(target_pos, target_quat):
                    self.fert_pos = fert_pos
                    self.fert_z = z_axis
                    self.ph = 2
                    self.st = time.time()
        
        elif p == 2:
            fert_pos, fert_quat = self.get_tf(f'{TEAM_ID}_fertilizer_1')

            
            if fert_pos is not None:
                rot = R.from_quat(fert_quat)
                z_axis = rot.as_matrix()[:, 2]
                
                offset = 0.05
                target_pos = fert_pos - offset * z_axis
                target_quat = self.calc_ee_orientation(z_axis, for_pick=True)
                
                self.log_phase("Phase 2: Grasping fertilizer")
                
                if self.servo_pose(target_pos, target_quat):
                    self.fert_pos = fert_pos
                    self.fert_z = z_axis
                    self.ph = 3
                    self.st = time.time()
        
        elif p == 3:
            self.log_phase("Phase 3: Attaching fertilizer")
            if self.wait(0.3):
                self.attach('fertiliser_can')
                self.ph = 4
                self.st = time.time()
        
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
                    self.ph = 5
                    self.st = time.time()
            else:
                time.sleep(0.05)
        
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
                    # ✅ FIX: correct TF names as per Task 3A
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
        
        elif p == 12:
            if not hasattr(self, 'phase12_target'):
                current_pos, _ = self.get_eef_pose()
                if current_pos is not None:
                    approach_offset = 0.08
                    self.phase12_target = np.array([current_pos[0], current_pos[1], current_pos[2] - approach_offset])
            
            if hasattr(self, 'phase12_target'):
                self.log_phase(f"Phase 12: Approaching fruit {self.current_fruit_id}")
                if self.servo_to_pos(self.phase12_target):
                    delattr(self, 'phase12_target')
                    self.ph = 13
                    self.st = time.time()
        
        elif p == 13:
            current_pos, _ = self.get_eef_pose()
            if current_pos is not None:
                grasp_offset = 0.03
                target_pos = np.array([current_pos[0], current_pos[1], current_pos[2] - grasp_offset])
                
                self.log_phase(f"Phase 13: Grasping fruit {self.current_fruit_id}")
                
                if self.servo_to_pos(target_pos):
                    self.ph = 14
                    self.st = time.time()
        
        elif p == 14:
            self.log_phase(f"Phase 14: Attaching fruit {self.current_fruit_id}")
            if self.wait(0.3):
                self.attach('bad_fruit')
                self.picked_fruits.add(self.current_fruit_id)
                self.ph = 15
                self.st = time.time()
        
        elif p == 15:
            if not hasattr(self, 'phase15_target'):
                current_pos, current_quat = self.get_eef_pose()
                if current_quat is not None:
                    rot = R.from_quat(current_quat)
                    ee_z_axis = rot.as_matrix()[:, 2]
                    
                    self.phase15_target = current_pos - 0.17 * ee_z_axis
            
            if hasattr(self, 'phase15_target'):
                self.log_phase(f"Phase 15: Lifting fruit {self.current_fruit_id}")
                if self.servo_to_pos(self.phase15_target):
                    delattr(self, 'phase15_target')
                    self.ph = 16
                    self.st = time.time()
        
        elif p == 16:
            target_pos = self.p3
            
            self.log_phase(f"Phase 16: Moving to trash bin with fruit {self.current_fruit_id}")
            
            if self.servo_to_pos(target_pos):
                self.ph = 17
                self.st = time.time()
        
        elif p == 17:
            self.log_phase(f"Phase 17: Detaching fruit {self.current_fruit_id}")
            if self.wait(0.3):
                self.detach('bad_fruit')
                
                total_actual_fruits = len(self.fruit_positions)
                if len(self.picked_fruits) < total_actual_fruits:
                    self.ph = 10
                    self.st = time.time()
                else:
                    self.get_logger().info("All tasks complete")
                    self.ph = 99
        
        elif p == 99:
            # ✅ FIX: do NOT call rclpy.shutdown() here
            # Just stop sending twist and stay idle; main() will handle shutdown.
            self.stop_t()
            # Only log once
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