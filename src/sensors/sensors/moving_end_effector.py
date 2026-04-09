#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

TEAM_ID = "2635"

class RotationTester(Node):
    def __init__(self):
        super().__init__('rotation_tester')
        
        self.tp = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)
        
        # Test different axes
        self.test_axes = {
            'X+ (towards positive X)': np.array([1.0, 0.0, 0.0]),
            'X- (towards negative X)': np.array([-1.0, 0.0, 0.0]),
            'Y+ (towards positive Y)': np.array([0.0, 1.0, 0.0]),
            'Y- (towards negative Y)': np.array([0.0, -1.0, 0.0]),
            'Z+ (upward)': np.array([0.0, 0.0, 1.0]),
            'Z- (downward)': np.array([0.0, 0.0, -1.0]),
        }
        
        self.current_test = 0
        self.test_names = list(self.test_axes.keys())
        
        self.K_rot = 1.8
        self.ma = 1.5
        self.rt = 0.10
        
        self.aligned = False
        self.align_time = None
        self.hold_duration = 100.0  # Hold for 5 seconds before switching
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🔄 ROTATION TESTER - Auto-cycles every 5 seconds")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Testing: {self.test_names[self.current_test]}")
        
        self.timer = self.create_timer(0.02, self.update)
        
    def get_eef_pose(self):
        try:
            t = self.tf_buf.lookup_transform('base_link', 'tool0', rclpy.time.Time())
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
    
    def calc_ee_orientation(self, z_axis):
        """Align end-effector's Z-axis with given direction"""
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        ee_z = z_axis
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
        
        ee_x = ee_x / np.linalg.norm(ee_x)
        ee_y = ee_y / np.linalg.norm(ee_y)
        ee_z = ee_z / np.linalg.norm(ee_z)
        
        rot_mat = np.column_stack([ee_x, ee_y, ee_z])
        quat = R.from_matrix(rot_mat).as_quat()
        return quat
    
    def update(self):
        axis_name = self.test_names[self.current_test]
        target_axis = self.test_axes[axis_name]
        
        _, eef_quat = self.get_eef_pose()
        if eef_quat is None:
            return
        
        target_quat = self.calc_ee_orientation(target_axis)
        
        eef_rot = R.from_quat(eef_quat)
        target_rot = R.from_quat(target_quat)
        rot_err = target_rot * eef_rot.inv()
        rot_vec = rot_err.as_rotvec()
        rot_dist = np.linalg.norm(rot_vec)
        
        # Check if aligned
        if rot_dist < self.rt:
            if not self.aligned:
                self.aligned = True
                self.align_time = time.time()
                self.get_logger().info(f"✅ ALIGNED with {axis_name}")
                self.get_logger().info("👀 CHECK: Does gripper point toward the can?")
            
            # Hold for duration, then switch
            if time.time() - self.align_time > self.hold_duration:
                self.switch_axis()
            
            # Stop movement
            tw = Twist()
            self.tp.publish(tw)
            return
        
        # Not aligned yet - keep rotating
        self.aligned = False
        ang_vel = np.clip(self.K_rot * rot_vec, -self.ma, self.ma)
        
        tw = Twist()
        tw.angular.x = float(ang_vel[0])
        tw.angular.y = float(ang_vel[1])
        tw.angular.z = float(ang_vel[2])
        self.tp.publish(tw)
    
    def switch_axis(self):
        self.current_test = (self.current_test + 1) % len(self.test_names)
        self.aligned = False
        self.align_time = None
        
        self.get_logger().info("\n" + "=" * 60)
        self.get_logger().info(f"🔄 Switching to: {self.test_names[self.current_test]}")
        self.get_logger().info("=" * 60 + "\n")

def main():
    rclpy.init()
    node = RotationTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\n🛑 Shutting down...")
    finally:
        tw = Twist()
        node.tp.publish(tw)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()