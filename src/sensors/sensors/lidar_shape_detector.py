#!/usr/bin/env python3
'''
# Team ID: <Team-ID>
# Theme: Krishi coBot
# Author List: <Add your names>
# Filename: lidar_raw_data.py
# Functions: scan_callback
# Brief: LiDAR raw data reader publishing every 5 seconds
'''
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np
import time
import json

class LiDARRawReader(Node):
    def __init__(self):
        super().__init__('lidar_raw_reader')
        
        # Subscribe to LiDAR scan topic
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher for raw LiDAR data
        self.data_pub = self.create_publisher(String, '/lidar_raw_data', 10)
        
        self.get_logger().info("📡 LiDAR Raw Data Reader Initialized")
        
        self.last_publish_time = 0
        self.publish_interval = 5.0  # Publish every 5 seconds
        
    def scan_callback(self, msg: LaserScan):
        """Read LiDAR scan and publish raw data every 5 seconds"""
        
        # Check if it's time to publish
        now = time.time()
        if now - self.last_publish_time < self.publish_interval:
            return
            
        self.last_publish_time = now
        
        # Get all ranges and angles
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        self.get_logger().info(f"Total scan points: {len(ranges)}")
        
        # Filter valid readings (distance > 0.02m and not inf/nan)
        valid = (ranges > 0.02) & (ranges < float('inf')) & (~np.isnan(ranges))
        
        # Get valid data
        valid_r = ranges[valid]
        valid_theta = angles[valid]
        
        # Convert to Cartesian coordinates
        xs = valid_r * np.cos(valid_theta)
        ys = valid_r * np.sin(valid_theta)
        
        # Convert theta to degrees
        theta_degrees = np.degrees(valid_theta)
        
        # Now includes ALL angles from -90° to +90° (or full 360° range)
        
        # Create data list
        data_points = []
        for i in range(len(xs)):
            data_points.append({
                'x': float(xs[i]),
                'y': float(ys[i]),
                'r': float(valid_r[i]),
                'theta': float(theta_degrees[i])
            })
        
        # Publish data
        msg_data = {
            'timestamp': now,
            'num_points': len(data_points),
            'points': data_points
        }
        
        json_str = json.dumps(msg_data)
        pub_msg = String()
        pub_msg.data = json_str
        self.data_pub.publish(pub_msg)
        
        self.get_logger().info(
            f"📊 Published {len(data_points)} LiDAR points "
            f"(Theta range: {theta_degrees.min():.1f}° to {theta_degrees.max():.1f}°)"
        )
        
        # Print ALL points
        self.get_logger().info("=== ALL LiDAR Points ===")
        for i, p in enumerate(data_points):
            self.get_logger().info(
                f"Point {i+1}: x={p['x']:.3f}, y={p['y']:.3f}, "
                f"r={p['r']:.3f}, θ={p['theta']:.1f}°"
            )
        self.get_logger().info(f"=== Total: {len(data_points)} points ===")

def main(args=None):
    rclpy.init(args=args)
    node = LiDARRawReader()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()