
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
import numpy as np
import time
import math
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class LiDARRansacVisualizer(Node):
    
    def __init__(self):
        
        super().__init__('lidar_ransac_visualizer')

        # Subscribe to LiDAR scan data
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher for broadcasting detected shape information
        self.shape_pub = self.create_publisher(String, '/detected_shape', 10)
        
        # Subscriber to enable/disable detection on demand
        self.create_subscription(Bool, '/enable_detection', self.enable_detection_callback, 10)
        
        # Subscriber to control filtering of negative theta angles
        self.create_subscription(Bool, '/filter_negative_theta', self.filter_callback, 10)
        
        self.get_logger().info("LiDAR Shape Detection System Initialized")

        # Timing control for processing rate limiting
        self.last_display_time = 0
        self.display_interval = 0.5  # Process every 0.5 seconds
        
        # Storage for latest LiDAR point cloud
        self.latest_points = None
        
        # Detection control flags
        self.detection_enabled = True  # Whether to actively detect shapes
        self.filter_negative_theta = False  # Whether to ignore points with negative angles
        
        # Shape confirmation tracking to avoid false positives
        self.shape_detection_history = []  # Recent detections for confirmation
        self.confirmation_threshold = 2  # Number of consistent detections needed
        self.confirmation_position_tol = 0.3  # Max position variation for confirmation (meters)

    def enable_detection_callback(self, msg: Bool):
        self.detection_enabled = msg.data
        
    def filter_callback(self, msg: Bool):
        
        self.filter_negative_theta = msg.data

    def scan_callback(self, msg: LaserScan):
     
        # Extract range measurements and calculate corresponding angles
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Filter valid points (remove zero/invalid readings and limit to reasonable range)
        valid = (ranges > 0.02) & (ranges < 0.9)
        valid_r = ranges[valid]
        valid_theta = angles[valid]

        # Convert polar coordinates (r, theta) to Cartesian (x, y)
        xs = valid_r * np.cos(valid_theta)
        ys = valid_r * np.sin(valid_theta)
        
        # Create point array with x, y, radius, and angle in degrees
        points = np.vstack((xs, ys, valid_r, np.degrees(valid_theta))).T

        if len(points) == 0:
            return

        # Store latest point cloud for processing
        self.latest_points = points

        # Rate limiting - only process at specified intervals
        now = time.time()
        if now - self.last_display_time >= self.display_interval:
            self.last_display_time = now
            if self.detection_enabled:
                self.process_and_visualize(points)

    def fit_line_auto(self, x_data, y_data, threshold=0.02):
        # Need at least 3 points to fit a reliable line
        if len(x_data) < 3:
            return None
        
        # Determine fitting orientation based on data spread
        x_range = np.ptp(x_data)  # Peak-to-peak (max - min)
        y_range = np.ptp(y_data)
        
        # If x varies less than y, treat as vertical line (x as function of y)
        if x_range < y_range / 2:
            X = y_data.reshape(-1, 1)
            y_target = x_data
            vertical = True
        else:
            # Horizontal or diagonal line (y as function of x)
            X = x_data.reshape(-1, 1)
            y_target = y_data
            vertical = False
        
        # Fit line using RANSAC to reject outliers
        ransac = RANSACRegressor(random_state=42, min_samples=2, residual_threshold=threshold)
        ransac.fit(X, y_target)
        
        # Extract line parameters
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        inliers = np.sum(ransac.inlier_mask_)
        
        return {
            "slope": slope,
            "intercept": intercept,
            "vertical": vertical,
            "n_points": inliers,
            "x_min": np.min(x_data),
            "x_max": np.max(x_data),
            "y_min": np.min(y_data),
            "y_max": np.max(y_data),
        }

    def find_optimal_clusters(self, angles, max_clusters=2):
        # Need at least 3 points for clustering
        if len(angles) < 3:
            return 1
        
        theta = angles.reshape(-1, 1)
        best_score = -1
        best_n = 1
        
        # Try different numbers of clusters and evaluate quality
        for n_clusters in range(2, min(max_clusters + 1, len(angles))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(theta)
            
            # Skip if clustering failed to separate data
            if len(np.unique(labels)) < 2:
                continue
            
            # Silhouette score measures how well-separated clusters are
            score = silhouette_score(theta, labels)
            if score > best_score:
                best_score = score
                best_n = n_clusters
        
        return best_n

    def cluster_by_angle(self, x_data, y_data, angles, n_clusters):
        if len(x_data) == 0:
            return []
        
        # Cluster based on angle only
        theta = angles.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(theta)
        
        # Separate points into clusters
        clusters = []
        for i in range(n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                clusters.append((x_data[mask], y_data[mask], angles[mask]))
        
        return clusters

    def filter_similar_lines(self, lines, slope_threshold=1.0, intercept_threshold=0.1):
        filtered_lines = []
        
        for i, line1 in enumerate(lines):
            is_duplicate = False
            
            # Check against already filtered lines
            for j, line2 in enumerate(filtered_lines):
                # Only compare lines with same orientation (both vertical or both horizontal)
                if line1["vertical"] == line2["vertical"]:
                    slope_diff = abs(line1["slope"] - line2["slope"])
                    intercept_diff = abs(line1["intercept"] - line2["intercept"])
                    
                    # Lines are similar if both slope and intercept are close
                    if slope_diff < slope_threshold and intercept_diff < intercept_threshold:
                        is_duplicate = True
                        # Keep the line with more supporting points
                        if line1["n_points"] > line2["n_points"]:
                            filtered_lines[j] = line1
                        break
            
            if not is_duplicate:
                filtered_lines.append(line1)
        
        return filtered_lines

    def calculate_angle_between_lines(self, line1, line2):
        
        # Convert vertical line representations to standard slopes
        if line1["vertical"] and line2["vertical"]:
            m1 = 1.0 / line1["slope"] if line1["slope"] != 0 else np.inf
            m2 = 1.0 / line2["slope"] if line2["slope"] != 0 else np.inf
        elif line1["vertical"]:
            m1 = 1.0 / line1["slope"] if line1["slope"] != 0 else np.inf
            m2 = line2["slope"]
        elif line2["vertical"]:
            m1 = line1["slope"]
            m2 = 1.0 / line2["slope"] if line2["slope"] != 0 else np.inf
        else:
            m1 = line1["slope"]
            m2 = line2["slope"]
        
        # Handle vertical lines (infinite slope)
        if np.isinf(m1) or np.isinf(m2):
            return 90.0
        
        # Calculate angle using arctangent formula
        angle = np.abs(np.arctan((m2 - m1) / (1 + m1 * m2)))
        return np.degrees(angle)

    def count_lines(self, lines):
       
        return len(lines)

    def process_and_visualize(self, points):
    
        # Extract coordinate components
        x = points[:, 0]
        y = points[:, 1]
        angles = points[:, 3]

        # Separate points by positive and negative angles
        mask_pos = angles > 0
        mask_neg = angles < 0

        x_pos, y_pos, ang_pos = x[mask_pos], y[mask_pos], angles[mask_pos]
        x_neg, y_neg, ang_neg = x[mask_neg], y[mask_neg], angles[mask_neg]

        lines_pos = []
        lines_neg = []

        # Process positive angle points (front-facing)
        if len(ang_pos) > 0:
            # Automatically determine number of line segments
            n_clusters_pos = self.find_optimal_clusters(ang_pos)
            clusters_pos = self.cluster_by_angle(x_pos, y_pos, ang_pos, n_clusters_pos)
            
            # Fit line to each cluster
            for i, (xc, yc, ac) in enumerate(clusters_pos):
                result = self.fit_line_auto(xc, yc)
                # Only keep lines with sufficient support points
                if result and result["n_points"] >= 30:
                    result["name"] = f"Positive Segment {i+1}"
                    lines_pos.append(result)

        # Process negative angle points (side/rear facing) if not filtered
        if len(ang_neg) > 0 and not self.filter_negative_theta:
            n_clusters_neg = self.find_optimal_clusters(ang_neg)
            clusters_neg = self.cluster_by_angle(x_neg, y_neg, ang_neg, n_clusters_neg)
            
            for i, (xc, yc, ac) in enumerate(clusters_neg):
                result = self.fit_line_auto(xc, yc)
                if result and result["n_points"] >= 30:
                    result["name"] = f"Negative Segment {i+1}"
                    lines_neg.append(result)

        # Remove duplicate/similar lines
        lines_pos = self.filter_similar_lines(lines_pos)
        lines_neg = self.filter_similar_lines(lines_neg)

        # Shape detection logic
        detected_shape = None
        shape_position = None
        all_lines = lines_pos + lines_neg
        
        # Check for Pentagon - needs exactly 5 distinct line segments
        if len(all_lines) >= 5:
            # Calculate centroid of detected pentagon
            avg_x = sum(line['x_min'] + line['x_max'] for line in all_lines[:5]) / 10
            avg_y = sum(line['y_min'] + line['y_max'] for line in all_lines[:5]) / 10
            detected_shape = "PENTAGON"
            shape_position = (avg_x, avg_y)
        
        # Check for Triangle - needs 2 lines at 30-80 degree angle
        elif len(lines_pos) >= 2:
            for i in range(len(lines_pos)):
                for j in range(i + 1, len(lines_pos)):
                    angle = self.calculate_angle_between_lines(lines_pos[i], lines_pos[j])
                    
                    # Calculate average position of the two lines
                    avg_x = (lines_pos[i]['x_min'] + lines_pos[i]['x_max'] + 
                            lines_pos[j]['x_min'] + lines_pos[j]['x_max']) / 4
                    avg_y = (lines_pos[i]['y_min'] + lines_pos[i]['y_max'] + 
                            lines_pos[j]['y_min'] + lines_pos[j]['y_max']) / 4
                    
                    # Triangle angles typically between 30-80 degrees
                    if 30 <= angle <= 80:
                        detected_shape = "TRIANGLE"
                        shape_position = (avg_x, avg_y)
                        break
                if detected_shape:
                    break
        
        # Check for Square - needs parallel or perpendicular lines
        if not detected_shape and len(lines_pos) >= 2:
            for i in range(len(lines_pos)):
                for j in range(i + 1, len(lines_pos)):
                    angle = self.calculate_angle_between_lines(lines_pos[i], lines_pos[j])
                    intercept_diff = abs(lines_pos[i]['intercept'] - lines_pos[j]['intercept'])
                    
                    avg_x = (lines_pos[i]['x_min'] + lines_pos[i]['x_max'] + 
                            lines_pos[j]['x_min'] + lines_pos[j]['x_max']) / 4
                    avg_y = (lines_pos[i]['y_min'] + lines_pos[i]['y_max'] + 
                            lines_pos[j]['y_min'] + lines_pos[j]['y_max']) / 4
                    
                    # Square has either parallel (0-20°) or perpendicular (80-90°) sides
                    # Also check intercept difference to ensure lines are separated
                    if (0 <= angle <= 20 or 80 <= angle <= 90) and intercept_diff > 0.2:
                        detected_shape = "SQUARE"
                        shape_position = (avg_x, avg_y)
                        break
                if detected_shape:
                    break
        
        # Check negative theta segments for shapes if enabled
        if not detected_shape and len(lines_neg) >= 2 and not self.filter_negative_theta:
            for i in range(len(lines_neg)):
                for j in range(i + 1, len(lines_neg)):
                    angle = self.calculate_angle_between_lines(lines_neg[i], lines_neg[j])
                    intercept_diff = abs(lines_neg[i]['intercept'] - lines_neg[j]['intercept'])
                    
                    avg_x = (lines_neg[i]['x_min'] + lines_neg[i]['x_max'] + 
                            lines_neg[j]['x_min'] + lines_neg[j]['x_max']) / 4
                    avg_y = (lines_neg[i]['y_min'] + lines_neg[i]['y_max'] + 
                            lines_neg[j]['y_min'] + lines_neg[j]['y_max']) / 4
                    
                    if 30 <= angle <= 80:
                        detected_shape = "TRIANGLE"
                        shape_position = (avg_x, avg_y)
                        break
                    elif (0 <= angle <= 20 or 80 <= angle <= 90) and intercept_diff > 0.2:
                        detected_shape = "SQUARE"
                        shape_position = (avg_x, avg_y)
                        break
                if detected_shape:
                    break

        # Confirmation logic to reduce false positives
        if detected_shape and shape_position:
            current_time = time.time()
            
            # Add current detection to history
            self.shape_detection_history.append((detected_shape, shape_position[0], shape_position[1], current_time))
            
            # Remove old detections (older than 2 seconds)
            self.shape_detection_history = [
                det for det in self.shape_detection_history 
                if current_time - det[3] < 2.0
            ]
            
            # Check if we have enough recent detections for confirmation
            if len(self.shape_detection_history) >= self.confirmation_threshold:
                recent_detections = self.shape_detection_history[-self.confirmation_threshold:]
                shape_types = [det[0] for det in recent_detections]
                
                # All recent detections must be the same shape
                if len(set(shape_types)) == 1:
                    positions = [(det[1], det[2]) for det in recent_detections]
                    avg_x = sum(p[0] for p in positions) / len(positions)
                    avg_y = sum(p[1] for p in positions) / len(positions)
                    
                    # Check that all positions are close together (not jumping around)
                    all_similar = all(
                        math.hypot(p[0] - avg_x, p[1] - avg_y) < self.confirmation_position_tol
                        for p in positions
                    )
                    
                    # Publish confirmed detection
                    if all_similar:
                        msg = String()
                        msg.data = f"{detected_shape}|{avg_x:.3f}|{avg_y:.3f}"
                        self.shape_pub.publish(msg)
                        # Clear history after publishing to avoid repeated publications
                        self.shape_detection_history.clear()


def main(args=None):
    # Initialize ROS2 system
    rclpy.init(args=args)
    
    # Create the LiDAR detection node
    node = LiDARRansacVisualizer()
    
    # Keep node running and processing callbacks
    rclpy.spin(node)
    
    # Clean up on shutdown
    node.destroy_node()
    rclpy.shutdown()


# Standard Python idiom to check if this script is being run directly
if __name__ == '__main__':
    main()