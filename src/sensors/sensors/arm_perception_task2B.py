#!/usr/bin/python3
# -*- coding: utf-8 -*-


'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  TEAM ID:          2635
*  TEAM MEMBERS:     Animesh, Swayam, Harshit, Gautam
*  FILENAME:         task3b_perception.py
*  DESCRIPTION:
*       Detects:
*         - Bad fruits (grey color)
*         - ArUco markers (DICT 4x4_50)
*
*       Publishes TFs:
*         - camera_link → bad fruit
*         - camera_link → ArUco
*         - base_link  → bad fruit / ArUco
*
*****************************************************************************************
'''

import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


SHOW_IMAGE = True
DISABLE_MULTITHREADING = False
TEAM_ID = "2635"


class CombinedDetector(Node):
    """
    PURPOSE:
        - Subscribes to RGB + Depth images
        - Detects bad fruits and ArUco markers
        - Publishes TF transforms of detected objects
        - Converts depth values to metric distance
        - Visual debugging through OpenCV windows

    INPUTS  :
        /camera/image_raw  - RGB image
        /camera/depth/image_raw - aligned depth

    OUTPUTS :
        TF:
            camera_link → bad fruit
            camera_link → aruco marker
            base_link   → bad fruit
            base_link   → aruco marker
    """
    def __init__(self):
        super().__init__('combined_detector')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.window_initialized = False
        self.bad_window_open = False
        self.aruco_window_open = False

        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        self.create_subscription(Image, '/camera/image_raw',
                                 self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_raw',
                                 self.depthimagecb, 10, callback_group=self.cb_group)

        self.br = tf2_ros.TransformBroadcaster(self)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.cam_mat = np.array([[915.3003540039062, 0.0, 642.724365234375],
                                 [0.0, 914.0320434570312, 361.9780578613281],
                                 [0.0, 0.0, 1.0]])
        self.dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.aruco_size = 0.13

        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)
        self.create_timer(1.0, self.print_camera_to_base, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('Actual_RGB', cv2.WINDOW_NORMAL)

        self.get_logger().info("Combined detector node started.")

         # ===========================================================================
    #                         IMAGE CALLBACKS
    # ===========================================================================

    def depthimagecb(self, data):
        """Converts depth image topic to cv2 matrix"""
        
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def colorimagecb(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Color conversion failed: {e}")

              # ===========================================================================
    #                          BAD FRUIT DETECTION
    # ===========================================================================

    def bad_fruit_detection(self, rgb_image):
        """
        Detect grey-ish fruits using HSV thresholding.
        Returns list of detected fruits.
        """
        
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        lower_grey = np.array([10, 10, 110])
        upper_grey = np.array([20, 40, 160])
        mask = cv2.inRange(hsv, lower_grey, upper_grey)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bad_fruits = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cX, cY = x + w // 2, y + h // 2
            bad_fruits.append(((cX, cY), area, w, h))
        return bad_fruits, mask
    
     # ===========================================================================
    #                            ARUCO DETECTION
    # ===========================================================================

    def detect_aruco(self, image):
        """
        Detects ArUco markers and estimates pose.
        Returns list of detected markers + annotated image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)
        
        markers = []
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.aruco_size, self.cam_mat, self.dist_mat)
            
            for i, corner in enumerate(corners):
                area = cv2.contourArea(corner)
                
                if area < 1500:
                    continue
                
                cX = int(corner[0][:, 0].mean())
                cY = int(corner[0][:, 1].mean())
                
                marker_id = ids[i][0]
                rvec = rvecs[i][0]
                
                cv2.drawFrameAxes(image, self.cam_mat, self.dist_mat, 
                                rvec, tvecs[i][0], 0.1)
                
                cv2.circle(image, (cX, cY), 7, (255, 0, 0), -1)
                cv2.putText(image, f"ID:{marker_id}", (cX - 30, cY - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                markers.append((marker_id, cX, cY, area, rvec))
        
        return markers, image
    # ===========================================================================
    #                          UTILITY  (Quaternion / Rotation)
    # ===========================================================================

    def quaternion_multiply(self, q1, q2):
        """Multiplies two quaternions."""

       
    
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return [x, y, z, w]

    def rotate_vector(self, q, v):
        """Rotates vector v by quaternion q."""
        x, y, z = v
        qx, qy, qz, qw = q
        vx, vy, vz, vw = x, y, z, 0
        q_conj = [-qx, -qy, -qz, qw]
        v_rot = self.quaternion_multiply(
            self.quaternion_multiply([qx, qy, qz, qw], [vx, vy, vz, vw]),
            q_conj
        )
        return v_rot[:3]
    
      # ===========================================================================
    #                      PUBLISH TF — BAD FRUIT
    # ===========================================================================

    def publish_fruit_tf(self, fruit_id, cX, cY, distance_from_rgb_m):
        """Publishes TF for detected bad fruit."""
        try:
            if distance_from_rgb_m is None:
                return
            if isinstance(distance_from_rgb_m, float):
                if np.isnan(distance_from_rgb_m) or distance_from_rgb_m <= 0.0:
                    return
            else:
                try:
                    distance_from_rgb_m = float(distance_from_rgb_m)
                    if np.isnan(distance_from_rgb_m) or distance_from_rgb_m <= 0.0:
                        return
                except:
                    return

            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 642.724365234375
            centerCamY = 361.9780578613281
            focalX = 915.3003540039062
            focalY = 914.0320434570312

            x = distance_from_rgb_m * (sizeCamX - cX - centerCamX) / focalX
            y = distance_from_rgb_m * (sizeCamY - cY - centerCamY) / focalY
            z = distance_from_rgb_m

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "camera_link"
            t.child_frame_id = f"2635_bad_fruit_{fruit_id}_"

            t.transform.translation.x = float(z)
            t.transform.translation.y = float(x)
            t.transform.translation.z = float(y)

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.br.sendTransform(t)

            try:
                cam_to_base = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_link', rclpy.time.Time()
                )

                bf_to_cam_translation = [t.transform.translation.x,
                                         t.transform.translation.y,
                                         t.transform.translation.z]
                bf_to_cam_rotation = [t.transform.rotation.x,
                                      t.transform.rotation.y,
                                      t.transform.rotation.z,
                                      t.transform.rotation.w]

                cam_to_base_translation = [cam_to_base.transform.translation.x,
                                           cam_to_base.transform.translation.y,
                                           cam_to_base.transform.translation.z]
                cam_to_base_rotation = [cam_to_base.transform.rotation.x,
                                        cam_to_base.transform.rotation.y,
                                        cam_to_base.transform.rotation.z,
                                        cam_to_base.transform.rotation.w]

                rotated_translation = self.rotate_vector(cam_to_base_rotation, bf_to_cam_translation)
                final_translation = [rotated_translation[i] + cam_to_base_translation[i] for i in range(3)]
                final_rotation = self.quaternion_multiply(cam_to_base_rotation, bf_to_cam_rotation)

                bad_to_base = TransformStamped()
                bad_to_base.header.stamp = self.get_clock().now().to_msg()
                bad_to_base.header.frame_id = "base_link"
                bad_to_base.child_frame_id = f"{2635}_bad_fruit_{fruit_id}"
                bad_to_base.transform.translation.x = final_translation[0]
                bad_to_base.transform.translation.y = final_translation[1]
                bad_to_base.transform.translation.z = final_translation[2]
                bad_to_base.transform.rotation.x = 0.0
                bad_to_base.transform.rotation.y = 0.707
                bad_to_base.transform.rotation.z = 0.0
                bad_to_base.transform.rotation.w = 0.707

                self.br.sendTransform(bad_to_base)

            except Exception as e:
                self.get_logger().warn(f"Could not publish bad_fruit -> base_link transform: {e}")

        except Exception as e:
            self.get_logger().error(f"Error publishing TF: {e}")

             # ===========================================================================
    #                      PUBLISH TF — ARUCO MARKER
    # ===========================================================================

    def publish_aruco_tf(self, marker_id, cX, cY, distance_from_rgb_m, rvec):
        """Publishes TF for detected ArUco marker."""
        try:
            if distance_from_rgb_m is None:
                return
            if isinstance(distance_from_rgb_m, float):
                if np.isnan(distance_from_rgb_m) or distance_from_rgb_m <= 0.0:
                    return
            else:
                try:
                    distance_from_rgb_m = float(distance_from_rgb_m)
                    if np.isnan(distance_from_rgb_m) or distance_from_rgb_m <= 0.0:
                        return
                except:
                    return

            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 642.724365234375
            centerCamY = 361.9780578613281
            focalX = 915.3003540039062
            focalY = 914.0320434570312

            x = distance_from_rgb_m * (sizeCamX - cX - centerCamX) / focalX
            y = distance_from_rgb_m * (sizeCamY - cY - centerCamY) / focalY
            z = distance_from_rgb_m

            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            z_axis = rotation_matrix[:, 2]
            
            if z_axis[2] > 0:
                flip_matrix = np.array([[1, 0, 0],
                                       [0, -1, 0],
                                       [0, 0, -1]])
                rotation_matrix = rotation_matrix @ flip_matrix
            
            rot = R.from_matrix(rotation_matrix)
            quat_cam = rot.as_quat()

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "camera_link"
            t.child_frame_id = f"cam_{marker_id}_"

            t.transform.translation.x = float(z)
            t.transform.translation.y = float(x)
            t.transform.translation.z = float(y)

            t.transform.rotation.x = float(quat_cam[0])
            t.transform.rotation.y = float(quat_cam[1])
            t.transform.rotation.z = float(quat_cam[2])
            t.transform.rotation.w = float(quat_cam[3])

            self.br.sendTransform(t)

            try:
                cam_to_base = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_link', rclpy.time.Time()
                )

                aruco_to_cam_translation = [t.transform.translation.x,
                                            t.transform.translation.y,
                                            t.transform.translation.z]
                aruco_to_cam_rotation = [t.transform.rotation.x,
                                         t.transform.rotation.y,
                                         t.transform.rotation.z,
                                         t.transform.rotation.w]

                cam_to_base_translation = [cam_to_base.transform.translation.x,
                                           cam_to_base.transform.translation.y,
                                           cam_to_base.transform.translation.z]
                cam_to_base_rotation = [cam_to_base.transform.rotation.x,
                                        cam_to_base.transform.rotation.y,
                                        cam_to_base.transform.rotation.z,
                                        cam_to_base.transform.rotation.w]

                rotated_translation = self.rotate_vector(cam_to_base_rotation, aruco_to_cam_translation)
                final_translation = [rotated_translation[i] + cam_to_base_translation[i] for i in range(3)]
                
                if marker_id == 6:
                    final_rotation = [0.0, 0.0, 0.707, 0.707]
                elif marker_id == 3:
                    final_rotation = [0.707, 0.0, 0.0, 0.707]
                else:
                    final_rotation = self.quaternion_multiply(cam_to_base_rotation, aruco_to_cam_rotation)

                aruco_to_base = TransformStamped()
                aruco_to_base.header.stamp = self.get_clock().now().to_msg()
                aruco_to_base.header.frame_id = "base_link"
                
                if marker_id == 3:
                    aruco_to_base.child_frame_id = f"{TEAM_ID}_fertilizer_1"
                    self.get_logger().info(f"Detected Fertilizer (ID {marker_id})")

                elif marker_id == 6:
                    aruco_to_base.child_frame_id = f"{TEAM_ID}_ebot_aruco"
                    self.get_logger().info(f"Detected eBot ArUco (ID {marker_id}) - Rack mounted")
                else:
                    aruco_to_base.child_frame_id = f"obj_{marker_id}"
                
                aruco_to_base.transform.translation.x = final_translation[0]
                aruco_to_base.transform.translation.y = final_translation[1]
                aruco_to_base.transform.translation.z = final_translation[2]
                
                aruco_to_base.transform.rotation.x = final_rotation[0]
                aruco_to_base.transform.rotation.y = final_rotation[1]
                aruco_to_base.transform.rotation.z = final_rotation[2]
                aruco_to_base.transform.rotation.w = final_rotation[3]

                self.br.sendTransform(aruco_to_base)

            except Exception as e:
                self.get_logger().warn(f"Could not publish aruco -> base_link transform: {e}")

        except Exception as e:
            self.get_logger().error(f"Error publishing ArUco TF: {e}")

            # ===========================================================================
    #                 PRINT CAMERA→BASE TRANSFORM (DEBUG)
    # ===========================================================================

    def print_camera_to_base(self):
        """Prints transform camera_link → base_link."""
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
            self.get_logger().info(
                f"Transform from camera_link → base_link: "
                f"translation=({t.transform.translation.x:.3f}, "
                f"{t.transform.translation.y:.3f}, {t.transform.translation.z:.3f}), "
                f"rotation=({t.transform.rotation.x:.3f}, {t.transform.rotation.y:.3f}, "
                f"{t.transform.rotation.z:.3f}, {t.transform.rotation.w:.3f})"
            )
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")
   # ===========================================================================
    #                  DEPTH EXTRACTOR (pixel → meters)
    # ===========================================================================
    def _get_depth_meters(self, px, py):
        """Returns depth at pixel (px, py) in meters."""
        try:
            h, w = self.depth_image.shape[:2]
            if px < 0 or py < 0 or px >= w or py >= h:
                return np.nan
            raw = self.depth_image[int(py), int(px)]
        except Exception:
            return np.nan

        try:
            raw_val = float(raw)
        except Exception:
            return np.nan

        dtype = getattr(self.depth_image, 'dtype', None)
        if dtype is not None:
            if np.issubdtype(dtype, np.integer):
                depth_m = raw_val / 1000.0
                if depth_m <= 0.0:
                    return np.nan
                return depth_m
            else:
                depth_m = raw_val
                if depth_m <= 0.0 or np.isnan(depth_m):
                    return np.nan
                if depth_m > 20.0:
                    depth_m = depth_m / 1000.0
                return depth_m

        if raw_val <= 0.0 or np.isnan(raw_val):
            return np.nan
        if raw_val > 20.0:
            return raw_val / 1000.0
        return raw_val

    def process_image(self):
        """
        Processes RGB + Depth:
            - Detect bad fruits
            - Detect ArUco markers
            - Publish TFs
            - Draw annotations
        """
        if self.cv_image is None or self.depth_image is None:
            return

        rgb = self.cv_image.copy()
        
        bad_fruits, mask = self.bad_fruit_detection(rgb)
        markers, annotated = self.detect_aruco(rgb)

        if SHOW_IMAGE:
            try:
                cv2.imshow('Actual_RGB', rgb)
                cv2.waitKey(1)
            except cv2.error:
                pass

        if len(bad_fruits) > 0:
            for idx, ((cX, cY), area, w, h) in enumerate(bad_fruits, start=1):
                depth_m = self._get_depth_meters(cX, cY)
                self.publish_fruit_tf(idx, cX, cY, depth_m)

                cv2.circle(rgb, (cX - 5, cY - 20), 6, (0, 255, 0), -1)
                cv2.putText(rgb, "bad_fruit", (cX - w // 2, cY - h // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(rgb, (x := cX - w // 2, y := cY - h // 2 - 10),
                              (x + w, y + h), (0, 255, 0), 2)

            if SHOW_IMAGE:
                if not self.bad_window_open:
                    cv2.namedWindow('Bad_Fruit_Detection', cv2.WINDOW_NORMAL)
                    self.bad_window_open = True
                try:
                    cv2.imshow('Bad_Fruit_Detection', rgb)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
        else:
            if self.bad_window_open:
                cv2.destroyWindow('Bad_Fruit_Detection')
                self.bad_window_open = False

        if len(markers) > 0:
            for marker_id, cX, cY, area, rvec in markers:
                depth_m = self._get_depth_meters(cX, cY)
                self.publish_aruco_tf(marker_id, cX, cY, depth_m, rvec)

            if SHOW_IMAGE:
                try:
                    if not self.aruco_window_open:
                        cv2.namedWindow('ArUco_Detection', cv2.WINDOW_NORMAL)
                        self.aruco_window_open = True
                    cv2.imshow('ArUco_Detection', annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
        else:
            if self.aruco_window_open:
                cv2.destroyWindow('ArUco_Detection')
                self.aruco_window_open = False


def main(args=None):
    rclpy.init(args=args)
    node = CombinedDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Combined Detector")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()