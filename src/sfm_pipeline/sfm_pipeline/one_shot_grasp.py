import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter

from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import open3d as o3d
from PIL import Image as PILImage

from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import sensor_msgs_py.point_cloud2 as pc2


class AutoGraspScanner(Node):
    def __init__(self):
        super().__init__(
            'auto_grasp_scanner',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )

        self.bridge = CvBridge()

        self.last_grasp_pose = None

        self.image_sub = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )

        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/grasp_scan_command',
            self.command_callback,
            10
        )

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/ur_manipulator_controller/joint_trajectory',
            10
        )

        self.pc_pub = self.create_publisher(
            PointCloud2,
            '/ai_scanned_pointcloud',
            10
        )

        self.grasp_pub = self.create_publisher(
            PoseStamped,
            '/target_grasp_pose',
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            '/target_grasp_marker',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(
            f"Loading ZoeDepth model on [{self.device.type.upper()}]..."
        )

        self.model = torch.hub.load(
            "isl-org/ZoeDepth",
            "ZoeD_N",
            pretrained=True
        )

        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info("ZoeDepth loaded.")

        # Frames
        self.target_frame = "base_link"
        self.camera_frame = "camera_link"

        # Camera intrinsics
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.image_width = 0
        self.image_height = 0
        self.camera_info_received = False

        # Current working image correction
        self.flip_image_vertical = True
        self.flip_image_horizontal = False

        # Crop RGB/depth bottom region to remove gripper/robot body in camera view.
        # 0.82 means keep top 82% and ignore bottom 18%.
        self.image_keep_ratio_y = 0.82

        # Current working axis mapping:
        # X_link = depth
        # Y_link = x_opt
        # Z_link = -y_opt
        self.use_positive_x_opt = True

        # Point cloud storage
        self.global_pcd = o3d.geometry.PointCloud()
        self.final_pcd = o3d.geometry.PointCloud()

        # Capture settings
        self.frames_per_view = 10
        self.capture_remaining = 0
        self.is_capturing = False
        self.frame_counter = 0
        self.process_every_n_frames = 2

        # Point cloud settings
        self.pixel_step = 5
        self.frame_voxel_size = 0.010
        self.final_voxel_size = 0.008

        # Workspace filter in base_link
        self.min_base_z = 0.005
        self.max_base_z = 0.400

        self.workspace_x_min = -0.10
        self.workspace_x_max = 0.85
        self.workspace_y_min = -0.50
        self.workspace_y_max = 0.50

        # Plane/object extraction settings
        self.plane_distance_threshold = 0.012
        self.object_above_plane_threshold = 0.018
        self.dbscan_eps = 0.035
        self.dbscan_min_points = 20

        # Grasp estimation
        self.grasp_z_offset = 0.035

        # UR joint names
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Main top-down pose
        self.center_pose = [
            -0.2194,
            -1.9407,
            1.8675,
            -1.4961,
            -1.5702,
            2.9206
        ]

        # Right view pose
        self.right_pose = [
            0.1177,
            -1.8458,
            1.8050,
            -1.5289,
            -1.5698,
            3.2577
        ]

        # Symmetric left pose:
        # left = 2 * center - right
        self.left_pose = [
            -0.5565,
            -2.0356,
            1.9300,
            -1.4633,
            -1.5706,
            2.5835
        ]

        self.scan_waypoints = [
            {
                "name": "center_topdown",
                "positions": self.center_pose,
                "capture": True
            },
            {
                "name": "right_view",
                "positions": self.right_pose,
                "capture": True
            },
            {
                "name": "left_view_symmetric",
                "positions": self.left_pose,
                "capture": True
            },
            {
                "name": "return_center",
                "positions": self.center_pose,
                "capture": False
            }
        ]

        self.current_waypoint_index = 0

        # Motion timing
        self.move_duration_sec = 3.0
        self.settle_time_sec = 1.0

        # State machine
        self.state = "IDLE"
        self.state_start_time = self.get_clock().now()
        self.scan_active = False
        self.scan_finished = False

        self.timer = self.create_timer(
            0.1,
            self.timer_callback
        )

        self.get_logger().info("AutoGraspScanner ready.")
        self.get_logger().info("Use:")
        self.get_logger().info(
            "ros2 topic pub --once /grasp_scan_command std_msgs/msg/String \"{data: 'auto_scan'}\""
        )
        self.get_logger().info("Other commands: reset, publish, compute")

    def camera_info_callback(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height

        raw_fx = msg.k[0]
        raw_fy = msg.k[4]
        raw_cx = msg.k[2]
        raw_cy = msg.k[5]

        self.fx = raw_fx
        self.fy = raw_fy

        if self.flip_image_horizontal:
            self.cx = (self.image_width - 1) - raw_cx
        else:
            self.cx = raw_cx

        if self.flip_image_vertical:
            self.cy = (self.image_height - 1) - raw_cy
        else:
            self.cy = raw_cy

        self.camera_info_received = True

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command == "auto_scan":
            self.start_auto_scan()

        elif command == "reset":
            self.reset_scan()

        elif command == "publish":
            if len(self.final_pcd.points) > 0:
                self.publish_pointcloud(self.final_pcd)
            else:
                self.publish_pointcloud(self.global_pcd)

        elif command == "compute":
            self.compute_grasp_from_cloud()

        else:
            self.get_logger().warn(
                f"Unknown command: {command}. Use auto_scan, reset, publish, compute."
            )

    def reset_scan(self):
        self.global_pcd = o3d.geometry.PointCloud()
        self.final_pcd = o3d.geometry.PointCloud()

        self.capture_remaining = 0
        self.is_capturing = False
        self.frame_counter = 0

        self.current_waypoint_index = 0
        self.state = "IDLE"
        self.scan_active = False
        self.scan_finished = False

        self.get_logger().info("Scan reset.")

    def start_auto_scan(self):
        if not self.camera_info_received:
            self.get_logger().warn("Cannot start: /camera_info not received yet.")
            return

        if self.scan_active:
            self.get_logger().warn("Auto scan is already active.")
            return

        self.global_pcd = o3d.geometry.PointCloud()
        self.final_pcd = o3d.geometry.PointCloud()

        self.current_waypoint_index = 0
        self.scan_active = True
        self.scan_finished = False

        self.state = "MOVING"
        self.state_start_time = self.get_clock().now()

        first_wp = self.scan_waypoints[self.current_waypoint_index]

        self.get_logger().info(
            f"Auto scan started. Moving to {first_wp['name']}."
        )

        self.publish_joint_goal(first_wp["positions"])

    def timer_callback(self):
        if self.state == "IDLE":
            # Keep publishing final cloud after scan is finished,
            # so RViz can display it even if display is added late.
            if (
                self.scan_finished
                and self.final_pcd is not None
                and len(self.final_pcd.points) > 0
            ):
                self.publish_pointcloud(self.final_pcd)

            # Keep publishing final grasp pose and marker.
            # This lets grasp_executor receive the pose even if it starts later.
            if self.scan_finished and self.last_grasp_pose is not None:
                self.last_grasp_pose.header.stamp = self.get_clock().now().to_msg()
                self.grasp_pub.publish(self.last_grasp_pose)
                self.publish_grasp_marker(self.last_grasp_pose)

            return

        now = self.get_clock().now()
        elapsed = (now - self.state_start_time).nanoseconds / 1e9

        if self.state == "MOVING":
            wait_time = self.move_duration_sec + self.settle_time_sec

            if elapsed >= wait_time:
                wp = self.scan_waypoints[self.current_waypoint_index]
                should_capture = wp.get("capture", True)

                if should_capture:
                    self.get_logger().info(
                        f"Reached {wp['name']}. Starting capture."
                    )

                    self.capture_remaining = self.frames_per_view
                    self.is_capturing = True
                    self.frame_counter = 0

                    self.state = "CAPTURING"
                    self.state_start_time = now

                else:
                    self.get_logger().info(
                        f"Reached {wp['name']}. No capture required."
                    )
                    self.finish_current_waypoint()

        elif self.state == "CAPTURING":
            if not self.is_capturing:
                wp = self.scan_waypoints[self.current_waypoint_index]
                self.get_logger().info(
                    f"Finished capture at {wp['name']}."
                )
                self.finish_current_waypoint()

        elif self.state == "COMPUTING":
            self.compute_grasp_from_cloud()
            self.state = "IDLE"
            self.scan_active = False
            self.scan_finished = True

    def finish_current_waypoint(self):
        self.current_waypoint_index += 1

        if self.current_waypoint_index >= len(self.scan_waypoints):
            self.get_logger().info("All waypoints completed. Computing grasp.")
            self.state = "COMPUTING"
            self.state_start_time = self.get_clock().now()
            return

        next_wp = self.scan_waypoints[self.current_waypoint_index]

        self.get_logger().info(
            f"Moving to next waypoint: {next_wp['name']}"
        )

        self.publish_joint_goal(next_wp["positions"])

        self.state = "MOVING"
        self.state_start_time = self.get_clock().now()

    def image_callback(self, msg):
        if not self.is_capturing:
            return

        if not self.camera_info_received:
            return

        self.frame_counter += 1

        if self.frame_counter % self.process_every_n_frames != 0:
            return

        if self.capture_remaining <= 0:
            self.is_capturing = False
            return

        pcd = self.create_pcd_from_image(msg)

        if pcd is None or len(pcd.points) == 0:
            self.get_logger().warn("No valid point cloud from this frame.")
            return

        pcd = pcd.voxel_down_sample(voxel_size=self.frame_voxel_size)

        self.global_pcd += pcd

        self.capture_remaining -= 1
        captured = self.frames_per_view - self.capture_remaining

        self.get_logger().info(
            f"Captured frame {captured}/{self.frames_per_view}, "
            f"current global points: {len(self.global_pcd.points)}"
        )

        if captured % 3 == 0:
            self.publish_pointcloud(self.global_pcd)

        if self.capture_remaining <= 0:
            self.is_capturing = False

    def create_pcd_from_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return None

        if self.flip_image_vertical:
            cv_image = cv2.flip(cv_image, 0)

        if self.flip_image_horizontal:
            cv_image = cv2.flip(cv_image, 1)

        pil_img = PILImage.fromarray(cv_image)

        try:
            with torch.no_grad():
                depth_map = self.model.infer_pil(pil_img)

            depth_map = np.asarray(depth_map).astype(np.float32)

        except Exception as e:
            self.get_logger().error(f"ZoeDepth failed: {e}")
            return None

        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(
                f"Waiting for TF {self.target_frame} <- {self.camera_frame}: {e}"
            )
            return None

        depth_map = self.scale_depth_to_camera_height(depth_map, trans)

        points_base, colors = self.depth_rgb_to_base_points(
            depth_map,
            cv_image,
            trans
        )

        if points_base is None or len(points_base) == 0:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_base.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd

    def scale_depth_to_camera_height(self, depth_map, trans):
        q = trans.transform.rotation
        t = trans.transform.translation

        rot_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        # In current mapping, camera_link X is depth direction.
        cam_depth_dir_in_base = rot_mat[:, 0]
        cos_theta = abs(cam_depth_dir_in_base[2])

        center_depth_pred = depth_map[
            depth_map.shape[0] // 2,
            depth_map.shape[1] // 2
        ]

        if cos_theta > 0.1 and center_depth_pred > 0.01:
            true_depth = abs(t.z) / cos_theta
            scale = true_depth / center_depth_pred
        else:
            scale = 1.0

        return depth_map * scale

    def depth_rgb_to_base_points(self, depth_map, cv_image, trans):
        h, w = depth_map.shape
        step = self.pixel_step

        uu, vv = np.meshgrid(
            np.arange(0, w, step),
            np.arange(0, h, step)
        )

        u = uu.flatten()
        v = vv.flatten()

        z_depth = depth_map[v, u]

        # Remove lower part of image where gripper/robot body appears.
        v_limit = int(h * self.image_keep_ratio_y)

        valid = (
            np.isfinite(z_depth)
            & (z_depth > 0.02)
            & (v < v_limit)
        )

        u = u[valid]
        v = v[valid]
        z_depth = z_depth[valid]

        if len(z_depth) == 0:
            return None, None

        x_opt = (u - self.cx) * z_depth / self.fx
        y_opt = (v - self.cy) * z_depth / self.fy

        if self.use_positive_x_opt:
            y_link = x_opt
        else:
            y_link = -x_opt

        # Current working mapping:
        # camera_link X = depth
        # camera_link Y = image x direction
        # camera_link Z = negative image y direction
        points_camera_link = np.stack(
            (
                z_depth,
                y_link,
                -y_opt
            ),
            axis=-1
        )

        q = trans.transform.rotation
        t = trans.transform.translation

        rot_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        trans_vec = np.array([t.x, t.y, t.z])

        points_base = (rot_mat @ points_camera_link.T).T + trans_vec

        colors = cv_image[v, u, :].astype(np.float32) / 255.0

        valid_base = (
            np.isfinite(points_base[:, 0])
            & np.isfinite(points_base[:, 1])
            & np.isfinite(points_base[:, 2])
            & (points_base[:, 2] > self.min_base_z)
            & (points_base[:, 2] < self.max_base_z)
        )

        if self.workspace_x_min is not None:
            valid_base = valid_base & (points_base[:, 0] > self.workspace_x_min)

        if self.workspace_x_max is not None:
            valid_base = valid_base & (points_base[:, 0] < self.workspace_x_max)

        if self.workspace_y_min is not None:
            valid_base = valid_base & (points_base[:, 1] > self.workspace_y_min)

        if self.workspace_y_max is not None:
            valid_base = valid_base & (points_base[:, 1] < self.workspace_y_max)

        points_base = points_base[valid_base]
        colors = colors[valid_base]

        return points_base, colors

    def remove_dominant_plane_and_cluster(self, pcd):
        if pcd is None or len(pcd.points) < 50:
            self.get_logger().warn("Input point cloud too small for plane removal.")
            return None

        # First remove sparse noise.
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=1.5
        )

        if len(pcd.points) < 50:
            self.get_logger().warn("Point cloud too small after statistical outlier removal.")
            return None

        # Estimate dominant plane. Usually this is table/floor.
        try:
            plane_model, plane_inliers = pcd.segment_plane(
                distance_threshold=self.plane_distance_threshold,
                ransac_n=3,
                num_iterations=800
            )
        except Exception as e:
            self.get_logger().warn(f"Plane segmentation failed: {e}")
            return None

        a, b, c, d = plane_model
        points = np.asarray(pcd.points)

        has_colors = len(pcd.colors) == len(pcd.points)
        if has_colors:
            colors = np.asarray(pcd.colors)
        else:
            colors = np.ones_like(points)

        # Signed distance to plane.
        dist = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
        norm = np.sqrt(a * a + b * b + c * c)

        if norm < 1e-6:
            self.get_logger().warn("Invalid plane normal.")
            return None

        dist = dist / norm

        # Make normal roughly point upward in base_link.
        if c < 0:
            dist = -dist

        # Keep points above the plane.
        above_mask = dist > self.object_above_plane_threshold

        object_points = points[above_mask]
        object_colors = colors[above_mask]

        if len(object_points) < 30:
            self.get_logger().warn(
                f"Not enough points above plane: {len(object_points)}"
            )
            return None

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_points.astype(np.float64))
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors.astype(np.float64))

        # DBSCAN clustering removes remaining fragments.
        try:
            labels = np.array(
                object_pcd.cluster_dbscan(
                    eps=self.dbscan_eps,
                    min_points=self.dbscan_min_points,
                    print_progress=False
                )
            )
        except Exception as e:
            self.get_logger().warn(f"DBSCAN failed: {e}")
            return object_pcd

        valid_labels = labels[labels >= 0]

        if len(valid_labels) == 0:
            self.get_logger().warn("DBSCAN found no valid cluster. Returning plane-filtered cloud.")
            return object_pcd

        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        best_label = unique_labels[np.argmax(counts)]

        cluster_mask = labels == best_label

        cluster_points = object_points[cluster_mask]
        cluster_colors = object_colors[cluster_mask]

        if len(cluster_points) < 30:
            self.get_logger().warn(
                f"Selected DBSCAN cluster too small: {len(cluster_points)}"
            )
            return object_pcd

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points.astype(np.float64))
        cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors.astype(np.float64))

        self.get_logger().info(
            f"Plane removed. Object cluster points: {len(cluster_points)}"
        )

        return cluster_pcd

    def compute_grasp_from_cloud(self):
        if len(self.global_pcd.points) == 0:
            self.get_logger().warn("No point cloud captured. Cannot compute grasp.")
            return

        self.get_logger().info("Cleaning fused point cloud...")

        pcd = self.global_pcd.voxel_down_sample(
            voxel_size=self.final_voxel_size
        )

        object_pcd = self.remove_dominant_plane_and_cluster(pcd)

        if object_pcd is None or len(object_pcd.points) < 30:
            self.get_logger().warn("Failed to extract clean object point cloud.")
            return

        object_points = np.asarray(object_pcd.points)

        grasp_x = float(np.median(object_points[:, 0]))
        grasp_y = float(np.median(object_points[:, 1]))
        top_z = float(np.percentile(object_points[:, 2], 95))
        grasp_z = top_z + self.grasp_z_offset

        yaw = self.estimate_yaw_from_pca(object_points)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.target_frame

        pose.pose.position.x = grasp_x
        pose.pose.position.y = grasp_y
        pose.pose.position.z = grasp_z

        quat = R.from_euler('z', yaw).as_quat()

        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        self.last_grasp_pose = pose
        self.grasp_pub.publish(pose)
        self.publish_grasp_marker(pose)

        self.get_logger().info("Published /target_grasp_pose and /target_grasp_marker")
        self.get_logger().info(
            f"Position: x={grasp_x:.3f}, y={grasp_y:.3f}, z={grasp_z:.3f}"
        )
        self.get_logger().info(
            f"Yaw: {np.degrees(yaw):.1f} deg"
        )

        # Keep cleaned object cloud for RViz continuous publishing.
        self.final_pcd = object_pcd

        save_dir = os.path.expanduser('~/robot_description/sfm_dataset/dense')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'auto_grasp_clean_object.ply')

        o3d.io.write_point_cloud(save_path, object_pcd)

        self.get_logger().info(f"Saved cleaned object cloud to: {save_path}")

        self.publish_pointcloud(self.final_pcd)

    def estimate_yaw_from_pca(self, points):
        xy = points[:, :2]
        xy = xy - np.mean(xy, axis=0)

        if len(xy) < 3:
            return 0.0

        cov = np.cov(xy.T)
        eig_vals, eig_vecs = np.linalg.eig(cov)

        main_axis = eig_vecs[:, np.argmax(eig_vals)]
        yaw = np.arctan2(main_axis[1], main_axis[0])

        return float(yaw)

    def publish_grasp_marker(self, pose):
        marker = Marker()
        marker.header = pose.header
        marker.ns = "grasp_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose = pose.pose

        # Arrow size
        marker.scale.x = 0.12
        marker.scale.y = 0.025
        marker.scale.z = 0.025

        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.1
        marker.color.a = 1.0

        marker.lifetime.sec = 0

        self.marker_pub.publish(marker)

    def publish_pointcloud(self, pcd):
        if pcd is None or len(pcd.points) == 0:
            return

        points = np.asarray(pcd.points, dtype=np.float32)

        if len(pcd.colors) == len(pcd.points):
            colors = np.asarray(pcd.colors, dtype=np.float32)
        else:
            colors = np.ones_like(points, dtype=np.float32)

        rgba = np.zeros((colors.shape[0], 4), dtype=np.uint8)
        rgba[:, 0] = np.clip(colors[:, 0] * 255.0, 0, 255).astype(np.uint8)
        rgba[:, 1] = np.clip(colors[:, 1] * 255.0, 0, 255).astype(np.uint8)
        rgba[:, 2] = np.clip(colors[:, 2] * 255.0, 0, 255).astype(np.uint8)
        rgba[:, 3] = 255

        rgb_float = rgba.view(np.float32).reshape(-1, 1)

        pc_data = np.hstack((points, rgb_float))

        fields = [
            PointField(
                name='x',
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='y',
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='z',
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='rgb',
                offset=12,
                datatype=PointField.FLOAT32,
                count=1
            ),
        ]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.target_frame

        msg = pc2.create_cloud(header, fields, pc_data)
        self.pc_pub.publish(msg)

    def publish_joint_goal(self, positions):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.time_from_start = Duration(
            seconds=self.move_duration_sec
        ).to_msg()

        msg.points.append(point)

        self.traj_pub.publish(msg)

        self.get_logger().info(
            "Published joint trajectory goal: "
            + ", ".join([f"{x:.3f}" for x in positions])
        )


def main(args=None):
    rclpy.init(args=args)

    node = AutoGraspScanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()