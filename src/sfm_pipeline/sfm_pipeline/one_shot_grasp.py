import os

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

from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import sensor_msgs_py.point_cloud2 as pc2

from sfm_pipeline.depth_models.factory import create_depth_model
from sfm_pipeline.grasp_models.factory import create_grasp_model
from sfm_pipeline.grasp_models.base import GraspContext


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
        self.last_rgb_image = None
        self.last_depth_map = None
        self.last_camera_transform = None

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

        # -----------------------------
        # ROS parameters for model selection
        # -----------------------------
        self.declare_parameter("depth_model", "zoedepth")
        self.declare_parameter("grasp_model", "pca")

        # Depth Anything V2 parameters
        self.declare_parameter("depth_anything_checkpoint", "")
        self.declare_parameter("depth_anything_encoder", "vitb")
        self.declare_parameter("depth_anything_max_depth", 20.0)

        # Depth Pro parameters
        self.declare_parameter("depth_pro_input_size", 384)
        self.declare_parameter("depth_pro_force_cpu", False)

        # Image-space vertical crop before point cloud generation
        self.declare_parameter("image_ignore_top_ratio_y", 0.12)
        self.declare_parameter("image_keep_ratio_y", 0.82)

        # Debug point cloud saving
        self.declare_parameter("save_debug_pcd", True)
        self.declare_parameter(
            "debug_pcd_dir",
            os.path.expanduser("~/robot_description/sfm_dataset/debug_pcd")
        )

        # Point cloud sampling parameters
        self.declare_parameter("pixel_step", 4)
        self.declare_parameter("frame_voxel_size", 0.010)
        self.declare_parameter("final_voxel_size", 0.006)

        # Plane removal parameters
        self.declare_parameter("enable_plane_removal", False)
        self.declare_parameter("plane_distance_threshold", 0.020)
        self.declare_parameter("object_above_plane_threshold", 0.006)
        self.declare_parameter("plane_min_inlier_ratio", 0.18)
        self.declare_parameter("plane_min_normal_z", 0.75)

        # Sparse noise filtering parameters
        self.declare_parameter("enable_radius_outlier_removal", True)
        self.declare_parameter("radius_outlier_nb_points", 3)
        self.declare_parameter("radius_outlier_radius", 0.045)

        self.declare_parameter("enable_statistical_outlier_removal", True)
        self.declare_parameter("outlier_nb_neighbors", 12)
        self.declare_parameter("outlier_std_ratio", 3.0)

        # DBSCAN parameters
        self.declare_parameter("dbscan_eps", 0.065)
        self.declare_parameter("dbscan_min_points", 8)

        # Cluster merging parameters
        self.declare_parameter("keep_nearby_clusters", True)
        self.declare_parameter("nearby_cluster_xy_radius", 0.140)
        self.declare_parameter("nearby_cluster_z_radius", 0.100)
        self.declare_parameter("min_cluster_size_to_keep", 15)

        self.depth_model_name = self.get_parameter("depth_model").value
        self.grasp_model_name = self.get_parameter("grasp_model").value

        self.depth_anything_checkpoint = self.get_parameter(
            "depth_anything_checkpoint"
        ).value

        self.depth_anything_encoder = self.get_parameter(
            "depth_anything_encoder"
        ).value

        self.depth_anything_max_depth = float(
            self.get_parameter("depth_anything_max_depth").value
        )

        self.depth_pro_input_size = int(
            self.get_parameter("depth_pro_input_size").value
        )

        self.depth_pro_force_cpu = bool(
            self.get_parameter("depth_pro_force_cpu").value
        )

        self.image_ignore_top_ratio_y = float(
            self.get_parameter("image_ignore_top_ratio_y").value
        )

        self.image_keep_ratio_y = float(
            self.get_parameter("image_keep_ratio_y").value
        )

        self.save_debug_pcd = bool(
            self.get_parameter("save_debug_pcd").value
        )

        self.debug_pcd_dir = self.get_parameter("debug_pcd_dir").value

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------------
        # Frames
        # -----------------------------
        self.target_frame = "base_link"
        self.camera_frame = "camera_link"

        # -----------------------------
        # Camera intrinsics
        # -----------------------------
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.image_width = 0
        self.image_height = 0
        self.camera_info_received = False

        # -----------------------------
        # Image correction
        # -----------------------------
        self.flip_image_vertical = True
        self.flip_image_horizontal = False

        # Current working axis mapping:
        # X_link = depth
        # Y_link = x_opt
        # Z_link = -y_opt
        self.use_positive_x_opt = True

        # -----------------------------
        # Point cloud storage
        # -----------------------------
        self.global_pcd = o3d.geometry.PointCloud()
        self.final_pcd = o3d.geometry.PointCloud()

        # -----------------------------
        # Capture settings
        # -----------------------------
        self.frames_per_view = 10
        self.capture_remaining = 0
        self.is_capturing = False
        self.frame_counter = 0
        self.process_every_n_frames = 2

        # -----------------------------
        # Point cloud settings
        # -----------------------------
        self.pixel_step = int(self.get_parameter("pixel_step").value)
        self.frame_voxel_size = float(
            self.get_parameter("frame_voxel_size").value
        )
        self.final_voxel_size = float(
            self.get_parameter("final_voxel_size").value
        )

        # -----------------------------
        # Workspace filter in base_link
        # These are intentionally kept broad.
        # -----------------------------
        self.min_base_z = 0.005
        self.max_base_z = 0.400

        self.workspace_x_min = -0.10
        self.workspace_x_max = 0.85
        self.workspace_y_min = -0.50
        self.workspace_y_max = 0.50

        # -----------------------------
        # Cleaning settings
        # -----------------------------
        self.enable_plane_removal = bool(
            self.get_parameter("enable_plane_removal").value
        )
        self.plane_distance_threshold = float(
            self.get_parameter("plane_distance_threshold").value
        )
        self.object_above_plane_threshold = float(
            self.get_parameter("object_above_plane_threshold").value
        )
        self.plane_min_inlier_ratio = float(
            self.get_parameter("plane_min_inlier_ratio").value
        )
        self.plane_min_normal_z = float(
            self.get_parameter("plane_min_normal_z").value
        )

        self.enable_radius_outlier_removal = bool(
            self.get_parameter("enable_radius_outlier_removal").value
        )
        self.radius_outlier_nb_points = int(
            self.get_parameter("radius_outlier_nb_points").value
        )
        self.radius_outlier_radius = float(
            self.get_parameter("radius_outlier_radius").value
        )

        self.enable_statistical_outlier_removal = bool(
            self.get_parameter("enable_statistical_outlier_removal").value
        )
        self.outlier_nb_neighbors = int(
            self.get_parameter("outlier_nb_neighbors").value
        )
        self.outlier_std_ratio = float(
            self.get_parameter("outlier_std_ratio").value
        )

        self.dbscan_eps = float(
            self.get_parameter("dbscan_eps").value
        )
        self.dbscan_min_points = int(
            self.get_parameter("dbscan_min_points").value
        )

        self.keep_nearby_clusters = bool(
            self.get_parameter("keep_nearby_clusters").value
        )
        self.nearby_cluster_xy_radius = float(
            self.get_parameter("nearby_cluster_xy_radius").value
        )
        self.nearby_cluster_z_radius = float(
            self.get_parameter("nearby_cluster_z_radius").value
        )
        self.min_cluster_size_to_keep = int(
            self.get_parameter("min_cluster_size_to_keep").value
        )

        # -----------------------------
        # Grasp estimation
        # -----------------------------
        self.grasp_z_offset = 0.035

        # -----------------------------
        # Create selected models
        # -----------------------------
        self.get_logger().info(
            f"Selected depth model: {self.depth_model_name}"
        )
        self.get_logger().info(
            f"Selected grasp model: {self.grasp_model_name}"
        )

        self.get_logger().info(
            f"Depth Pro input size: {self.depth_pro_input_size}"
        )
        self.get_logger().info(
            f"Depth Pro force CPU: {self.depth_pro_force_cpu}"
        )

        self.get_logger().info(
            f"Image crop ratio: top_ignore={self.image_ignore_top_ratio_y}, "
            f"bottom_keep={self.image_keep_ratio_y}"
        )

        self.get_logger().info(
            f"Debug PCD saving: {self.save_debug_pcd}"
        )
        self.get_logger().info(
            f"Debug PCD directory: {self.debug_pcd_dir}"
        )

        self.get_logger().info(
            "Cleaning parameters: "
            f"pixel_step={self.pixel_step}, "
            f"final_voxel={self.final_voxel_size}, "
            f"plane_removal={self.enable_plane_removal}, "
            f"plane_dist={self.plane_distance_threshold}, "
            f"above_plane={self.object_above_plane_threshold}, "
            f"plane_min_inlier_ratio={self.plane_min_inlier_ratio}, "
            f"plane_min_normal_z={self.plane_min_normal_z}, "
            f"radius_filter={self.enable_radius_outlier_removal}, "
            f"radius_nb={self.radius_outlier_nb_points}, "
            f"radius={self.radius_outlier_radius}, "
            f"stat_filter={self.enable_statistical_outlier_removal}, "
            f"stat_neighbors={self.outlier_nb_neighbors}, "
            f"stat_std={self.outlier_std_ratio}, "
            f"dbscan_eps={self.dbscan_eps}, "
            f"dbscan_min_points={self.dbscan_min_points}, "
            f"keep_nearby_clusters={self.keep_nearby_clusters}"
        )

        self.depth_model = create_depth_model(
            model_name=self.depth_model_name,
            device=self.device,
            logger=self.get_logger(),
            checkpoint_path=self.depth_anything_checkpoint,
            encoder=self.depth_anything_encoder,
            max_depth=self.depth_anything_max_depth,
            depth_pro_input_size=self.depth_pro_input_size,
            depth_pro_force_cpu=self.depth_pro_force_cpu,
        )

        self.grasp_model = create_grasp_model(
            model_name=self.grasp_model_name,
            logger=self.get_logger(),
            z_offset=self.grasp_z_offset
        )

        # -----------------------------
        # UR joint names
        # -----------------------------
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

        # -----------------------------
        # Motion timing
        # -----------------------------
        self.move_duration_sec = 3.0
        self.settle_time_sec = 1.0

        # -----------------------------
        # State machine
        # -----------------------------
        self.state = "IDLE"
        self.state_start_time = self.get_clock().now()
        self.scan_active = False
        self.scan_finished = False

        self.timer = self.create_timer(
            0.1,
            self.timer_callback
        )

        self.get_logger().info("AutoGraspScanner ready.")
        self.get_logger().info(
            "Use: ros2 topic pub --once /grasp_scan_command "
            "std_msgs/msg/String \"{data: 'auto_scan'}\""
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

        self.last_grasp_pose = None
        self.last_rgb_image = None
        self.last_depth_map = None
        self.last_camera_transform = None

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

        self.last_grasp_pose = None
        self.last_rgb_image = None
        self.last_depth_map = None
        self.last_camera_transform = None

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
            if (
                self.scan_finished
                and self.final_pcd is not None
                and len(self.final_pcd.points) > 0
            ):
                self.publish_pointcloud(self.final_pcd)

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

            self.capture_remaining -= 1
            failed = self.frames_per_view - self.capture_remaining

            self.get_logger().warn(
                f"Failed capture frame {failed}/{self.frames_per_view}"
            )

            if self.capture_remaining <= 0:
                self.is_capturing = False

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

        try:
            depth_map = self.depth_model.predict(
                cv_image,
                camera_intrinsics={
                    "fx": self.fx,
                    "fy": self.fy,
                    "cx": self.cx,
                    "cy": self.cy,
                    "width": self.image_width,
                    "height": self.image_height,
                }
            )
            depth_map = np.asarray(depth_map).astype(np.float32)

        except Exception as e:
            self.get_logger().error(
                f"Depth model [{self.depth_model_name}] failed: {e}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        self.last_rgb_image = cv_image.copy()
        self.last_depth_map = depth_map.copy()
        self.last_camera_transform = trans

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

        v_min = int(h * self.image_ignore_top_ratio_y)
        v_max = int(h * self.image_keep_ratio_y)

        valid = (
            np.isfinite(z_depth)
            & (z_depth > 0.02)
            & (v > v_min)
            & (v < v_max)
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

    def save_debug_pointcloud(self, filename, pcd):
        if not self.save_debug_pcd:
            return

        if pcd is None or len(pcd.points) == 0:
            return

        os.makedirs(self.debug_pcd_dir, exist_ok=True)

        save_path = os.path.join(self.debug_pcd_dir, filename)

        try:
            o3d.io.write_point_cloud(save_path, pcd)
            self.get_logger().info(f"Saved debug point cloud: {save_path}")
        except Exception as e:
            self.get_logger().warn(
                f"Failed to save debug point cloud {save_path}: {e}"
            )

    def copy_pcd_from_arrays(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd

    def remove_dominant_plane_and_cluster(self, pcd):
        if pcd is None or len(pcd.points) < 50:
            self.get_logger().warn("Input point cloud too small for cleaning.")
            return None

        self.get_logger().info(
            f"Cleaning input points: {len(pcd.points)}"
        )

        self.save_debug_pointcloud(
            "01_input_downsampled.ply",
            pcd
        )

        # Step 1: Remove sparse isolated points using radius outlier removal.
        if self.enable_radius_outlier_removal and len(pcd.points) >= 50:
            try:
                pcd, _ = pcd.remove_radius_outlier(
                    nb_points=self.radius_outlier_nb_points,
                    radius=self.radius_outlier_radius
                )

                self.get_logger().info(
                    f"After radius outlier removal: {len(pcd.points)} points"
                )

                self.save_debug_pointcloud(
                    "02_after_radius_outlier.ply",
                    pcd
                )

            except Exception as e:
                self.get_logger().warn(f"Radius outlier removal failed: {e}")

        if len(pcd.points) < 50:
            self.get_logger().warn(
                "Point cloud too small after radius outlier removal."
            )
            return None

        # Step 2: Apply mild statistical outlier removal.
        if self.enable_statistical_outlier_removal and len(pcd.points) >= 50:
            try:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=self.outlier_nb_neighbors,
                    std_ratio=self.outlier_std_ratio
                )

                self.get_logger().info(
                    f"After statistical outlier removal: {len(pcd.points)} points"
                )

                self.save_debug_pointcloud(
                    "03_after_statistical_outlier.ply",
                    pcd
                )

            except Exception as e:
                self.get_logger().warn(f"Statistical outlier removal failed: {e}")

        if len(pcd.points) < 50:
            self.get_logger().warn(
                "Point cloud too small after statistical outlier removal."
            )
            return None

        # Step 3: Optional plane removal.
        if not self.enable_plane_removal:
            self.get_logger().info(
                "Plane removal disabled. Using filtered point cloud as object candidate."
            )

            object_pcd = pcd
            object_points = np.asarray(object_pcd.points)

            if len(object_pcd.colors) == len(object_pcd.points):
                object_colors = np.asarray(object_pcd.colors)
            else:
                object_colors = np.ones_like(object_points)

            self.save_debug_pointcloud(
                "04_plane_removal_skipped.ply",
                object_pcd
            )

        else:
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

            if len(pcd.colors) == len(pcd.points):
                colors = np.asarray(pcd.colors)
            else:
                colors = np.ones_like(points)

            norm = np.sqrt(a * a + b * b + c * c)

            if norm < 1e-6:
                self.get_logger().warn("Invalid plane normal.")
                return None

            normal_z = abs(c / norm)
            inlier_ratio = float(len(plane_inliers)) / float(len(points))

            self.get_logger().info(
                f"Detected plane: inlier_ratio={inlier_ratio:.3f}, "
                f"normal_z={normal_z:.3f}, "
                f"inliers={len(plane_inliers)}/{len(points)}"
            )

            plane_is_reliable = (
                inlier_ratio >= self.plane_min_inlier_ratio
                and normal_z >= self.plane_min_normal_z
            )

            if not plane_is_reliable:
                self.get_logger().warn(
                    "Plane is not reliable. Skipping plane removal to avoid cutting the object."
                )

                object_pcd = pcd
                object_points = points
                object_colors = colors

                self.save_debug_pointcloud(
                    "04_plane_removal_rejected.ply",
                    object_pcd
                )

            else:
                dist = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
                dist = dist / norm

                if c < 0:
                    dist = -dist

                above_mask = dist > self.object_above_plane_threshold

                object_points = points[above_mask]
                object_colors = colors[above_mask]

                self.get_logger().info(
                    f"After plane removal: {len(object_points)} object candidate points"
                )

                if len(object_points) < 30:
                    self.get_logger().warn(
                        f"Not enough points above plane: {len(object_points)}. "
                        "Falling back to filtered point cloud."
                    )

                    object_points = points
                    object_colors = colors

                object_pcd = self.copy_pcd_from_arrays(
                    object_points,
                    object_colors
                )

                self.save_debug_pointcloud(
                    "04_after_plane_removal.ply",
                    object_pcd
                )

        if object_points is None or len(object_points) < 30:
            self.get_logger().warn(
                "Not enough object candidate points."
            )
            return None

        # Step 4: DBSCAN to remove far-away fragments while keeping nearby object fragments.
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
            self.get_logger().warn(
                "DBSCAN found no valid cluster. Returning filtered object candidate cloud."
            )
            return object_pcd

        unique_labels, counts = np.unique(valid_labels, return_counts=True)

        largest_index = int(np.argmax(counts))
        primary_label = unique_labels[largest_index]
        primary_count = counts[largest_index]

        primary_mask = labels == primary_label
        primary_points = object_points[primary_mask]
        primary_colors = object_colors[primary_mask]

        primary_center = np.median(primary_points, axis=0)

        self.get_logger().info(
            f"DBSCAN clusters: {len(unique_labels)}, "
            f"primary label: {primary_label}, "
            f"primary points: {primary_count}"
        )

        if not self.keep_nearby_clusters:
            selected_points = primary_points
            selected_colors = primary_colors

        else:
            keep_mask = np.zeros_like(labels, dtype=bool)
            kept_labels = []

            for label, count in zip(unique_labels, counts):
                if count < self.min_cluster_size_to_keep:
                    continue

                cluster_mask = labels == label
                cluster_points = object_points[cluster_mask]

                if len(cluster_points) == 0:
                    continue

                cluster_center = np.median(cluster_points, axis=0)

                xy_dist = np.linalg.norm(
                    cluster_center[:2] - primary_center[:2]
                )
                z_dist = abs(cluster_center[2] - primary_center[2])

                if (
                    label == primary_label
                    or (
                        xy_dist <= self.nearby_cluster_xy_radius
                        and z_dist <= self.nearby_cluster_z_radius
                    )
                ):
                    keep_mask = keep_mask | cluster_mask
                    kept_labels.append(int(label))

            selected_points = object_points[keep_mask]
            selected_colors = object_colors[keep_mask]

            self.get_logger().info(
                f"Kept nearby cluster labels: {kept_labels}, "
                f"selected points: {len(selected_points)}"
            )

            if len(selected_points) < 30:
                self.get_logger().warn(
                    "Nearby cluster selection produced too few points. "
                    "Falling back to primary cluster."
                )
                selected_points = primary_points
                selected_colors = primary_colors

        if len(selected_points) < 30:
            self.get_logger().warn(
                f"Selected object cloud too small: {len(selected_points)}"
            )
            return object_pcd

        selected_pcd = self.copy_pcd_from_arrays(
            selected_points,
            selected_colors
        )

        self.save_debug_pointcloud(
            "05_after_dbscan_selected_clusters.ply",
            selected_pcd
        )

        self.get_logger().info(
            f"Final selected object points: {len(selected_points)}"
        )

        return selected_pcd

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

        context = GraspContext(
            rgb_image=self.last_rgb_image,
            depth_map=self.last_depth_map,
            object_pcd=object_pcd,
            camera_intrinsics={
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "width": self.image_width,
                "height": self.image_height,
            },
            transform_base_from_camera=self.last_camera_transform,
            frame_id=self.target_frame,
            stamp=self.get_clock().now().to_msg(),
        )

        try:
            prediction = self.grasp_model.predict(context)
        except Exception as e:
            self.get_logger().error(
                f"Grasp model [{self.grasp_model_name}] failed: {e}"
            )
            return

        pose = prediction.pose

        self.last_grasp_pose = pose
        self.grasp_pub.publish(pose)
        self.publish_grasp_marker(pose)

        self.get_logger().info("Published /target_grasp_pose and /target_grasp_marker")
        self.get_logger().info(
            f"Position: x={pose.pose.position.x:.3f}, "
            f"y={pose.pose.position.y:.3f}, "
            f"z={pose.pose.position.z:.3f}"
        )

        if "yaw_deg" in prediction.debug:
            self.get_logger().info(
                f"Yaw: {prediction.debug['yaw_deg']:.1f} deg"
            )

        if prediction.width is not None:
            self.get_logger().info(
                f"Predicted gripper width: {prediction.width:.3f} m"
            )

        self.get_logger().info(
            f"Grasp score: {prediction.score:.3f}"
        )

        self.final_pcd = object_pcd

        save_dir = os.path.expanduser('~/robot_description/sfm_dataset/dense')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'auto_grasp_clean_object.ply')

        o3d.io.write_point_cloud(save_path, object_pcd)

        self.get_logger().info(f"Saved cleaned object cloud to: {save_path}")

        self.publish_pointcloud(self.final_pcd)

    def publish_grasp_marker(self, pose):
        marker = Marker()
        marker.header = pose.header
        marker.ns = "grasp_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose = pose.pose

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