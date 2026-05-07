import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time

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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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

        self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)
        self.command_sub = self.create_subscription(String, '/grasp_scan_command', self.command_callback, 10)

        self.traj_pub = self.create_publisher(JointTrajectory, '/ur_manipulator_controller/joint_trajectory', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/ai_scanned_pointcloud', 10)
        self.gt_pc_pub = self.create_publisher(PointCloud2, '/gt_object_pointcloud', 10)
        self.grasp_pub = self.create_publisher(PoseStamped, '/target_grasp_pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/target_grasp_marker', 10)

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

        # Image-space crop before point cloud generation.
        self.declare_parameter("image_ignore_top_ratio_y", 0.12)
        self.declare_parameter("image_keep_ratio_y", 0.82)

        # Depth scaling.
        # This avoids the old centre-pixel scaling problem where the centre pixel can hit the object
        # rather than the table. The table plane is an environment calibration, not an object ROI.
        self.declare_parameter("use_table_depth_scale", True)
        self.declare_parameter("table_z", 0.0)
        self.declare_parameter("table_scale_v_min_ratio", 0.58)
        self.declare_parameter("table_scale_v_max_ratio", 0.92)
        self.declare_parameter("table_scale_step", 8)
        self.declare_parameter("min_table_scale_samples", 40)

        # Fused-cloud z alignment using known table height.
        # This corrects residual vertical offset after monocular depth scaling.
        self.declare_parameter("enable_fused_table_z_alignment", True)
        self.declare_parameter("table_alignment_percentile", 2.0)
        # Safety update: the old percentile method could lift the whole cloud if
        # low streak/noise points existed below the table. The new default still
        # allows table alignment, but only when a reliable table-like plane is found.
        self.declare_parameter("table_alignment_use_plane", True)
        self.declare_parameter("max_table_z_correction", 0.030)
        self.declare_parameter("reject_large_table_z_correction", True)
        self.declare_parameter("max_table_plane_z_error", 0.060)

        # Non-hardcoded table-height foreground filtering.
        # This removes table/background points by height above the known table plane,
        # instead of using an x/y ROI around the object.
        self.declare_parameter("enable_table_height_filter", True)
        self.declare_parameter("min_object_height_above_table", 0.012)
        self.declare_parameter("max_object_height_above_table", 0.300)

        # Optional depth percentile trimming. This uses the monocular depth heatmap distribution
        # to remove extreme depth outliers before 3D projection. It is intentionally broad.
        self.declare_parameter("enable_depth_percentile_filter", True)
        self.declare_parameter("depth_percentile_min", 1.0)
        self.declare_parameter("depth_percentile_max", 99.0)

        # Debug point cloud saving
        self.declare_parameter("save_debug_pcd", True)
        self.declare_parameter(
            "debug_pcd_dir",
            os.path.expanduser("~/robot_description/sfm_dataset/debug_pcd")
        )

        # Evaluation output saving
        self.declare_parameter("save_eval_outputs", True)
        self.declare_parameter(
            "eval_output_dir",
            os.path.expanduser("~/robot_description/sfm_dataset/eval_results")
        )
        self.declare_parameter(
            "ground_truth_csv",
            os.path.expanduser("~/robot_description/sfm_dataset/ground_truth_objects.csv")
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
        self.declare_parameter("dbscan_eps", 0.055)
        self.declare_parameter("dbscan_min_points", 10)

        # Cluster merging parameters
        self.declare_parameter("keep_nearby_clusters", True)
        self.declare_parameter("nearby_cluster_xy_radius", 0.140)
        self.declare_parameter("nearby_cluster_z_radius", 0.100)
        self.declare_parameter("min_cluster_size_to_keep", 15)

        # Cluster scoring. This avoids blindly selecting the largest cluster.
        # The selected cluster is the one that is both sufficiently large and elevated above the table.
        self.declare_parameter("cluster_height_weight", 6.0)
        self.declare_parameter("cluster_compactness_weight", 1.5)

        self.depth_model_name = self.get_parameter("depth_model").value
        self.grasp_model_name = self.get_parameter("grasp_model").value

        self.depth_anything_checkpoint = self.get_parameter("depth_anything_checkpoint").value
        self.depth_anything_encoder = self.get_parameter("depth_anything_encoder").value
        self.depth_anything_max_depth = float(self.get_parameter("depth_anything_max_depth").value)

        self.depth_pro_input_size = int(self.get_parameter("depth_pro_input_size").value)
        self.depth_pro_force_cpu = bool(self.get_parameter("depth_pro_force_cpu").value)

        self.image_ignore_top_ratio_y = float(self.get_parameter("image_ignore_top_ratio_y").value)
        self.image_keep_ratio_y = float(self.get_parameter("image_keep_ratio_y").value)

        self.use_table_depth_scale = bool(self.get_parameter("use_table_depth_scale").value)
        self.table_z = float(self.get_parameter("table_z").value)
        self.table_scale_v_min_ratio = float(self.get_parameter("table_scale_v_min_ratio").value)
        self.table_scale_v_max_ratio = float(self.get_parameter("table_scale_v_max_ratio").value)
        self.table_scale_step = int(self.get_parameter("table_scale_step").value)
        self.min_table_scale_samples = int(self.get_parameter("min_table_scale_samples").value)

        self.enable_fused_table_z_alignment = bool(
            self.get_parameter("enable_fused_table_z_alignment").value
        )
        self.table_alignment_percentile = float(
            self.get_parameter("table_alignment_percentile").value
        )
        self.table_alignment_use_plane = bool(
            self.get_parameter("table_alignment_use_plane").value
        )
        self.max_table_z_correction = float(
            self.get_parameter("max_table_z_correction").value
        )
        self.reject_large_table_z_correction = bool(
            self.get_parameter("reject_large_table_z_correction").value
        )
        self.max_table_plane_z_error = float(
            self.get_parameter("max_table_plane_z_error").value
        )

        self.enable_table_height_filter = bool(self.get_parameter("enable_table_height_filter").value)
        self.min_object_height_above_table = float(self.get_parameter("min_object_height_above_table").value)
        self.max_object_height_above_table = float(self.get_parameter("max_object_height_above_table").value)

        self.enable_depth_percentile_filter = bool(self.get_parameter("enable_depth_percentile_filter").value)
        self.depth_percentile_min = float(self.get_parameter("depth_percentile_min").value)
        self.depth_percentile_max = float(self.get_parameter("depth_percentile_max").value)

        self.save_debug_pcd = bool(self.get_parameter("save_debug_pcd").value)
        self.debug_pcd_dir = self.get_parameter("debug_pcd_dir").value

        self.save_eval_outputs = bool(self.get_parameter("save_eval_outputs").value)
        self.eval_output_dir = Path(os.path.expanduser(self.get_parameter("eval_output_dir").value))
        self.ground_truth_csv = Path(os.path.expanduser(self.get_parameter("ground_truth_csv").value))

        self.current_trial_id = None
        self.current_trial_dir = None
        self.current_image_dir = None
        self.current_figure_dir = None
        self.current_pcd_dir = None
        self.results_csv_path = self.eval_output_dir / "experiment_log.csv"

        self.cleaning_metrics = {}
        self.frame_processing_times = []
        self.saved_frame_index = 0

        self.last_depth_scale = float("nan")
        self.last_depth_scale_method = "none"

        self.cluster_height_weight = float(self.get_parameter("cluster_height_weight").value)
        self.cluster_compactness_weight = float(self.get_parameter("cluster_compactness_weight").value)

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
        self.gt_pcd = o3d.geometry.PointCloud()

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
        self.frame_voxel_size = float(self.get_parameter("frame_voxel_size").value)
        self.final_voxel_size = float(self.get_parameter("final_voxel_size").value)

        # -----------------------------
        # Broad workspace filter in base_link
        # -----------------------------
        self.min_base_z = -0.050
        self.max_base_z = 0.450

        self.workspace_x_min = -0.10
        self.workspace_x_max = 0.85
        self.workspace_y_min = -0.50
        self.workspace_y_max = 0.50

        # -----------------------------
        # Cleaning settings
        # -----------------------------
        self.enable_plane_removal = bool(self.get_parameter("enable_plane_removal").value)
        self.plane_distance_threshold = float(self.get_parameter("plane_distance_threshold").value)
        self.object_above_plane_threshold = float(self.get_parameter("object_above_plane_threshold").value)
        self.plane_min_inlier_ratio = float(self.get_parameter("plane_min_inlier_ratio").value)
        self.plane_min_normal_z = float(self.get_parameter("plane_min_normal_z").value)

        self.enable_radius_outlier_removal = bool(self.get_parameter("enable_radius_outlier_removal").value)
        self.radius_outlier_nb_points = int(self.get_parameter("radius_outlier_nb_points").value)
        self.radius_outlier_radius = float(self.get_parameter("radius_outlier_radius").value)

        self.enable_statistical_outlier_removal = bool(self.get_parameter("enable_statistical_outlier_removal").value)
        self.outlier_nb_neighbors = int(self.get_parameter("outlier_nb_neighbors").value)
        self.outlier_std_ratio = float(self.get_parameter("outlier_std_ratio").value)

        self.dbscan_eps = float(self.get_parameter("dbscan_eps").value)
        self.dbscan_min_points = int(self.get_parameter("dbscan_min_points").value)

        self.keep_nearby_clusters = bool(self.get_parameter("keep_nearby_clusters").value)
        self.nearby_cluster_xy_radius = float(self.get_parameter("nearby_cluster_xy_radius").value)
        self.nearby_cluster_z_radius = float(self.get_parameter("nearby_cluster_z_radius").value)
        self.min_cluster_size_to_keep = int(self.get_parameter("min_cluster_size_to_keep").value)

        # -----------------------------
        # Grasp estimation
        # -----------------------------
        self.grasp_z_offset = 0.035

        # -----------------------------
        # Create selected models
        # -----------------------------
        self.get_logger().info(f"Selected depth model: {self.depth_model_name}")
        self.get_logger().info(f"Selected grasp model: {self.grasp_model_name}")
        self.get_logger().info(f"Depth Pro input size: {self.depth_pro_input_size}")
        self.get_logger().info(f"Depth Pro force CPU: {self.depth_pro_force_cpu}")
        self.get_logger().info(
            f"Image crop ratio: top_ignore={self.image_ignore_top_ratio_y}, "
            f"bottom_keep={self.image_keep_ratio_y}"
        )
        self.get_logger().info(
            "Depth scale settings: "
            f"use_table_depth_scale={self.use_table_depth_scale}, "
            f"table_z={self.table_z}, "
            f"table_scale_band=({self.table_scale_v_min_ratio}, {self.table_scale_v_max_ratio}), "
            f"table_scale_step={self.table_scale_step}"
        )
        self.get_logger().info(
            "Fused table z alignment: "
            f"enabled={self.enable_fused_table_z_alignment}, "
            f"use_plane={self.table_alignment_use_plane}, "
            f"percentile={self.table_alignment_percentile}, "
            f"max_correction={self.max_table_z_correction}, "
            f"reject_large={self.reject_large_table_z_correction}, "
            f"max_plane_z_error={self.max_table_plane_z_error}"
        )
        self.get_logger().info(
            "Foreground height filter: "
            f"enabled={self.enable_table_height_filter}, "
            f"min_above_table={self.min_object_height_above_table}, "
            f"max_above_table={self.max_object_height_above_table}"
        )
        self.get_logger().info(
            "Depth percentile filter: "
            f"enabled={self.enable_depth_percentile_filter}, "
            f"range=({self.depth_percentile_min}, {self.depth_percentile_max})"
        )
        self.get_logger().info(f"Debug PCD saving: {self.save_debug_pcd}")
        self.get_logger().info(f"Debug PCD directory: {self.debug_pcd_dir}")
        self.get_logger().info(f"Evaluation output saving: {self.save_eval_outputs}")
        self.get_logger().info(f"Evaluation output directory: {self.eval_output_dir}")
        self.get_logger().info(f"Ground-truth CSV path: {self.ground_truth_csv}")

        self.get_logger().info(
            "Cleaning parameters: "
            f"pixel_step={self.pixel_step}, "
            f"final_voxel={self.final_voxel_size}, "
            f"plane_removal={self.enable_plane_removal}, "
            f"plane_dist={self.plane_distance_threshold}, "
            f"above_plane={self.object_above_plane_threshold}, "
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
        self.center_pose = [-0.2194, -1.9407, 1.8675, -1.4961, -1.5702, 2.9206]
        self.right_pose = [0.1177, -1.8458, 1.8050, -1.5289, -1.5698, 3.2577]
        self.left_pose = [-0.5565, -2.0356, 1.9300, -1.4633, -1.5706, 2.5835]

        self.scan_waypoints = [
            {"name": "center_topdown", "positions": self.center_pose, "capture": True},
            {"name": "right_view", "positions": self.right_pose, "capture": True},
            {"name": "left_view_symmetric", "positions": self.left_pose, "capture": True},
            {"name": "return_center", "positions": self.center_pose, "capture": False},
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

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("AutoGraspScanner ready.")
        self.get_logger().info(
            "Use: ros2 topic pub --once /grasp_scan_command "
            "std_msgs/msg/String \"{data: 'auto_scan'}\""
        )
        self.get_logger().info("Other commands: reset, publish, publish_gt, compute")

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
            self.load_and_publish_latest_gt_pointcloud()

        elif command == "compute":
            self.compute_grasp_from_cloud()

        elif command == "publish_gt":
            self.load_and_publish_latest_gt_pointcloud()

        else:
            self.get_logger().warn(
                f"Unknown command: {command}. Use auto_scan, reset, publish, publish_gt, compute."
            )

    def reset_scan(self):
        self.global_pcd = o3d.geometry.PointCloud()
        self.final_pcd = o3d.geometry.PointCloud()
        self.gt_pcd = o3d.geometry.PointCloud()

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

        self.current_trial_id = None
        self.current_trial_dir = None
        self.current_image_dir = None
        self.current_figure_dir = None
        self.current_pcd_dir = None
        self.cleaning_metrics = {}
        self.frame_processing_times = []
        self.saved_frame_index = 0

        self.last_depth_scale = float("nan")
        self.last_depth_scale_method = "none"

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
        self.gt_pcd = o3d.geometry.PointCloud()

        self.start_new_trial()

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

        self.get_logger().info(f"Auto scan started. Moving to {first_wp['name']}.")
        self.publish_joint_goal(first_wp["positions"])

    def timer_callback(self):
        if self.state == "IDLE":
            if self.scan_finished and self.final_pcd is not None and len(self.final_pcd.points) > 0:
                self.publish_pointcloud(self.final_pcd)

            if self.scan_finished and self.gt_pcd is not None and len(self.gt_pcd.points) > 0:
                self.publish_pointcloud_to_publisher(
                    pcd=self.gt_pcd,
                    publisher=self.gt_pc_pub,
                    frame_id=self.target_frame
                )

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
                    self.get_logger().info(f"Reached {wp['name']}. Starting capture.")
                    self.capture_remaining = self.frames_per_view
                    self.is_capturing = True
                    self.frame_counter = 0
                    self.state = "CAPTURING"
                    self.state_start_time = now

                else:
                    self.get_logger().info(f"Reached {wp['name']}. No capture required.")
                    self.finish_current_waypoint()

        elif self.state == "CAPTURING":
            if not self.is_capturing:
                wp = self.scan_waypoints[self.current_waypoint_index]
                self.get_logger().info(f"Finished capture at {wp['name']}.")
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
        self.get_logger().info(f"Moving to next waypoint: {next_wp['name']}")
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

        frame_start_time = time.perf_counter()
        pcd = self.create_pcd_from_image(msg)
        frame_processing_time = time.perf_counter() - frame_start_time
        self.frame_processing_times.append(frame_processing_time)

        if pcd is None or len(pcd.points) == 0:
            self.get_logger().warn("No valid point cloud from this frame.")
            self.capture_remaining -= 1
            failed = self.frames_per_view - self.capture_remaining
            self.get_logger().warn(f"Failed capture frame {failed}/{self.frames_per_view}")

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
            self.get_logger().error(f"Depth model [{self.depth_model_name}] failed: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return None

        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                Time(),
                timeout=Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f"Waiting for TF {self.target_frame} <- {self.camera_frame}: {e}")
            return None

        if self.use_table_depth_scale:
            depth_map = self.scale_depth_to_table_plane(depth_map, trans)
        else:
            depth_map = self.scale_depth_to_camera_height(depth_map, trans)

        self.last_rgb_image = cv_image.copy()
        self.last_depth_map = depth_map.copy()
        self.last_camera_transform = trans

        try:
            waypoint_name = self.scan_waypoints[self.current_waypoint_index]["name"]
        except Exception:
            waypoint_name = "unknown_view"

        self.save_rgb_depth_debug_images(
            rgb_image=cv_image,
            depth_map=depth_map,
            waypoint_name=waypoint_name
        )

        points_base, colors = self.depth_rgb_to_base_points(depth_map, cv_image, trans)

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

        center_depth_pred = depth_map[depth_map.shape[0] // 2, depth_map.shape[1] // 2]

        if cos_theta > 0.1 and center_depth_pred > 0.01:
            true_depth = abs(t.z - self.table_z) / cos_theta
            scale = true_depth / center_depth_pred
        else:
            scale = 1.0

        self.last_depth_scale = float(scale)
        self.last_depth_scale_method = "camera_height"

        return depth_map * scale

    def scale_depth_to_table_plane(self, depth_map, trans):
        """
        Scale monocular depth using the known table plane z = table_z.

        This is more stable than centre-pixel scaling because it uses many pixels
        from a lower image band that usually observes the table.
        """
        h, w = depth_map.shape

        if self.fx <= 1e-6 or self.fy <= 1e-6:
            self.get_logger().warn("Invalid camera intrinsics. Falling back to camera-height scaling.")
            return self.scale_depth_to_camera_height(depth_map, trans)

        q = trans.transform.rotation
        t = trans.transform.translation

        rot_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        cam_z = float(t.z)

        step = max(2, int(self.table_scale_step))

        v_min = int(h * self.table_scale_v_min_ratio)
        v_max = int(h * self.table_scale_v_max_ratio)
        v_min = max(0, min(h - 1, v_min))
        v_max = max(v_min + 1, min(h, v_max))

        uu, vv = np.meshgrid(
            np.arange(0, w, step),
            np.arange(v_min, v_max, step)
        )

        u = uu.flatten()
        v = vv.flatten()

        pred_depth = depth_map[v, u]

        valid = np.isfinite(pred_depth) & (pred_depth > 0.02)

        if self.enable_depth_percentile_filter:
            all_valid_depth = depth_map[np.isfinite(depth_map) & (depth_map > 0.02)]

            if len(all_valid_depth) > 50:
                d_min = np.percentile(all_valid_depth, self.depth_percentile_min)
                d_max = np.percentile(all_valid_depth, self.depth_percentile_max)
                valid = valid & (pred_depth >= d_min) & (pred_depth <= d_max)

        u = u[valid]
        v = v[valid]
        pred_depth = pred_depth[valid]

        if len(pred_depth) < self.min_table_scale_samples:
            self.get_logger().warn(
                "Not enough valid pixels for table-plane depth scaling. "
                "Falling back to camera-height scaling."
            )
            return self.scale_depth_to_camera_height(depth_map, trans)

        x_opt = (u - self.cx) / self.fx
        y_opt = (v - self.cy) / self.fy

        if self.use_positive_x_opt:
            y_link = x_opt
        else:
            y_link = -x_opt

        rays_camera_link = np.stack(
            (
                np.ones_like(x_opt),
                y_link,
                -y_opt
            ),
            axis=-1
        )

        rays_base = (rot_mat @ rays_camera_link.T).T

        denom = pred_depth * rays_base[:, 2]
        scale_values = (self.table_z - cam_z) / (denom + 1e-9)

        valid_scale = np.isfinite(scale_values) & (scale_values > 0.05) & (scale_values < 50.0)
        scale_values = scale_values[valid_scale]

        if len(scale_values) < self.min_table_scale_samples:
            self.get_logger().warn(
                "Not enough valid scale values from table-plane scaling. "
                "Falling back to camera-height scaling."
            )
            return self.scale_depth_to_camera_height(depth_map, trans)

        # Median is robust if some lower-band pixels see the object instead of the table.
        scale = float(np.median(scale_values))

        self.last_depth_scale = scale
        self.last_depth_scale_method = "table_plane"

        self.get_logger().info(
            f"Table-plane depth scale: {scale:.4f}, samples={len(scale_values)}"
        )

        return depth_map * scale

    def depth_rgb_to_base_points(self, depth_map, cv_image, trans):
        h, w = depth_map.shape
        step = self.pixel_step

        uu, vv = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))

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

        if self.enable_depth_percentile_filter:
            valid_depth_values = depth_map[np.isfinite(depth_map) & (depth_map > 0.02)]

            if len(valid_depth_values) > 50:
                d_min = np.percentile(valid_depth_values, self.depth_percentile_min)
                d_max = np.percentile(valid_depth_values, self.depth_percentile_max)
                valid = valid & (z_depth >= d_min) & (z_depth <= d_max)

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

        # Do not apply the table-height object filter at single-frame level.
        # Monocular depth can have small per-frame vertical offsets; applying the
        # height filter here can remove all points from a valid frame. The same
        # foreground filter is applied later after multi-view fusion.

        points_base = points_base[valid_base]
        colors = colors[valid_base]

        return points_base, colors

    def start_new_trial(self):
        if not self.save_eval_outputs:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_trial_id = f"trial_{timestamp}"

        self.current_trial_dir = self.eval_output_dir / self.current_trial_id
        self.current_image_dir = self.current_trial_dir / "images"
        self.current_figure_dir = self.current_trial_dir / "figures"
        self.current_pcd_dir = self.current_trial_dir / "pcd"

        self.current_image_dir.mkdir(parents=True, exist_ok=True)
        self.current_figure_dir.mkdir(parents=True, exist_ok=True)
        self.current_pcd_dir.mkdir(parents=True, exist_ok=True)

        self.cleaning_metrics = {}
        self.frame_processing_times = []
        self.saved_frame_index = 0

        self.write_trial_metadata()

        self.get_logger().info(f"Started evaluation trial: {self.current_trial_id}")

    def write_trial_metadata(self):
        if not self.save_eval_outputs or self.current_trial_dir is None:
            return

        metadata = {
            "trial_id": self.current_trial_id,
            "depth_model": self.depth_model_name,
            "grasp_model": self.grasp_model_name,
            "pixel_step": self.pixel_step,
            "frame_voxel_size": self.frame_voxel_size,
            "final_voxel_size": self.final_voxel_size,
            "frames_per_view": self.frames_per_view,
            "process_every_n_frames": self.process_every_n_frames,
            "depth_scaling": {
                "use_table_depth_scale": self.use_table_depth_scale,
                "table_z": self.table_z,
                "table_scale_v_min_ratio": self.table_scale_v_min_ratio,
                "table_scale_v_max_ratio": self.table_scale_v_max_ratio,
                "table_scale_step": self.table_scale_step,
            },
            "fused_table_z_alignment": {
                "enabled": self.enable_fused_table_z_alignment,
                "table_alignment_use_plane": self.table_alignment_use_plane,
                "table_alignment_percentile": self.table_alignment_percentile,
                "max_table_z_correction": self.max_table_z_correction,
                "reject_large_table_z_correction": self.reject_large_table_z_correction,
                "max_table_plane_z_error": self.max_table_plane_z_error,
            },
            "table_height_filter": {
                "enabled": self.enable_table_height_filter,
                "min_object_height_above_table": self.min_object_height_above_table,
                "max_object_height_above_table": self.max_object_height_above_table,
            },
            "scan_waypoints": self.scan_waypoints,
            "workspace": {
                "x_min": self.workspace_x_min,
                "x_max": self.workspace_x_max,
                "y_min": self.workspace_y_min,
                "y_max": self.workspace_y_max,
                "z_min": self.min_base_z,
                "z_max": self.max_base_z,
            },
            "cleaning": {
                "enable_plane_removal": self.enable_plane_removal,
                "plane_distance_threshold": self.plane_distance_threshold,
                "object_above_plane_threshold": self.object_above_plane_threshold,
                "enable_radius_outlier_removal": self.enable_radius_outlier_removal,
                "radius_outlier_nb_points": self.radius_outlier_nb_points,
                "radius_outlier_radius": self.radius_outlier_radius,
                "enable_statistical_outlier_removal": self.enable_statistical_outlier_removal,
                "outlier_nb_neighbors": self.outlier_nb_neighbors,
                "outlier_std_ratio": self.outlier_std_ratio,
                "dbscan_eps": self.dbscan_eps,
                "dbscan_min_points": self.dbscan_min_points,
                "keep_nearby_clusters": self.keep_nearby_clusters,
            }
        }

        metadata_path = self.current_trial_dir / "trial_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def save_debug_pointcloud(self, filename, pcd):
        if pcd is None or len(pcd.points) == 0:
            return

        if self.save_debug_pcd:
            os.makedirs(self.debug_pcd_dir, exist_ok=True)
            save_path = os.path.join(self.debug_pcd_dir, filename)

            try:
                o3d.io.write_point_cloud(save_path, pcd)
                self.get_logger().info(f"Saved debug point cloud: {save_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to save debug point cloud {save_path}: {e}")

        if self.save_eval_outputs and self.current_pcd_dir is not None:
            trial_save_path = self.current_pcd_dir / filename

            try:
                o3d.io.write_point_cloud(str(trial_save_path), pcd)
            except Exception as e:
                self.get_logger().warn(f"Failed to save trial point cloud {trial_save_path}: {e}")

    def save_rgb_depth_debug_images(self, rgb_image, depth_map, waypoint_name):
        if (
            not self.save_eval_outputs
            or self.current_image_dir is None
            or rgb_image is None
            or depth_map is None
        ):
            return

        try:
            base_name = f"{waypoint_name}_{self.saved_frame_index:03d}"
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            depth_valid = depth_map[np.isfinite(depth_map)]
            if len(depth_valid) == 0:
                return

            d_min = np.percentile(depth_valid, 2)
            d_max = np.percentile(depth_valid, 98)

            depth_norm = np.clip((depth_map - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
            depth_uint8 = (depth_norm * 255.0).astype(np.uint8)

            depth_heatmap = cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_TURBO)

            rgb_path = self.current_image_dir / f"{base_name}_rgb.png"
            depth_path = self.current_image_dir / f"{base_name}_depth_heatmap.png"
            combined_path = self.current_image_dir / f"{base_name}_rgb_depth.png"

            cv2.imwrite(str(rgb_path), rgb_bgr)
            cv2.imwrite(str(depth_path), depth_heatmap)

            combined = np.concatenate((rgb_bgr, depth_heatmap), axis=1)

            cv2.putText(
                combined,
                "RGB input",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                combined,
                f"Depth map ({self.last_depth_scale_method}, scale={self.last_depth_scale:.3f})",
                (rgb_bgr.shape[1] + 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imwrite(str(combined_path), combined)
            self.saved_frame_index += 1

        except Exception as e:
            self.get_logger().warn(f"Failed to save RGB/depth debug images: {e}")

    def copy_pcd_from_arrays(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        return pcd

    def filter_pcd_by_table_height(self, pcd):
        if not self.enable_table_height_filter:
            return pcd

        if pcd is None or len(pcd.points) == 0:
            return pcd

        points = np.asarray(pcd.points)

        if len(pcd.colors) == len(pcd.points):
            colors = np.asarray(pcd.colors)
        else:
            colors = np.ones_like(points)

        z_min = self.table_z + self.min_object_height_above_table
        z_max = self.table_z + self.max_object_height_above_table

        mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)

        if np.count_nonzero(mask) < 30:
            self.get_logger().warn(
                "Table-height filter would leave too few points during cleaning. Keeping original cloud."
            )
            return pcd

        filtered = self.copy_pcd_from_arrays(points[mask], colors[mask])
        self.get_logger().info(f"After table-height filtering: {len(filtered.points)} points")

        return filtered

    def estimate_reliable_table_plane(self, pcd, purpose="table"):
        """
        Estimate a table-like plane and reject object-side/noise planes.

        Returns a dictionary with the oriented plane model if reliable, otherwise None.
        The plane normal is oriented so that +distance is above the table.
        """
        if pcd is None or len(pcd.points) < 50:
            return None

        try:
            plane_model, plane_inliers = pcd.segment_plane(
                distance_threshold=self.plane_distance_threshold,
                ransac_n=3,
                num_iterations=1200
            )
        except Exception as e:
            self.get_logger().warn(f"{purpose}: plane segmentation failed: {e}")
            return None

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if len(pcd.colors) == len(pcd.points) else np.ones_like(points)

        a, b, c, d = [float(x) for x in plane_model]
        norm = float(np.sqrt(a * a + b * b + c * c))

        if norm < 1e-9 or abs(c) < 1e-6:
            self.get_logger().warn(f"{purpose}: invalid or near-vertical plane normal.")
            return None

        # Orient plane normal upward. Then positive signed distance means above table.
        if c < 0.0:
            a, b, c, d = -a, -b, -c, -d

        normal_z = abs(c / norm)
        inlier_ratio = float(len(plane_inliers)) / float(len(points))

        if len(plane_inliers) > 0:
            inlier_points = points[np.asarray(plane_inliers, dtype=int)]
            plane_z_median = float(np.median(inlier_points[:, 2]))
            plane_z_p05 = float(np.percentile(inlier_points[:, 2], 5))
            plane_z_p95 = float(np.percentile(inlier_points[:, 2], 95))
        else:
            plane_z_median = float("nan")
            plane_z_p05 = float("nan")
            plane_z_p95 = float("nan")

        z_error = abs(plane_z_median - self.table_z) if np.isfinite(plane_z_median) else float("inf")

        self.get_logger().info(
            f"{purpose}: plane candidate: inlier_ratio={inlier_ratio:.3f}, "
            f"normal_z={normal_z:.3f}, z_median={plane_z_median:.4f}, "
            f"z_range=[{plane_z_p05:.4f}, {plane_z_p95:.4f}], "
            f"z_error={z_error:.4f}, inliers={len(plane_inliers)}/{len(points)}"
        )

        reliable = (
            inlier_ratio >= self.plane_min_inlier_ratio
            and normal_z >= self.plane_min_normal_z
            and z_error <= self.max_table_plane_z_error
        )

        if not reliable:
            self.get_logger().warn(
                f"{purpose}: rejected plane. "
                f"Need inlier_ratio>={self.plane_min_inlier_ratio:.3f}, "
                f"normal_z>={self.plane_min_normal_z:.3f}, "
                f"z_error<={self.max_table_plane_z_error:.3f}."
            )
            return None

        return {
            "model": (a, b, c, d),
            "norm": norm,
            "inliers": plane_inliers,
            "points": points,
            "colors": colors,
            "normal_z": normal_z,
            "inlier_ratio": inlier_ratio,
            "plane_z_median": plane_z_median,
        }

    def align_fused_cloud_to_table_z(self, pcd):
        """
        Correct residual z-offset using a reliable table plane.

        Important change:
        The previous percentile-based method could be fooled by low streak/noise points
        below the table, which lifted the whole object cloud. This version uses a
        RANSAC table plane when possible and skips large/unreliable corrections.
        """
        if not self.enable_fused_table_z_alignment:
            return pcd

        if pcd is None or len(pcd.points) < 50:
            return pcd

        correction = 0.0
        observed_table_z = float("nan")
        method = "none"

        if self.table_alignment_use_plane:
            plane_info = self.estimate_reliable_table_plane(pcd, purpose="fused table alignment")
            if plane_info is not None:
                observed_table_z = float(plane_info["plane_z_median"])
                correction = self.table_z - observed_table_z
                method = "ransac_table_plane"
            else:
                self.get_logger().warn(
                    "Fused table z alignment: no reliable table plane found. "
                    "Skipping z alignment to avoid lifting the cloud."
                )
                return pcd
        else:
            # Legacy fallback. Keep available for debugging, but the plane method is safer.
            points = np.asarray(pcd.points)
            valid_z = points[
                np.isfinite(points[:, 2])
                & (points[:, 2] > self.min_base_z)
                & (points[:, 2] < self.max_base_z),
                2
            ]

            if len(valid_z) < 50:
                self.get_logger().warn(
                    "Not enough valid z values for fused table alignment. Skipping."
                )
                return pcd

            observed_table_z = float(np.percentile(valid_z, self.table_alignment_percentile))
            correction = self.table_z - observed_table_z
            method = "legacy_percentile"

        if abs(correction) > self.max_table_z_correction:
            message = (
                f"Fused table z alignment: correction {correction:.4f} m from {method} "
                f"exceeds max_table_z_correction={self.max_table_z_correction:.4f} m."
            )

            if self.reject_large_table_z_correction:
                self.get_logger().warn(message + " Skipping correction.")
                return pcd

            self.get_logger().warn(message + " Clamping correction.")
            correction = float(
                np.clip(
                    correction,
                    -self.max_table_z_correction,
                    self.max_table_z_correction
                )
            )

        if abs(correction) < 1e-6:
            return pcd

        corrected_pcd = o3d.geometry.PointCloud(pcd)
        corrected_pcd.translate((0.0, 0.0, correction))

        self.get_logger().info(
            f"Fused table z alignment ({method}): observed_table_z={observed_table_z:.4f}, "
            f"target_table_z={self.table_z:.4f}, applied_correction={correction:.4f} m"
        )

        return corrected_pcd

    def remove_reliable_table_plane(self, pcd):
        """
        Remove a reliable table plane before table-height filtering and DBSCAN.

        This prevents RANSAC from being run after the table has already been removed,
        which was the main reason the cleaner could pick the object side/top as a
        false plane and make the selected cloud appear to rise.
        """
        if not self.enable_plane_removal:
            self.get_logger().info(
                "Plane removal disabled. Skipping geometric table-plane removal."
            )
            return pcd

        plane_info = self.estimate_reliable_table_plane(pcd, purpose="table plane removal")

        if plane_info is None:
            self.save_debug_pointcloud("02_plane_removal_rejected.ply", pcd)
            return pcd

        a, b, c, d = plane_info["model"]
        norm = plane_info["norm"]
        points = plane_info["points"]
        colors = plane_info["colors"]

        signed_dist = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm

        # Keep only points clearly above the table plane. This removes table points and
        # below-table streaks. The normal was already oriented upward.
        keep_mask = signed_dist > self.object_above_plane_threshold

        if np.count_nonzero(keep_mask) < 30:
            self.get_logger().warn(
                "Plane removal would leave too few points. Keeping original cloud."
            )
            self.save_debug_pointcloud("02_plane_removal_too_few_points.ply", pcd)
            return pcd

        filtered = self.copy_pcd_from_arrays(points[keep_mask], colors[keep_mask])

        self.get_logger().info(
            f"After reliable table-plane removal: {len(filtered.points)} points"
        )

        self.save_debug_pointcloud("02_after_plane_removal.ply", filtered)
        return filtered

    def record_cloud_stage(self, stage_name, pcd):
        count = 0 if pcd is None else len(pcd.points)
        self.cleaning_metrics[stage_name] = int(count)

    def append_csv_row(self, csv_path, row):
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(row.keys())
        file_exists = csv_path.exists()

        if file_exists:
            try:
                with open(csv_path, "r", newline="") as f:
                    existing_header = f.readline().strip().split(",")

                if existing_header != fieldnames:
                    backup_path = csv_path.with_name(
                        csv_path.stem + f"_backup_{int(time.time())}" + csv_path.suffix
                    )
                    os.rename(csv_path, backup_path)
                    self.get_logger().warn(
                        f"Experiment CSV header changed. Backed up old file to: {backup_path}"
                    )
                    file_exists = False

            except Exception as e:
                self.get_logger().warn(f"Could not check experiment CSV header: {e}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def load_latest_ground_truth(self):
        if not self.ground_truth_csv.exists():
            return None

        try:
            with open(self.ground_truth_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if len(rows) == 0:
                return None

            return rows[-1]

        except Exception as e:
            self.get_logger().warn(f"Failed to read ground-truth CSV: {e}")
            return None

    def safe_float_from_dict(self, data, key):
        try:
            return float(data[key])
        except Exception:
            return float("nan")

    def angle_error_pi(self, estimated, ground_truth):
        """
        Angular error with 180-degree symmetry.
        This is useful for PCA axes where theta and theta + pi represent
        the same physical axis.
        """
        if not np.isfinite(estimated) or not np.isfinite(ground_truth):
            return float("nan")

        diff = (estimated - ground_truth + np.pi / 2.0) % np.pi - np.pi / 2.0
        return abs(diff)

    def yaw_error_with_gripper_symmetry(self, predicted_yaw, gt_yaw):
        """
        Gripper yaw may be either parallel or perpendicular to the object yaw.
        This compares the predicted grasp yaw against both object-parallel and
        object-perpendicular directions, while still applying 180-degree symmetry.
        """
        if not np.isfinite(predicted_yaw) or not np.isfinite(gt_yaw):
            return float("nan")

        candidates = [
            gt_yaw,
            gt_yaw + np.pi / 2.0,
            gt_yaw - np.pi / 2.0,
        ]

        errors = []

        for candidate in candidates:
            diff = (predicted_yaw - candidate + np.pi / 2.0) % np.pi - np.pi / 2.0
            errors.append(abs(diff))

        return min(errors)

    def sample_points_for_metric(self, points, max_points=15000):
        """
        Deterministically sample points to keep cloud-distance computation fast.
        """
        points = np.asarray(points, dtype=np.float64)

        if len(points) <= max_points:
            return points

        rng = np.random.default_rng(0)
        indices = rng.choice(len(points), size=max_points, replace=False)

        return points[indices]

    def nearest_neighbor_stats_mm(self, source_points, target_points, max_points=15000):
        """
        Compute nearest-neighbour distances from source_points to target_points.
        Returned distances are in millimetres.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)

        if len(source_points) == 0 or len(target_points) == 0:
            return {
                "mean_mm": float("nan"),
                "median_mm": float("nan"),
                "p95_mm": float("nan"),
                "n": 0,
            }

        source_points = self.sample_points_for_metric(source_points, max_points=max_points)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        kdtree = o3d.geometry.KDTreeFlann(target_pcd)

        distances_mm = []

        for point in source_points:
            k, _, dist2 = kdtree.search_knn_vector_3d(point, 1)

            if k > 0 and len(dist2) > 0:
                distances_mm.append(np.sqrt(dist2[0]) * 1000.0)

        if len(distances_mm) == 0:
            return {
                "mean_mm": float("nan"),
                "median_mm": float("nan"),
                "p95_mm": float("nan"),
                "n": 0,
            }

        distances_mm = np.asarray(distances_mm, dtype=np.float64)

        return {
            "mean_mm": float(np.mean(distances_mm)),
            "median_mm": float(np.median(distances_mm)),
            "p95_mm": float(np.percentile(distances_mm, 95)),
            "n": int(len(distances_mm)),
        }

    def compute_cloud_to_cloud_metrics(self, estimated_pcd, gt_data):
        """
        Compare the estimated object cloud with the ground-truth object cloud.

        est->GT measures geometric alignment of reconstructed points.
        GT->est indicates coverage/completeness and can be larger because the
        monocular scan may not observe the entire object surface.
        """
        metrics = {
            "cloud_est_to_gt_mean_mm": float("nan"),
            "cloud_est_to_gt_median_mm": float("nan"),
            "cloud_est_to_gt_p95_mm": float("nan"),
            "cloud_gt_to_est_mean_mm": float("nan"),
            "cloud_gt_to_est_median_mm": float("nan"),
            "cloud_gt_to_est_p95_mm": float("nan"),
            "cloud_metric_est_points": 0,
            "cloud_metric_gt_points": 0,
        }

        if estimated_pcd is None or len(estimated_pcd.points) == 0:
            return metrics

        if gt_data is None:
            return metrics

        gt_pcd_path = gt_data.get("gt_pcd_path", "")

        if gt_pcd_path is None or gt_pcd_path == "":
            return metrics

        gt_pcd_path = os.path.expanduser(gt_pcd_path)

        if not os.path.exists(gt_pcd_path):
            self.get_logger().warn(
                f"Cannot compute cloud metric. GT point cloud not found: {gt_pcd_path}"
            )
            return metrics

        try:
            gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)

            if gt_pcd is None or len(gt_pcd.points) == 0:
                self.get_logger().warn(
                    f"Cannot compute cloud metric. GT point cloud is empty: {gt_pcd_path}"
                )
                return metrics

            estimated_points = np.asarray(estimated_pcd.points, dtype=np.float64)
            gt_points = np.asarray(gt_pcd.points, dtype=np.float64)

            metrics["cloud_metric_est_points"] = int(len(estimated_points))
            metrics["cloud_metric_gt_points"] = int(len(gt_points))

            est_to_gt = self.nearest_neighbor_stats_mm(
                source_points=estimated_points,
                target_points=gt_points,
                max_points=15000
            )

            gt_to_est = self.nearest_neighbor_stats_mm(
                source_points=gt_points,
                target_points=estimated_points,
                max_points=15000
            )

            metrics["cloud_est_to_gt_mean_mm"] = est_to_gt["mean_mm"]
            metrics["cloud_est_to_gt_median_mm"] = est_to_gt["median_mm"]
            metrics["cloud_est_to_gt_p95_mm"] = est_to_gt["p95_mm"]

            metrics["cloud_gt_to_est_mean_mm"] = gt_to_est["mean_mm"]
            metrics["cloud_gt_to_est_median_mm"] = gt_to_est["median_mm"]
            metrics["cloud_gt_to_est_p95_mm"] = gt_to_est["p95_mm"]

            self.get_logger().info(
                "Cloud distance metrics: "
                f"est->GT mean={metrics['cloud_est_to_gt_mean_mm']:.2f} mm, "
                f"median={metrics['cloud_est_to_gt_median_mm']:.2f} mm, "
                f"p95={metrics['cloud_est_to_gt_p95_mm']:.2f} mm"
            )

            return metrics

        except Exception as e:
            self.get_logger().warn(f"Failed to compute cloud-to-cloud metrics: {e}")
            return metrics

    def pose_to_yaw_rad(self, pose):
        q = pose.pose.orientation

        try:
            yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
            return float(yaw)
        except Exception:
            return float("nan")

    def plot_point_count_by_stage(self):
        if (
            not self.save_eval_outputs
            or self.current_figure_dir is None
            or plt is None
            or len(self.cleaning_metrics) == 0
        ):
            return None

        try:
            labels = list(self.cleaning_metrics.keys())
            values = [self.cleaning_metrics[k] for k in labels]

            fig_path = self.current_figure_dir / "point_count_by_stage.png"

            plt.figure(figsize=(11, 4.8))
            plt.bar(labels, values)
            plt.ylabel("Number of points")
            plt.xlabel("Point-cloud processing stage")
            plt.title("Point-cloud reduction through cleaning pipeline")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=250)
            plt.close()

            self.get_logger().info(f"Saved point-count figure: {fig_path}")
            return fig_path

        except Exception as e:
            self.get_logger().warn(f"Failed to plot point counts: {e}")
            return None

    def plot_topdown_grasp_result(self, object_pcd, pose, gt_data=None):
        if (
            not self.save_eval_outputs
            or self.current_figure_dir is None
            or plt is None
            or object_pcd is None
            or len(object_pcd.points) == 0
            or pose is None
        ):
            return None

        try:
            points = np.asarray(object_pcd.points)

            grasp_x = pose.pose.position.x
            grasp_y = pose.pose.position.y
            yaw = self.pose_to_yaw_rad(pose)

            fig_path = self.current_figure_dir / "topdown_grasp_result.png"

            plt.figure(figsize=(6.5, 6.0))
            plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.65, label="Final object cloud")
            plt.scatter([grasp_x], [grasp_y], marker="x", s=90, label="Estimated grasp position")

            if np.isfinite(yaw):
                arrow_length = 0.08
                plt.arrow(
                    grasp_x,
                    grasp_y,
                    arrow_length * np.cos(yaw),
                    arrow_length * np.sin(yaw),
                    head_width=0.015,
                    length_includes_head=True
                )

            if gt_data is not None:
                gt_x = self.safe_float_from_dict(gt_data, "bbox_center_x")
                gt_y = self.safe_float_from_dict(gt_data, "bbox_center_y")

                if np.isfinite(gt_x) and np.isfinite(gt_y):
                    plt.scatter(
                        [gt_x],
                        [gt_y],
                        marker="o",
                        s=90,
                        facecolors="none",
                        edgecolors="black",
                        label="Ground-truth object centre"
                    )

            plt.xlabel("x in base_link (m)")
            plt.ylabel("y in base_link (m)")
            plt.title("Top-down object cloud and estimated grasp pose")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=250)
            plt.close()

            self.get_logger().info(f"Saved top-down grasp figure: {fig_path}")
            return fig_path

        except Exception as e:
            self.get_logger().warn(f"Failed to plot top-down grasp result: {e}")
            return None

    def build_experiment_row(self, object_pcd, pose, prediction, timing_data):
        points = np.asarray(object_pcd.points)

        # Estimated centre is the median of the reconstructed object cloud.
        # This should be interpreted as point-cloud localisation, not exact CAD centre estimation.
        est_center = np.median(points, axis=0)

        # Use 95th percentile instead of max to reduce sensitivity to isolated high outliers.
        est_top_z = float(np.percentile(points[:, 2], 95))

        grasp_yaw_rad = self.pose_to_yaw_rad(pose)

        if np.isfinite(grasp_yaw_rad):
            grasp_yaw_deg = np.degrees(grasp_yaw_rad)
        else:
            grasp_yaw_deg = float("nan")

        gt_data = self.load_latest_ground_truth()

        gt_object_name = ""
        gt_pcd_path = ""
        gt_center_x = float("nan")
        gt_center_y = float("nan")
        gt_center_z = float("nan")
        gt_top_z = float("nan")
        gt_yaw_rad = float("nan")

        center_error_xy_mm = float("nan")
        center_error_3d_mm = float("nan")
        top_z_error_mm = float("nan")
        yaw_error_deg = float("nan")

        if gt_data is not None:
            gt_object_name = gt_data.get("object_name", "")
            gt_pcd_path = gt_data.get("gt_pcd_path", "")

            gt_center_x = self.safe_float_from_dict(gt_data, "bbox_center_x")
            gt_center_y = self.safe_float_from_dict(gt_data, "bbox_center_y")
            gt_center_z = self.safe_float_from_dict(gt_data, "bbox_center_z")
            gt_top_z = self.safe_float_from_dict(gt_data, "top_z")
            gt_yaw_rad = self.safe_float_from_dict(gt_data, "yaw_rad")

            if np.isfinite(gt_center_x) and np.isfinite(gt_center_y):
                center_error_xy_mm = float(
                    np.linalg.norm(est_center[:2] - np.array([gt_center_x, gt_center_y])) * 1000.0
                )

            if np.isfinite(gt_center_x) and np.isfinite(gt_center_y) and np.isfinite(gt_center_z):
                center_error_3d_mm = float(
                    np.linalg.norm(est_center - np.array([gt_center_x, gt_center_y, gt_center_z])) * 1000.0
                )

            if np.isfinite(gt_top_z):
                top_z_error_mm = abs(est_top_z - gt_top_z) * 1000.0

            # This considers both parallel and perpendicular gripper/object axes.
            yaw_error = self.yaw_error_with_gripper_symmetry(
                predicted_yaw=grasp_yaw_rad,
                gt_yaw=gt_yaw_rad
            )

            if np.isfinite(yaw_error):
                yaw_error_deg = np.degrees(yaw_error)

        cloud_metrics = self.compute_cloud_to_cloud_metrics(
            estimated_pcd=object_pcd,
            gt_data=gt_data
        )

        row = {
            "trial_id": self.current_trial_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),

            "depth_model": self.depth_model_name,
            "grasp_model": self.grasp_model_name,
            "depth_scale_method": self.last_depth_scale_method,
            "last_depth_scale": self.last_depth_scale,
            "use_table_depth_scale": self.use_table_depth_scale,
            "enable_fused_table_z_alignment": self.enable_fused_table_z_alignment,
            "table_alignment_use_plane": self.table_alignment_use_plane,
            "max_table_z_correction": self.max_table_z_correction,
            "reject_large_table_z_correction": self.reject_large_table_z_correction,
            "max_table_plane_z_error": self.max_table_plane_z_error,
            "enable_table_height_filter": self.enable_table_height_filter,
            "enable_plane_removal": self.enable_plane_removal,
            "plane_distance_threshold": self.plane_distance_threshold,
            "object_above_plane_threshold": self.object_above_plane_threshold,
            "plane_min_inlier_ratio": self.plane_min_inlier_ratio,
            "plane_min_normal_z": self.plane_min_normal_z,
            "pixel_step": self.pixel_step,
            "frame_voxel_size": self.frame_voxel_size,
            "final_voxel_size": self.final_voxel_size,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_points": self.dbscan_min_points,

            "gt_object_name": gt_object_name,
            "gt_pcd_path": gt_pcd_path,

            "frames_per_view": self.frames_per_view,
            "frames_processed": len(self.frame_processing_times),
            "mean_frame_processing_time_s": float(np.mean(self.frame_processing_times))
            if len(self.frame_processing_times) > 0 else float("nan"),

            "global_points_before_final_downsample": len(self.global_pcd.points),
            "final_object_points": len(object_pcd.points),

            "stage_00_global_fused": self.cleaning_metrics.get("00_global_fused", 0),
            "stage_01_input_downsampled": self.cleaning_metrics.get("01_input_downsampled", 0),
            "stage_01b_table_height_filter": self.cleaning_metrics.get("01b_after_table_height_filter", 0),
            "stage_02_radius_outlier": self.cleaning_metrics.get("02_after_radius_outlier", 0),
            "stage_03_statistical_outlier": self.cleaning_metrics.get("03_after_statistical_outlier", 0),
            "stage_04_object_candidate": self.cleaning_metrics.get("04_object_candidate", 0),
            "stage_05_dbscan_selected": self.cleaning_metrics.get("05_after_dbscan_selected", 0),

            "est_object_center_x": float(est_center[0]),
            "est_object_center_y": float(est_center[1]),
            "est_object_center_z": float(est_center[2]),
            "est_object_top_z": est_top_z,

            "grasp_x": float(pose.pose.position.x),
            "grasp_y": float(pose.pose.position.y),
            "grasp_z": float(pose.pose.position.z),
            "grasp_yaw_deg": float(grasp_yaw_deg),

            "grasp_score": float(prediction.score),
            "predicted_gripper_width_m": float(prediction.width)
            if prediction.width is not None else float("nan"),

            "gt_center_x": gt_center_x,
            "gt_center_y": gt_center_y,
            "gt_center_z": gt_center_z,
            "gt_top_z": gt_top_z,
            "gt_yaw_rad": gt_yaw_rad,

            # Existing localisation metrics. Interpret as localisation error against GT bbox centre.
            "center_error_xy_mm": center_error_xy_mm,
            "center_error_3d_mm": center_error_3d_mm,
            "top_z_error_mm": top_z_error_mm,
            "yaw_error_deg": yaw_error_deg,

            # Cloud-to-cloud metrics. est->GT is the most useful geometry alignment metric.
            "cloud_est_to_gt_mean_mm": cloud_metrics["cloud_est_to_gt_mean_mm"],
            "cloud_est_to_gt_median_mm": cloud_metrics["cloud_est_to_gt_median_mm"],
            "cloud_est_to_gt_p95_mm": cloud_metrics["cloud_est_to_gt_p95_mm"],
            "cloud_gt_to_est_mean_mm": cloud_metrics["cloud_gt_to_est_mean_mm"],
            "cloud_gt_to_est_median_mm": cloud_metrics["cloud_gt_to_est_median_mm"],
            "cloud_gt_to_est_p95_mm": cloud_metrics["cloud_gt_to_est_p95_mm"],
            "cloud_metric_est_points": cloud_metrics["cloud_metric_est_points"],
            "cloud_metric_gt_points": cloud_metrics["cloud_metric_gt_points"],

            "cleaning_time_s": timing_data.get("cleaning_time_s", float("nan")),
            "grasp_prediction_time_s": timing_data.get("grasp_prediction_time_s", float("nan")),
            "total_compute_time_s": timing_data.get("total_compute_time_s", float("nan")),
        }

        return row

    def score_cluster(self, cluster_points):
        count = len(cluster_points)

        if count == 0:
            return -1.0

        z95 = float(np.percentile(cluster_points[:, 2], 95))
        z05 = float(np.percentile(cluster_points[:, 2], 5))
        height_above_table = max(z95 - self.table_z, 0.0)

        xy_extent = np.ptp(cluster_points[:, :2], axis=0)
        xy_size = float(np.linalg.norm(xy_extent))

        size_score = np.log1p(count)
        height_score = 1.0 + self.cluster_height_weight * height_above_table
        compactness_penalty = 1.0 + self.cluster_compactness_weight * xy_size

        return float(size_score * height_score / compactness_penalty)

    def remove_dominant_plane_and_cluster(self, pcd):
        self.cleaning_metrics = {}
        self.record_cloud_stage("00_global_fused", self.global_pcd)

        if pcd is None or len(pcd.points) < 50:
            self.get_logger().warn("Input point cloud too small for cleaning.")
            return None

        self.get_logger().info(f"Cleaning input points: {len(pcd.points)}")
        self.record_cloud_stage("01_input_downsampled", pcd)
        self.save_debug_pointcloud("01_input_downsampled.ply", pcd)

        # Step 1: remove the reliable table plane first, while table points still exist.
        # This is the key fix. Running RANSAC after the height filter can make the
        # mug side/top or high noise become the false plane.
        pcd = self.remove_reliable_table_plane(pcd)
        self.record_cloud_stage("02_after_plane_removal", pcd)

        if pcd is None or len(pcd.points) < 50:
            self.get_logger().warn("Point cloud too small after plane removal.")
            return None

        # Step 2: apply table-height foreground filter after plane removal.
        pcd = self.filter_pcd_by_table_height(pcd)
        self.record_cloud_stage("03_after_table_height_filter", pcd)
        # Also keep the old key for compatibility with the summary script.
        self.record_cloud_stage("01b_after_table_height_filter", pcd)
        self.save_debug_pointcloud("03_after_table_height_filter.ply", pcd)
        self.save_debug_pointcloud("01b_after_table_height_filter.ply", pcd)

        if pcd is None or len(pcd.points) < 50:
            self.get_logger().warn("Point cloud too small after table-height filtering.")
            return None

        # Step 3: remove sparse isolated points using radius outlier removal.
        if self.enable_radius_outlier_removal and len(pcd.points) >= 50:
            try:
                pcd, _ = pcd.remove_radius_outlier(
                    nb_points=self.radius_outlier_nb_points,
                    radius=self.radius_outlier_radius
                )
                self.get_logger().info(f"After radius outlier removal: {len(pcd.points)} points")
            except Exception as e:
                self.get_logger().warn(f"Radius outlier removal failed: {e}")

        self.record_cloud_stage("04_after_radius_outlier", pcd)
        # Also keep the old key for compatibility.
        self.record_cloud_stage("02_after_radius_outlier", pcd)
        self.save_debug_pointcloud("04_after_radius_outlier.ply", pcd)
        self.save_debug_pointcloud("02_after_radius_outlier.ply", pcd)

        if pcd is None or len(pcd.points) < 50:
            self.get_logger().warn("Point cloud too small after radius outlier removal.")
            return None

        # Step 4: apply mild statistical outlier removal.
        if self.enable_statistical_outlier_removal and len(pcd.points) >= 50:
            try:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=self.outlier_nb_neighbors,
                    std_ratio=self.outlier_std_ratio
                )
                self.get_logger().info(f"After statistical outlier removal: {len(pcd.points)} points")
            except Exception as e:
                self.get_logger().warn(f"Statistical outlier removal failed: {e}")

        self.record_cloud_stage("05_after_statistical_outlier", pcd)
        # Also keep the old key for compatibility.
        self.record_cloud_stage("03_after_statistical_outlier", pcd)
        self.save_debug_pointcloud("05_after_statistical_outlier.ply", pcd)
        self.save_debug_pointcloud("03_after_statistical_outlier.ply", pcd)

        if pcd is None or len(pcd.points) < 50:
            self.get_logger().warn("Point cloud too small after statistical outlier removal.")
            return None

        object_pcd = pcd
        object_points = np.asarray(object_pcd.points)

        if len(object_pcd.colors) == len(object_pcd.points):
            object_colors = np.asarray(object_pcd.colors)
        else:
            object_colors = np.ones_like(object_points)

        self.record_cloud_stage("04_object_candidate", object_pcd)
        self.save_debug_pointcloud("06_object_candidate.ply", object_pcd)

        if object_points is None or len(object_points) < 30:
            self.get_logger().warn("Not enough object candidate points.")
            return None

        # Step 5: DBSCAN to remove fragments.
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
            self.record_cloud_stage("05_after_dbscan_selected", object_pcd)
            self.plot_point_count_by_stage()
            return object_pcd

        valid_labels = labels[labels >= 0]

        if len(valid_labels) == 0:
            self.get_logger().warn(
                "DBSCAN found no valid cluster. Returning filtered object candidate cloud."
            )
            self.record_cloud_stage("05_after_dbscan_selected", object_pcd)
            self.plot_point_count_by_stage()
            return object_pcd

        unique_labels, counts = np.unique(valid_labels, return_counts=True)

        cluster_scores = []
        for label, count in zip(unique_labels, counts):
            if count < self.min_cluster_size_to_keep:
                cluster_scores.append(-1.0)
                continue

            cluster_points = object_points[labels == label]
            cluster_scores.append(self.score_cluster(cluster_points))

        best_index = int(np.argmax(cluster_scores))
        primary_label = unique_labels[best_index]
        primary_count = counts[best_index]
        primary_score = cluster_scores[best_index]

        primary_mask = labels == primary_label
        primary_points = object_points[primary_mask]
        primary_colors = object_colors[primary_mask]
        primary_center = np.median(primary_points, axis=0)

        self.get_logger().info(
            f"DBSCAN clusters: {len(unique_labels)}, "
            f"selected label: {primary_label}, "
            f"selected points: {primary_count}, "
            f"score={primary_score:.3f}"
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

                xy_dist = np.linalg.norm(cluster_center[:2] - primary_center[:2])
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
            self.get_logger().warn(f"Selected object cloud too small: {len(selected_points)}")
            self.record_cloud_stage("05_after_dbscan_selected", object_pcd)
            self.plot_point_count_by_stage()
            return object_pcd

        selected_pcd = self.copy_pcd_from_arrays(selected_points, selected_colors)

        self.save_debug_pointcloud("07_after_dbscan_selected_clusters.ply", selected_pcd)
        # Keep old filename for your existing inspection commands.
        self.save_debug_pointcloud("05_after_dbscan_selected_clusters.ply", selected_pcd)

        self.record_cloud_stage("05_after_dbscan_selected", selected_pcd)
        self.plot_point_count_by_stage()

        self.get_logger().info(f"Final selected object points: {len(selected_points)}")

        return selected_pcd

    def compute_grasp_from_cloud(self):
        compute_start_time = time.perf_counter()

        if self.save_eval_outputs and self.current_trial_id is None:
            self.start_new_trial()

        if len(self.global_pcd.points) == 0:
            self.get_logger().warn("No point cloud captured. Cannot compute grasp.")
            return

        self.get_logger().info("Cleaning fused point cloud...")

        pcd = self.global_pcd.voxel_down_sample(voxel_size=self.final_voxel_size)
        pcd = self.align_fused_cloud_to_table_z(pcd)

        cleaning_start_time = time.perf_counter()
        object_pcd = self.remove_dominant_plane_and_cluster(pcd)
        cleaning_time = time.perf_counter() - cleaning_start_time

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

        grasp_start_time = time.perf_counter()

        try:
            prediction = self.grasp_model.predict(context)
        except Exception as e:
            self.get_logger().error(f"Grasp model [{self.grasp_model_name}] failed: {e}")
            return

        grasp_prediction_time = time.perf_counter() - grasp_start_time
        total_compute_time = time.perf_counter() - compute_start_time

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
            self.get_logger().info(f"Yaw: {prediction.debug['yaw_deg']:.1f} deg")

        if prediction.width is not None:
            self.get_logger().info(f"Predicted gripper width: {prediction.width:.3f} m")

        self.get_logger().info(f"Grasp score: {prediction.score:.3f}")

        self.final_pcd = object_pcd

        save_dir = os.path.expanduser('~/robot_description/sfm_dataset/dense')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'auto_grasp_clean_object.ply')
        o3d.io.write_point_cloud(save_path, object_pcd)

        self.get_logger().info(f"Saved cleaned object cloud to: {save_path}")

        if self.save_eval_outputs and self.current_pcd_dir is not None:
            trial_clean_path = self.current_pcd_dir / "auto_grasp_clean_object.ply"
            o3d.io.write_point_cloud(str(trial_clean_path), object_pcd)

        self.publish_pointcloud(self.final_pcd)
        self.load_and_publish_latest_gt_pointcloud()

        timing_data = {
            "cleaning_time_s": cleaning_time,
            "grasp_prediction_time_s": grasp_prediction_time,
            "total_compute_time_s": total_compute_time,
        }

        gt_data = self.load_latest_ground_truth()

        self.plot_topdown_grasp_result(
            object_pcd=object_pcd,
            pose=pose,
            gt_data=gt_data
        )

        if self.save_eval_outputs:
            row = self.build_experiment_row(
                object_pcd=object_pcd,
                pose=pose,
                prediction=prediction,
                timing_data=timing_data
            )

            self.append_csv_row(self.results_csv_path, row)
            self.get_logger().info(f"Saved experiment row to: {self.results_csv_path}")

    def load_and_publish_latest_gt_pointcloud(self):
        gt_data = self.load_latest_ground_truth()

        if gt_data is None:
            self.get_logger().warn("No GT CSV row available. Cannot publish GT point cloud.")
            return

        gt_pcd_path = gt_data.get("gt_pcd_path", "")

        if gt_pcd_path is None or gt_pcd_path == "":
            self.get_logger().warn("Latest GT row has no gt_pcd_path.")
            return

        gt_pcd_path = os.path.expanduser(gt_pcd_path)

        if not os.path.exists(gt_pcd_path):
            self.get_logger().warn(f"GT point cloud file does not exist: {gt_pcd_path}")
            return

        try:
            gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)

            if gt_pcd is None or len(gt_pcd.points) == 0:
                self.get_logger().warn(f"GT point cloud is empty: {gt_pcd_path}")
                return

            # Force GT cloud colour to green for clear RViz comparison.
            green = np.tile(
                np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
                (len(gt_pcd.points), 1)
            )
            gt_pcd.colors = o3d.utility.Vector3dVector(green)

            self.gt_pcd = gt_pcd

            self.publish_pointcloud_to_publisher(
                pcd=self.gt_pcd,
                publisher=self.gt_pc_pub,
                frame_id=self.target_frame
            )

            self.get_logger().info(
                f"Published GT object point cloud: {gt_pcd_path}, "
                f"points={len(gt_pcd.points)}"
            )

        except Exception as e:
            self.get_logger().warn(f"Failed to load/publish GT point cloud: {e}")

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
        self.publish_pointcloud_to_publisher(
            pcd=pcd,
            publisher=self.pc_pub,
            frame_id=self.target_frame
        )

    def publish_pointcloud_to_publisher(self, pcd, publisher, frame_id):
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
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        msg = pc2.create_cloud(header, fields, pc_data)
        publisher.publish(msg)

    def publish_joint_goal(self, positions):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in positions]
        point.time_from_start = Duration(seconds=self.move_duration_sec).to_msg()

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