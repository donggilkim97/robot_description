import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import open3d as o3d
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
import os

class FastAIDepthScanner(Node):
    def __init__(self):
        super().__init__('fast_ai_depth_scanner', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, '/ur_manipulator_controller/joint_trajectory', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/ai_scanned_pointcloud', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading ZoeDepth model on [{self.device.type.upper()}]... (This may take a moment)")
        
        self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.global_pcd = o3d.geometry.PointCloud()
        self.is_scanning_finished = False
        
        self.frame_counter = 0
        self.processed_frames = 0
        self.process_every_n_frames = 3  # 3프레임 중 1프레임만 연산하여 속도 극대화
        
        self.waypoints = [
            [-0.2194, -1.9407, 1.8675, -1.4961, -1.5702, 2.9206],
            [0.1177, -1.8458, 1.805, -1.5289, -1.5698, 3.2577],
            [0.4045, -1.6577, 1.8437, -1.5484, -2.0196, 3.592],
            [-0.9338, -1.7309, 1.9168, -1.2612, -1.2349, 2.1211],
            [-0.7572, -2.8091, 2.3533, -1.4577, -1.2548, 2.4286],
            [-0.2194, -1.9407, 1.8675, -1.4961, -1.5702, 2.9206]
        ]
        self.state = 'INIT'
        self.wait_ticks = 0
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("[Scanner Started] Moving to the starting position...")

    def image_callback(self, msg):
        if self.is_scanning_finished:
            return

        # 1. 프레임 스킵 (스로틀링)
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return

        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"Waiting for TF...", throttle_duration_sec=2.0)
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        h, w = cv_image.shape[:2]
        crop_h = int(h * 5 / 6)
        cv_image = cv_image[:crop_h, :]

        pil_img = PILImage.fromarray(cv_image)

        with torch.no_grad():
            depth_map = self.model.infer_pil(pil_img)

        depth_map = depth_map.astype(np.float32)

        q = trans.transform.rotation
        t = trans.transform.translation
        rot_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        cam_z_dir = rot_mat[:, 0]
        cos_theta = abs(cam_z_dir[2])

        cy, cx = crop_h // 2, w // 2
        center_depth_pred = depth_map[cy, cx]

        if cos_theta > 0.1 and center_depth_pred > 0.01:
            true_depth = t.z / cos_theta
            dynamic_scale = true_depth / center_depth_pred
        else:
            dynamic_scale = 1.0

        depth_map = depth_map * dynamic_scale

        # 2. 이미지 크기 1/4 축소 (연산 속도 4배 향상)
        small_w, small_h = w // 2, crop_h // 2
        cv_image_small = cv2.resize(cv_image, (small_w, small_h))
        depth_map_small = cv2.resize(depth_map, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

        # 카메라 파라미터도 절반으로 축소
        fx_s = 1536.0 / 2.0
        fy_s = 1536.0 / 2.0
        cx_s = small_w / 2.0
        cy_s = small_h / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(small_w, small_h, fx_s, fy_s, cx_s, cy_s)

        o3d_color = o3d.geometry.Image(cv_image_small)
        o3d_depth = o3d.geometry.Image(depth_map_small)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        
        # 3. 추가하기 전에 개별적으로 가볍게 압축 (스노우볼 이펙트 방지)
        temp_pcd = temp_pcd.voxel_down_sample(voxel_size=0.015)
        
        cam_to_ros = np.array([[ 0,  0,  1,  0],
                               [-1,  0,  0,  0],
                               [ 0, -1,  0,  0],
                               [ 0,  0,  0,  1]])
        temp_pcd.transform(cam_to_ros)

        transform_mat = np.eye(4)
        transform_mat[:3, :3] = rot_mat
        transform_mat[0, 3] = t.x
        transform_mat[1, 3] = t.y
        transform_mat[2, 3] = t.z

        temp_pcd.transform(transform_mat)

        # 전체 맵 다운샘플링은 여기에서 제거됨
        self.global_pcd += temp_pcd
        
        self.processed_frames += 1
        if self.processed_frames % 5 == 0:
            self.get_logger().info(f"Adding points to map... Processed {self.processed_frames} fast frames.")

    def timer_callback(self):
        if self.state == 'INIT':
            self.move_to_start()
            self.wait_ticks = 30
            self.state = 'WAITING_START'
        elif self.state == 'WAITING_START':
            self.wait_ticks -= 1
            if self.wait_ticks <= 0:
                self.get_logger().info("Starting high-speed continuous scan trajectory...")
                self.execute_full_trajectory()
                self.wait_ticks = 150
                self.state = 'SCANNING'
        elif self.state == 'SCANNING':
            self.wait_ticks -= 1
            if self.wait_ticks <= 0:
                self.is_scanning_finished = True
                self.get_logger().info("Scan complete! Running final map optimization...")
                
                # 4. 스캔이 모두 끝난 뒤 딱 한 번만 전체 맵 깔끔하게 압축
                self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.01)
                
                save_dir = os.path.expanduser('~/robot_description/sfm_dataset/dense')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'ai_fused.ply')
                o3d.io.write_point_cloud(save_path, self.global_pcd)
                
                self.get_logger().info(f"[SUCCESS] Point cloud saved to: {save_path}")
                self.get_logger().info("[INFO] Publishing PointCloud2 to RViz. Press Ctrl+C to stop.")
                self.state = 'PUBLISHING'
        elif self.state == 'PUBLISHING':
            self.publish_ros_pointcloud()

    def publish_ros_pointcloud(self):
        if len(self.global_pcd.points) == 0:
            return

        points = np.asarray(self.global_pcd.points, dtype=np.float32)
        colors = np.asarray(self.global_pcd.colors, dtype=np.float32)

        rgba = np.zeros((colors.shape[0], 4), dtype=np.uint8)
        rgba[:, 0] = (colors[:, 0] * 255).astype(np.uint8)
        rgba[:, 1] = (colors[:, 1] * 255).astype(np.uint8)
        rgba[:, 2] = (colors[:, 2] * 255).astype(np.uint8)
        rgba[:, 3] = 255

        rgb_float = rgba.view(np.float32)
        pc_data = np.hstack((points, rgb_float))

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        pc_msg = pc2.create_cloud(header, fields, pc_data)
        self.pc_pub.publish(pc_msg)

    def move_to_start(self):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        point = JointTrajectoryPoint()
        point.positions = self.waypoints[0]
        point.time_from_start.sec = 2
        msg.points.append(point)
        self.traj_pub.publish(msg)

    def execute_full_trajectory(self):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        accumulated_time = 0
        for i in range(1, len(self.waypoints)):
            point = JointTrajectoryPoint()
            point.positions = self.waypoints[i]
            accumulated_time += 2
            point.time_from_start.sec = accumulated_time
            msg.points.append(point)
            
        self.traj_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FastAIDepthScanner()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()