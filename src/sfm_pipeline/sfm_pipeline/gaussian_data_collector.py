import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import os
import json
import math

class GaussianDataCollector(Node):
    def __init__(self):
        super().__init__('gaussian_data_collector', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, '/ur_manipulator_controller/joint_trajectory', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.save_dir = os.path.expanduser('~/robot_description/sfm_dataset/gaussian_data')
        self.image_dir = os.path.join(self.save_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.frames_data = []
        self.image_count = 0
        self.is_scanning_finished = False
        
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
        self.get_logger().info("[INIT] Gaussian Data Collector node started.")

    def image_callback(self, msg):
        if self.state != 'SCANNING' or self.is_scanning_finished:
            return

        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"Waiting for TF... ({e})", throttle_duration_sec=2.0)
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        
        filename = f"{self.image_count:04d}.png"
        filepath = os.path.join(self.image_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))

        q = trans.transform.rotation
        t = trans.transform.translation
        rot_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        transform_mat = np.eye(4)
        transform_mat[:3, :3] = rot_mat
        transform_mat[0, 3] = t.x
        transform_mat[1, 3] = t.y
        transform_mat[2, 3] = t.z

        cam_to_nerf = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ])
        
        nerf_matrix = transform_mat @ cam_to_nerf

        frame_info = {
            "file_path": f"images/{filename}",
            "transform_matrix": nerf_matrix.tolist()
        }
        self.frames_data.append(frame_info)
        self.image_count += 1

        if self.image_count % 20 == 0:
            self.get_logger().info(f"[SCANNING] Captured {self.image_count} images so far...")

    def timer_callback(self):
        if self.state == 'INIT':
            self.get_logger().info("[STAGE 1] Moving to start position. Waiting 3 seconds...")
            self.move_to_start()
            self.wait_ticks = 30
            self.state = 'WAITING_START'
        elif self.state == 'WAITING_START':
            self.wait_ticks -= 1
            if self.wait_ticks <= 0:
                self.get_logger().info("[STAGE 2] Starting full trajectory! Scanning for 26 seconds...")
                self.execute_full_trajectory()
                self.wait_ticks = 260
                self.state = 'SCANNING'
        elif self.state == 'SCANNING':
            self.wait_ticks -= 1
            if self.wait_ticks <= 0:
                self.get_logger().info("[STAGE 3] Trajectory finished. Saving data...")
                self.is_scanning_finished = True
                self.save_transforms()
                self.state = 'DONE'
        elif self.state == 'DONE':
            raise SystemExit

    def save_transforms(self):
        fl_x = 1536.0
        fl_y = 1536.0
        cx = 640.0
        cy = 360.0
        w = 1280
        h = 720
        
        camera_angle_x = math.atan(w / (fl_x * 2)) * 2

        transforms_dict = {
            "camera_angle_x": camera_angle_x,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "frames": self.frames_data
        }

        json_path = os.path.join(self.save_dir, 'transforms.json')
        with open(json_path, 'w') as f:
            json.dump(transforms_dict, f, indent=4)
        self.get_logger().info(f"[SUCCESS] Saved {self.image_count} frames to {json_path}")

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
            accumulated_time += 4
            point.time_from_start.sec = accumulated_time
            msg.points.append(point)
            
        self.traj_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GaussianDataCollector()
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