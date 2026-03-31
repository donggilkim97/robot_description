import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
import cv2
import json
import os
import sys
import ctypes

libc = ctypes.CDLL(None)
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
null_fd = os.open(os.devnull, os.O_RDWR)
libc.fflush(c_stderr)
os.dup2(null_fd, 2)

class AutoScanner(Node):
    def __init__(self):
        super().__init__(
            'auto_scanner',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)
        self.base_dir = "sfm_dataset"
        self.img_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.poses_data = {}
        self.image_count = 0
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.image_sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, '/ur_manipulator_controller/joint_trajectory', 10)
        self.latest_image = None
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
        self.timer = self.create_timer(0.2, self.timer_callback)
        print("\n[Auto Scanner Started] Moving to start position...\n")

    def image_callback(self, msg):
        self.latest_image = msg

    def timer_callback(self):
        if self.state == 'INIT':
            self.move_to_start()
            self.wait_ticks = 25
            self.state = 'WAITING_START'
        elif self.state == 'WAITING_START':
            self.wait_ticks -= 1
            if self.wait_ticks <= 0:
                print("Starting continuous scan trajectory...")
                self.execute_full_trajectory()
                self.wait_ticks = 100
                self.state = 'SCANNING'
        elif self.state == 'SCANNING':
            if self.latest_image is not None:
                self.save_data()
            self.wait_ticks -= 1
            if self.wait_ticks <= 0:
                print(f"\nDataset collection finished! Total {self.image_count} images saved.")
                self.timer.cancel()
                sys.exit(0)

    def move_to_start(self):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        point = JointTrajectoryPoint()
        point.positions = self.waypoints[0]
        point.time_from_start.sec = 4
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

    def save_data(self):
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time(seconds=0))
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            
            h, w = cv_image.shape[:2]
            crop_h = int(h * 5 / 6)
            cropped_image = cv_image[:crop_h, :]
            
            image_filename = f"image_{self.image_count:04d}.png"
            cv2.imwrite(os.path.join(self.img_dir, image_filename), cropped_image)
            self.poses_data[image_filename] = {
                "translation": [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
                "rotation": [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            }
            with open(os.path.join(self.base_dir, "transforms.json"), 'w') as f:
                json.dump(self.poses_data, f, indent=4)
            print(f"SUCCESS: Saved {image_filename}")
            self.image_count += 1
        except Exception as e:
            print(f"ERROR saving {self.image_count:04d}: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = AutoScanner()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()