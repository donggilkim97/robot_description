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
        self.save_dir = "sfm_dataset"
        os.makedirs(self.save_dir, exist_ok=True)
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
            [-0.7572, -2.8091, 2.3533, -1.4577, -1.2548, 2.4286]
        ]
        self.current_wp = 0
        self.timer = self.create_timer(8.0, self.timer_callback)
        print("\n[Auto Scanner Started] Waiting for first waypoint...\n")

    def image_callback(self, msg):
        self.latest_image = msg

    def timer_callback(self):
        if self.current_wp > 0:
            if self.latest_image is not None:
                self.save_data()
            else:
                print("ERROR: No image received")
        
        if self.current_wp < len(self.waypoints):
            print(f"Moving to waypoint {self.current_wp + 1}/{len(self.waypoints)}")
            self.move_to(self.waypoints[self.current_wp])
            self.current_wp += 1
        else:
            print(f"\nDataset collection finished! Total {self.image_count} images saved.")
            self.timer.cancel()
            sys.exit(0)

    def move_to(self, positions):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 5
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
            cv2.imwrite(os.path.join(self.save_dir, image_filename), cropped_image)
            self.poses_data[image_filename] = {
                "translation": [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
                "rotation": [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            }
            with open(os.path.join(self.save_dir, "transforms.json"), 'w') as f:
                json.dump(self.poses_data, f, indent=4)
            print(f"SUCCESS: Saved {image_filename} & TF data")
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