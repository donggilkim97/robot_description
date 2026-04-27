import copy
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration as RclpyDuration

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as BuiltinDuration
from builtin_interfaces.msg import Time as BuiltinTime
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from moveit_msgs.srv import GetPositionIK
from tf2_ros import Buffer, TransformListener


class GraspExecutor(Node):
    def __init__(self):
        super().__init__(
            'grasp_executor_node',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )

        self.callback_group = ReentrantCallbackGroup()

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/target_grasp_pose',
            self.pose_callback,
            10,
            callback_group=self.callback_group
        )

        self.command_sub = self.create_subscription(
            String,
            '/grasp_execute_command',
            self.command_callback,
            10,
            callback_group=self.callback_group
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )

        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_controller/joint_trajectory',
            10
        )

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/ur_manipulator_controller/joint_trajectory',
            10
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik',
            callback_group=self.callback_group
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # MoveIt / robot settings
        self.group_name = 'ur_manipulator'
        self.ik_link_name = 'tool0'
        self.base_frame = 'base_link'

        self.arm_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # If gripper does not move, check:
        # ros2 control list_joints
        self.gripper_joint_names = ['finger_joint']

        self.gripper_open_position = 0.0
        self.gripper_close_position = 0.70

        # TCP tuning.
        # target_grasp_pose is object grasp point.
        # tool0 needs to stay above that point by TCP length.
        self.tcp_offset_z = 0.15

        # Positive value makes the tool go deeper.
        # Start with 0.02 m. Try 0.03 or 0.04 if still too shallow.
        self.grasp_depth_extra = 0.035

        self.pre_grasp_clearance = 0.12
        self.lift_distance = 0.20

        self.arm_move_time_sec = 3
        self.gripper_move_time_sec = 1

        # Wait settings
        self.joint_goal_tolerance = 0.025
        self.joint_wait_timeout = 8.0
        self.settle_after_motion = 0.5
        self.settle_before_gripper = 0.8

        # First test safer with collision checking disabled.
        self.avoid_collisions = False

        # Use current tool0 orientation for top-down grasp.
        self.use_current_tool_orientation = True

        # Manual execution is safer.
        self.auto_execute_on_pose = False

        self.latest_target_pose = None
        self.current_joint_state = None
        self.is_executing = False

        # Reduce repeated pose logs
        self.last_logged_pose_time = 0.0
        self.pose_log_interval = 2.0

        self.get_logger().info("GraspExecutor ready.")
        self.get_logger().info("use_sim_time is enabled.")
        self.get_logger().info("Waiting for /target_grasp_pose...")
        self.get_logger().info("Manual commands:")
        self.get_logger().info(
            "ros2 topic pub --once /grasp_execute_command std_msgs/msg/String \"{data: 'go'}\""
        )
        self.get_logger().info(
            "ros2 topic pub --once /grasp_execute_command std_msgs/msg/String \"{data: 'open'}\""
        )
        self.get_logger().info(
            "ros2 topic pub --once /grasp_execute_command std_msgs/msg/String \"{data: 'close'}\""
        )
        self.get_logger().info(
            "ros2 topic pub --once /grasp_execute_command std_msgs/msg/String \"{data: 'test_arm'}\""
        )

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def pose_callback(self, msg):
        self.latest_target_pose = copy.deepcopy(msg)

        now = time.time()
        if now - self.last_logged_pose_time > self.pose_log_interval:
            self.last_logged_pose_time = now
            self.get_logger().info(
                f"Received target grasp pose: "
                f"x={msg.pose.position.x:.3f}, "
                f"y={msg.pose.position.y:.3f}, "
                f"z={msg.pose.position.z:.3f}"
            )

        if self.auto_execute_on_pose and not self.is_executing:
            thread = threading.Thread(
                target=self.execute_grasp,
                daemon=True
            )
            thread.start()

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command == "go":
            if self.latest_target_pose is None:
                self.get_logger().warn("No /target_grasp_pose received yet.")
                return

            if self.is_executing:
                self.get_logger().warn("Already executing grasp sequence.")
                return

            thread = threading.Thread(
                target=self.execute_grasp,
                daemon=True
            )
            thread.start()

        elif command == "open":
            self.control_gripper(close=False)

        elif command == "close":
            self.control_gripper(close=True)

        elif command == "test_arm":
            self.test_arm_motion()

        else:
            self.get_logger().warn(
                f"Unknown command: {command}. Use go, open, close, or test_arm."
            )

    def execute_grasp(self):
        if self.latest_target_pose is None:
            self.get_logger().warn("No target pose available.")
            return

        self.is_executing = True

        try:
            target_pose_msg = copy.deepcopy(self.latest_target_pose)
            target = target_pose_msg.pose

            self.get_logger().info("=== Starting grasp sequence ===")

            # 0. Open gripper
            self.get_logger().info("0. Opening gripper")
            self.control_gripper(close=False)
            time.sleep(self.gripper_move_time_sec + 0.5)

            # 1. Pre-grasp
            pre_grasp = copy.deepcopy(target)
            pre_grasp.position.z += self.tcp_offset_z + self.pre_grasp_clearance
            pre_grasp = self.apply_tool_orientation(pre_grasp)

            self.get_logger().info("1. Moving to pre-grasp")
            pre_grasp_joints = self.move_robot(pre_grasp)

            if pre_grasp_joints is None:
                self.get_logger().error("Failed at pre-grasp. Aborting.")
                return

            self.wait_until_joint_goal_reached(pre_grasp_joints)
            time.sleep(self.settle_after_motion)

            # 2. Move down to grasp
            grasp = copy.deepcopy(target)

            # Deeper grasp:
            # z = object grasp z + TCP length - extra downward depth
            grasp.position.z += self.tcp_offset_z - self.grasp_depth_extra
            grasp = self.apply_tool_orientation(grasp)

            self.get_logger().info(
                f"2. Moving down to grasp "
                f"(tcp_offset={self.tcp_offset_z:.3f}, "
                f"depth_extra={self.grasp_depth_extra:.3f})"
            )

            grasp_joints = self.move_robot(grasp)

            if grasp_joints is None:
                self.get_logger().error("Failed at grasp pose. Aborting.")
                return

            self.wait_until_joint_goal_reached(grasp_joints)

            self.get_logger().info(
                f"Waiting {self.settle_before_gripper:.1f}s before closing gripper..."
            )
            time.sleep(self.settle_before_gripper)

            # 3. Close gripper only after robot reached target
            self.get_logger().info("3. Closing gripper")
            self.control_gripper(close=True)
            time.sleep(self.gripper_move_time_sec + 1.0)

            # 4. Lift
            lift = copy.deepcopy(grasp)
            lift.position.z += self.lift_distance

            self.get_logger().info("4. Lifting object")
            lift_joints = self.move_robot(lift)

            if lift_joints is None:
                self.get_logger().error("Failed at lift pose.")
                return

            self.wait_until_joint_goal_reached(lift_joints)
            time.sleep(self.settle_after_motion)

            self.get_logger().info("=== Grasp sequence completed ===")

        except Exception as e:
            self.get_logger().error(f"Exception during grasp execution: {e}")

        finally:
            self.is_executing = False

    def apply_tool_orientation(self, pose):
        if not self.use_current_tool_orientation:
            return pose

        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ik_link_name,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.5)
            )

            pose.orientation = trans.transform.rotation

        except Exception as e:
            self.get_logger().warn(
                f"Could not get current {self.ik_link_name} orientation. "
                f"Using input pose orientation. Error: {e}"
            )

        return pose

    def move_robot(self, target_pose):
        if not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("/compute_ik service is not available.")
            return None

        req = GetPositionIK.Request()

        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.avoid_collisions = self.avoid_collisions

        req.ik_request.pose_stamped.header.frame_id = self.base_frame

        # Use zero stamp to avoid sim-time / wall-time mismatch.
        req.ik_request.pose_stamped.header.stamp = BuiltinTime(
            sec=0,
            nanosec=0
        )

        req.ik_request.pose_stamped.pose = target_pose
        req.ik_request.timeout = BuiltinDuration(sec=2, nanosec=0)

        if self.current_joint_state is not None:
            req.ik_request.robot_state.joint_state = self.current_joint_state

        self.get_logger().info(
            f"Requesting IK: "
            f"x={target_pose.position.x:.3f}, "
            f"y={target_pose.position.y:.3f}, "
            f"z={target_pose.position.z:.3f}"
        )

        future = self.ik_client.call_async(req)

        start_time = time.time()
        timeout_sec = 5.0

        while rclpy.ok() and not future.done():
            if time.time() - start_time > timeout_sec:
                self.get_logger().error("IK service call timed out.")
                return None
            time.sleep(0.02)

        res = future.result()

        if res is None:
            self.get_logger().error("IK service returned None.")
            return None

        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().error(
                f"IK failed. MoveIt error code: {res.error_code.val}"
            )
            return None

        solution_names = list(res.solution.joint_state.name)
        solution_positions = list(res.solution.joint_state.position)

        joint_position_map = {
            name: pos for name, pos in zip(solution_names, solution_positions)
        }

        ordered_positions = []

        for joint_name in self.arm_joint_names:
            if joint_name not in joint_position_map:
                self.get_logger().error(
                    f"IK solution missing joint: {joint_name}"
                )
                self.get_logger().error(
                    f"Available joints: {solution_names}"
                )
                return None

            ordered_positions.append(float(joint_position_map[joint_name]))

        self.publish_arm_trajectory(ordered_positions)

        return ordered_positions

    def publish_arm_trajectory(self, joint_positions):
        msg = JointTrajectory()

        # Zero stamp means "start immediately".
        msg.header.stamp = BuiltinTime(sec=0, nanosec=0)

        msg.joint_names = self.arm_joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joint_positions]
        point.time_from_start = BuiltinDuration(
            sec=self.arm_move_time_sec,
            nanosec=0
        )

        msg.points.append(point)

        self.arm_pub.publish(msg)

        self.get_logger().info(
            "Published arm trajectory: "
            + ", ".join([f"{p:.3f}" for p in joint_positions])
        )

    def get_current_arm_positions(self):
        if self.current_joint_state is None:
            return None

        name_to_pos = {
            name: pos
            for name, pos in zip(
                self.current_joint_state.name,
                self.current_joint_state.position
            )
        }

        current_positions = []

        for joint_name in self.arm_joint_names:
            if joint_name not in name_to_pos:
                return None

            current_positions.append(float(name_to_pos[joint_name]))

        return current_positions

    def wait_until_joint_goal_reached(self, target_positions):
        self.get_logger().info("Waiting until arm reaches target joint positions...")

        start_time = time.time()

        while rclpy.ok():
            current_positions = self.get_current_arm_positions()

            if current_positions is not None:
                errors = [
                    abs(c - t)
                    for c, t in zip(current_positions, target_positions)
                ]

                max_error = max(errors)

                if max_error < self.joint_goal_tolerance:
                    self.get_logger().info(
                        f"Arm reached target. Max joint error: {max_error:.4f} rad"
                    )
                    return True

            elapsed = time.time() - start_time

            if elapsed > self.joint_wait_timeout:
                self.get_logger().warn(
                    "Timed out waiting for arm target. Continuing anyway."
                )
                return False

            time.sleep(0.05)

    def control_gripper(self, close):
        msg = JointTrajectory()

        # Zero stamp means execute immediately.
        msg.header.stamp = BuiltinTime(sec=0, nanosec=0)

        msg.joint_names = self.gripper_joint_names

        point = JointTrajectoryPoint()

        if close:
            target = self.gripper_close_position
        else:
            target = self.gripper_open_position

        point.positions = [float(target)]
        point.time_from_start = BuiltinDuration(
            sec=self.gripper_move_time_sec,
            nanosec=0
        )

        msg.points.append(point)

        self.gripper_pub.publish(msg)

        if close:
            self.get_logger().info(f"Gripper close command: {target:.3f}")
        else:
            self.get_logger().info(f"Gripper open command: {target:.3f}")

    def test_arm_motion(self):
        self.get_logger().info("Sending simple test arm motion.")

        test_pose = [
            -0.2194,
            -1.9407,
            1.8675,
            -1.4961,
            -1.5702,
            2.9206
        ]

        self.publish_arm_trajectory(test_pose)


def main(args=None):
    rclpy.init(args=args)

    node = GraspExecutor()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()