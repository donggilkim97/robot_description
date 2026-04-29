import copy
import threading
import time
import math

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

from scipy.spatial.transform import Rotation as R


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

        # -----------------------------
        # ROS parameters
        # -----------------------------
        self.declare_parameter("orientation_mode", "current_with_target_yaw")
        self.declare_parameter("auto_execute_on_pose", False)
        self.declare_parameter("avoid_collisions", False)
        self.declare_parameter("grasp_yaw_offset_deg", -90.0)

        self.orientation_mode = self.get_parameter("orientation_mode").value
        self.grasp_yaw_offset_deg = float(
            self.get_parameter("grasp_yaw_offset_deg").value
        )
        self.get_logger().info(
            f"Grasp yaw offset: {self.grasp_yaw_offset_deg:.1f} deg"
        )
        self.grasp_yaw_offset_rad = math.radians(self.grasp_yaw_offset_deg)
        self.auto_execute_on_pose = bool(
            self.get_parameter("auto_execute_on_pose").value
        )
        self.avoid_collisions = bool(
            self.get_parameter("avoid_collisions").value
        )

        # -----------------------------
        # MoveIt / robot settings
        # -----------------------------
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

        self.gripper_joint_names = ['finger_joint']

        # -----------------------------
        # Gripper settings
        # -----------------------------
        self.gripper_open_position = 0.0

        # Do not over-close at first. 0.70 is too aggressive for contact simulation.
        self.gripper_close_position = 0.70

        # Slower close gives PhysX time to resolve contact.
        self.gripper_move_time_sec = 4.0

        # Extra wait after gripper command before lifting.
        self.min_close_wait_sec = 4.5
        self.post_grasp_hold_time = 2.0

        # Gripper wait settings
        self.gripper_wait_timeout = 8.0
        self.gripper_position_tolerance = 0.03
        self.gripper_required_stable_time = 0.7

        # -----------------------------
        # TCP / grasp depth settings
        # -----------------------------
        self.tcp_offset_z = 0.15

        # Positive = deeper.
        self.grasp_depth_extra = 0.00

        self.pre_grasp_clearance = 0.12
        self.lift_distance = 0.20

        # -----------------------------
        # Arm motion timing
        # -----------------------------
        self.arm_move_time_pregrasp = 4.0
        self.arm_move_time_grasp = 5.0
        self.arm_move_time_lift = 4.0
        self.default_arm_move_time = 4.0

        # -----------------------------
        # Arm wait settings
        # -----------------------------
        self.joint_goal_tolerance = 0.035
        self.required_stable_time = 0.6

        self.min_wait_timeout = 60.0
        self.wait_timeout_scale = 15.0

        self.settle_after_motion = 0.5
        self.settle_before_gripper = 1.5

        # Velocity check disabled because Isaac velocity can be noisy on slow machines.
        self.require_velocity_settle = False
        self.joint_velocity_tolerance = 0.08

        self.latest_target_pose = None
        self.current_joint_state = None
        self.is_executing = False

        self.last_logged_pose_time = 0.0
        self.pose_log_interval = 2.5

        self.get_logger().info("GraspExecutor ready.")
        self.get_logger().info(
            f"Orientation mode: {self.orientation_mode}"
        )
        self.get_logger().info(
            f"Auto execute on pose: {self.auto_execute_on_pose}"
        )
        self.get_logger().info(
            "Use: ros2 topic pub --once /grasp_execute_command "
            "std_msgs/msg/String \"{data: 'go'}\""
        )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def pose_callback(self, msg):
        if self.is_executing:
            return

        self.latest_target_pose = copy.deepcopy(msg)

        now = time.time()
        if now - self.last_logged_pose_time > self.pose_log_interval:
            self.last_logged_pose_time = now

            yaw_deg = self.get_yaw_from_pose_msg(msg)

            self.get_logger().info(
                f"Received target grasp pose: "
                f"x={msg.pose.position.x:.3f}, "
                f"y={msg.pose.position.y:.3f}, "
                f"z={msg.pose.position.z:.3f}, "
                f"yaw={yaw_deg:.1f} deg"
            )

        if self.auto_execute_on_pose and not self.is_executing:
            threading.Thread(
                target=self.execute_grasp,
                daemon=True
            ).start()

    def command_callback(self, msg):
        command = msg.data.strip().lower()

        if command == "go":
            if self.latest_target_pose is None:
                self.get_logger().warn("No /target_grasp_pose received yet.")
                return

            if self.is_executing:
                self.get_logger().warn("Already executing grasp sequence.")
                return

            threading.Thread(
                target=self.execute_grasp,
                daemon=True
            ).start()

        elif command == "open":
            self.control_gripper(close=False)

        elif command == "close":
            self.control_gripper(close=True)

        elif command == "test_arm":
            self.test_arm_motion()

        elif command == "status":
            self.print_status()

        else:
            self.get_logger().warn(
                f"Unknown command: {command}. Use go, open, close, test_arm, or status."
            )

    # ---------------------------------------------------------
    # Main grasp sequence
    # ---------------------------------------------------------
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
            self.wait_until_gripper_motion_done(
                target_position=self.gripper_open_position,
                label="open"
            )

            # 1. Pre-grasp
            pre_grasp = copy.deepcopy(target)
            pre_grasp.position.z += self.tcp_offset_z + self.pre_grasp_clearance
            pre_grasp = self.apply_grasp_orientation(pre_grasp)

            self.get_logger().info("1. Moving to pre-grasp")
            pre_grasp_joints = self.move_robot(
                pre_grasp,
                move_time_sec=self.arm_move_time_pregrasp
            )

            if pre_grasp_joints is None:
                self.get_logger().error("Failed at pre-grasp. Aborting.")
                return

            if not self.wait_until_arm_reached(
                pre_grasp_joints,
                move_time_sec=self.arm_move_time_pregrasp,
                label="pre-grasp"
            ):
                self.get_logger().error("Arm did not reach pre-grasp. Aborting.")
                return

            time.sleep(self.settle_after_motion)

            # 2. Move down to grasp
            grasp = copy.deepcopy(target)
            grasp.position.z += self.tcp_offset_z - self.grasp_depth_extra
            grasp = self.apply_grasp_orientation(grasp)

            self.get_logger().info(
                f"2. Moving down to grasp "
                f"(tcp_offset={self.tcp_offset_z:.3f}, "
                f"depth_extra={self.grasp_depth_extra:.3f})"
            )

            grasp_joints = self.move_robot(
                grasp,
                move_time_sec=self.arm_move_time_grasp
            )

            if grasp_joints is None:
                self.get_logger().error("Failed at grasp pose. Aborting.")
                return

            if not self.wait_until_arm_reached(
                grasp_joints,
                move_time_sec=self.arm_move_time_grasp,
                label="grasp"
            ):
                self.get_logger().error("Arm did not reach grasp pose. Aborting.")
                return

            self.get_logger().info(
                f"Arm reached grasp pose. Waiting "
                f"{self.settle_before_gripper:.1f}s before closing gripper..."
            )
            time.sleep(self.settle_before_gripper)

            # 3. Close gripper
            self.get_logger().info("3. Closing gripper")
            self.control_gripper(close=True)

            self.wait_until_gripper_motion_done(
                target_position=self.gripper_close_position,
                label="close"
            )

            self.get_logger().info(
                f"Extra post-grasp hold: {self.post_grasp_hold_time:.1f}s"
            )
            time.sleep(self.post_grasp_hold_time)

            # 4. Lift only after gripper has had time to close/contact.
            lift = copy.deepcopy(grasp)
            lift.position.z += self.lift_distance

            self.get_logger().info("4. Lifting object")
            lift_joints = self.move_robot(
                lift,
                move_time_sec=self.arm_move_time_lift
            )

            if lift_joints is None:
                self.get_logger().error("Failed at lift pose.")
                return

            self.wait_until_arm_reached(
                lift_joints,
                move_time_sec=self.arm_move_time_lift,
                label="lift"
            )

            self.get_logger().info("=== Grasp sequence completed ===")

        except Exception as e:
            self.get_logger().error(f"Exception during grasp execution: {e}")

        finally:
            self.is_executing = False

    # ---------------------------------------------------------
    # Orientation handling
    # ---------------------------------------------------------
    def apply_grasp_orientation(self, pose):
        mode = str(self.orientation_mode).lower().strip()

        if mode == "target":
            return pose

        if mode == "current":
            return self.apply_current_tool_orientation(pose)

        if mode == "current_with_target_yaw":
            return self.apply_current_orientation_with_target_yaw(pose)

        self.get_logger().warn(
            f"Unknown orientation_mode '{self.orientation_mode}'. Using target orientation."
        )
        return pose

    def apply_current_tool_orientation(self, pose):
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

    def apply_current_orientation_with_target_yaw(self, pose):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ik_link_name,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.5)
            )

            current_q = trans.transform.rotation
            current_rot = R.from_quat([
                current_q.x,
                current_q.y,
                current_q.z,
                current_q.w
            ])

            current_yaw = self.rotation_to_yaw(current_rot)

            target_q = pose.orientation
            target_rot = R.from_quat([
                target_q.x,
                target_q.y,
                target_q.z,
                target_q.w
            ])

            target_yaw = self.rotation_to_yaw(target_rot)
            target_yaw = self.normalize_angle(target_yaw + self.grasp_yaw_offset_rad)

            yaw_delta = self.normalize_angle(target_yaw - current_yaw)

            desired_rot = R.from_euler("z", yaw_delta) * current_rot
            desired_q = desired_rot.as_quat()

            pose.orientation.x = float(desired_q[0])
            pose.orientation.y = float(desired_q[1])
            pose.orientation.z = float(desired_q[2])
            pose.orientation.w = float(desired_q[3])

            self.get_logger().info(
                f"Applied target yaw with offset while preserving tool attitude: "
                f"current_yaw={math.degrees(current_yaw):.1f} deg, "
                f"target_yaw_with_offset={math.degrees(target_yaw):.1f} deg, "
                f"offset={self.grasp_yaw_offset_deg:.1f} deg, "
                f"delta={math.degrees(yaw_delta):.1f} deg"
            )

        except Exception as e:
            self.get_logger().warn(
                f"Could not apply current orientation with target yaw. "
                f"Using input pose orientation. Error: {e}"
            )

        return pose

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def rotation_to_yaw(rotation):
        rot_mat = rotation.as_matrix()
        yaw = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        return float(yaw)

    def get_yaw_from_pose_msg(self, msg):
        try:
            q = msg.pose.orientation
            rot = R.from_quat([q.x, q.y, q.z, q.w])
            yaw = self.rotation_to_yaw(rot)
            return math.degrees(yaw)
        except Exception:
            return 0.0

    # ---------------------------------------------------------
    # IK + trajectory
    # ---------------------------------------------------------
    def move_robot(self, target_pose, move_time_sec=None):
        if move_time_sec is None:
            move_time_sec = self.default_arm_move_time

        if not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("/compute_ik service is not available.")
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.avoid_collisions = self.avoid_collisions

        req.ik_request.pose_stamped.header.frame_id = self.base_frame
        req.ik_request.pose_stamped.header.stamp = BuiltinTime(sec=0, nanosec=0)
        req.ik_request.pose_stamped.pose = target_pose
        req.ik_request.timeout = BuiltinDuration(sec=2, nanosec=0)

        if self.current_joint_state is not None:
            req.ik_request.robot_state.joint_state = self.current_joint_state

        self.get_logger().info(
            f"Requesting IK: "
            f"x={target_pose.position.x:.3f}, "
            f"y={target_pose.position.y:.3f}, "
            f"z={target_pose.position.z:.3f}, "
            f"move_time={move_time_sec:.1f}s"
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

        self.publish_arm_trajectory(
            ordered_positions,
            move_time_sec=move_time_sec
        )

        return ordered_positions

    def publish_arm_trajectory(self, joint_positions, move_time_sec=None):
        if move_time_sec is None:
            move_time_sec = self.default_arm_move_time

        msg = JointTrajectory()
        msg.header.stamp = BuiltinTime(sec=0, nanosec=0)
        msg.joint_names = self.arm_joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in joint_positions]

        sec = int(math.floor(move_time_sec))
        nanosec = int((move_time_sec - sec) * 1e9)

        point.time_from_start = BuiltinDuration(sec=sec, nanosec=nanosec)
        msg.points.append(point)

        self.arm_pub.publish(msg)

        self.get_logger().info(
            "Published arm trajectory: "
            + ", ".join([f"{p:.3f}" for p in joint_positions])
        )

    # ---------------------------------------------------------
    # Joint-state checking
    # ---------------------------------------------------------
    def angle_error(self, current, target):
        return abs(math.atan2(
            math.sin(current - target),
            math.cos(current - target)
        ))

    def get_current_joint_position(self, joint_name):
        if self.current_joint_state is None:
            return None

        for name, pos in zip(
            self.current_joint_state.name,
            self.current_joint_state.position
        ):
            if name == joint_name:
                return float(pos)

        return None

    def get_current_arm_state(self):
        if self.current_joint_state is None:
            return None

        name_to_pos = {
            name: pos
            for name, pos in zip(
                self.current_joint_state.name,
                self.current_joint_state.position
            )
        }

        positions = []

        for joint_name in self.arm_joint_names:
            if joint_name not in name_to_pos:
                return None

            positions.append(float(name_to_pos[joint_name]))

        return positions

    def wait_until_arm_reached(self, target_positions, move_time_sec, label="target"):
        timeout_sec = max(
            self.min_wait_timeout,
            move_time_sec * self.wait_timeout_scale
        )

        self.get_logger().info(
            f"Waiting for arm to reach {label}. "
            f"timeout={timeout_sec:.1f}s, tolerance={self.joint_goal_tolerance:.3f} rad"
        )

        start_time = time.time()
        stable_start_time = None
        last_report_time = 0.0

        while rclpy.ok():
            current_positions = self.get_current_arm_state()

            if current_positions is not None:
                errors = [
                    self.angle_error(c, t)
                    for c, t in zip(current_positions, target_positions)
                ]

                max_error = max(errors)
                position_ok = max_error < self.joint_goal_tolerance

                if position_ok:
                    if stable_start_time is None:
                        stable_start_time = time.time()

                    if time.time() - stable_start_time >= self.required_stable_time:
                        self.get_logger().info(
                            f"Arm reached {label}. Max error={max_error:.4f} rad"
                        )
                        return True
                else:
                    stable_start_time = None

                now = time.time()
                if now - last_report_time > 2.0:
                    last_report_time = now
                    self.get_logger().info(
                        f"Waiting for {label}... max_error={max_error:.4f} rad"
                    )

            if time.time() - start_time > timeout_sec:
                self.get_logger().warn(
                    f"Timed out waiting for arm to reach {label}."
                )
                return False

            time.sleep(0.05)

        return False

    def wait_until_gripper_motion_done(self, target_position, label="gripper"):
        joint_name = self.gripper_joint_names[0]

        self.get_logger().info(
            f"Waiting for gripper {label}. "
            f"target={target_position:.3f}, timeout={self.gripper_wait_timeout:.1f}s"
        )

        start_time = time.time()
        stable_start_time = None
        last_report_time = 0.0

        while rclpy.ok():
            current_position = self.get_current_joint_position(joint_name)

            if current_position is not None:
                error = abs(current_position - target_position)

                if label == "close":
                    elapsed = time.time() - start_time

                    if elapsed < self.min_close_wait_sec:
                        now = time.time()
                        if now - last_report_time > 1.0:
                            last_report_time = now
                            self.get_logger().info(
                                f"Force waiting for gripper close... "
                                f"elapsed={elapsed:.1f}/{self.min_close_wait_sec:.1f}s, "
                                f"current={current_position:.3f}, "
                                f"target={target_position:.3f}"
                            )
                        time.sleep(0.05)
                        continue

                    self.get_logger().info(
                        f"Minimum gripper close wait finished. "
                        f"current={current_position:.3f}, target={target_position:.3f}"
                    )
                    return True

                if error < self.gripper_position_tolerance:
                    if stable_start_time is None:
                        stable_start_time = time.time()

                    if time.time() - stable_start_time >= self.gripper_required_stable_time:
                        self.get_logger().info(
                            f"Gripper {label} reached. "
                            f"current={current_position:.3f}, target={target_position:.3f}"
                        )
                        return True
                else:
                    stable_start_time = None

                now = time.time()
                if now - last_report_time > 1.0:
                    last_report_time = now
                    self.get_logger().info(
                        f"Waiting gripper {label}... "
                        f"current={current_position:.3f}, "
                        f"target={target_position:.3f}, error={error:.3f}"
                    )

            if time.time() - start_time > self.gripper_wait_timeout:
                self.get_logger().warn(
                    f"Timed out waiting for gripper {label}. Continuing."
                )
                return False

            time.sleep(0.05)

        return False

    # ---------------------------------------------------------
    # Gripper
    # ---------------------------------------------------------
    def control_gripper(self, close):
        msg = JointTrajectory()
        msg.header.stamp = BuiltinTime(sec=0, nanosec=0)
        msg.joint_names = self.gripper_joint_names

        point = JointTrajectoryPoint()

        if close:
            target = self.gripper_close_position
        else:
            target = self.gripper_open_position

        point.positions = [float(target)]

        sec = int(math.floor(self.gripper_move_time_sec))
        nanosec = int((self.gripper_move_time_sec - sec) * 1e9)

        point.time_from_start = BuiltinDuration(sec=sec, nanosec=nanosec)
        msg.points.append(point)

        self.gripper_pub.publish(msg)

        if close:
            self.get_logger().info(
                f"Gripper close command: {target:.3f}, time={self.gripper_move_time_sec:.1f}s"
            )
        else:
            self.get_logger().info(
                f"Gripper open command: {target:.3f}, time={self.gripper_move_time_sec:.1f}s"
            )

    # ---------------------------------------------------------
    # Test and status
    # ---------------------------------------------------------
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

        self.publish_arm_trajectory(
            test_pose,
            move_time_sec=self.default_arm_move_time
        )

    def print_status(self):
        self.get_logger().info("=== GraspExecutor status ===")
        self.get_logger().info(f"orientation_mode: {self.orientation_mode}")
        self.get_logger().info(f"auto_execute_on_pose: {self.auto_execute_on_pose}")
        self.get_logger().info(f"avoid_collisions: {self.avoid_collisions}")
        self.get_logger().info(f"is_executing: {self.is_executing}")

        if self.latest_target_pose is None:
            self.get_logger().info("latest_target_pose: None")
        else:
            pose = self.latest_target_pose.pose
            self.get_logger().info(
                f"latest_target_pose: "
                f"x={pose.position.x:.3f}, "
                f"y={pose.position.y:.3f}, "
                f"z={pose.position.z:.3f}, "
                f"yaw={self.get_yaw_from_pose_msg(self.latest_target_pose):.1f} deg"
            )


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