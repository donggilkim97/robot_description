import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Define the path to your custom xacro file
    pkg_share = get_package_share_directory('ur5e_gripper_setup')
    xacro_file = os.path.join(pkg_share, 'urdf', 'ur5e_with_robotiq.urdf.xacro')

    # 2. Use Command to process the xacro file into a standard URDF string
    robot_desc = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    # 3. Robot State Publisher Node (publishes tf and robot_description)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    # 4. Joint State Publisher GUI Node (provides sliders to move the robot)
    jsp_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    # 5. RViz2 Node (for 3D visualization)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen'
    )

    return LaunchDescription([
        rsp_node,
        jsp_gui_node,
        rviz_node
    ])
