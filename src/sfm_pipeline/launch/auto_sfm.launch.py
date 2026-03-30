from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ur5e_moveit_config'),
                'launch',
                'demo.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'use_rviz': 'false'
        }.items()
    )

    auto_scanner_node = Node(
        package='sfm_pipeline',
        executable='auto_scanner',
        name='auto_scanner',
        output='screen'
    )

    return LaunchDescription([
        demo_launch,
        auto_scanner_node
    ])