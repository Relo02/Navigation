"""Launch the local rolling 3D voxel map node.

Standalone:
    ros2 launch g1_local_map local_map.launch.py

Normally started by g1_bringup/real_localization.launch.py alongside DLIO.
Override params on the CLI, e.g.:
    ros2 launch g1_local_map local_map.launch.py voxel_size:=0.05 half_width:=6.0
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('g1_local_map'), 'config', 'local_map.yaml')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='YAML parameter file for local_voxel_map_node.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock (true in sim, false on the real robot).'),
        Node(
            package='g1_local_map',
            executable='local_voxel_map_node',
            name='local_voxel_map',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
    ])
