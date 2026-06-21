"""Unitree G1 gamepad -> /cmd_vel (reusable building block).

    joy_node ──/joy──> joy_to_cmdvel ──/cmd_vel──>  (sim stabilizer | cmd_vel_to_amo)

Included by real_teleop.launch.py and sim_teleop.launch.py; can also be run on
its own (e.g. backgrounded by launch_g1_sim.sh / run_amo.sh).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    joy_dev = LaunchConfiguration('joy_dev')
    deadman = LaunchConfiguration('deadman_button')

    return LaunchDescription([
        DeclareLaunchArgument('joy_dev', default_value='0',
                              description='joy_node device id (0 -> /dev/input/js0)'),
        DeclareLaunchArgument('deadman_button', default_value='-1',
                              description='Gamepad button to hold for motion (-1 = always on)'),
        Node(package='joy', executable='joy_node', name='joy_node',
             parameters=[{'device_id': joy_dev, 'autorepeat_rate': 20.0, 'deadzone': 0.05}],
             output='screen'),
        Node(package='g1_sim_bridge', executable='joy_to_cmdvel_node', name='joy_to_cmdvel',
             parameters=[{'enable_button': deadman}], output='screen'),
    ])
