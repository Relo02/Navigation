"""Real-robot joystick teleop: Unitree G1 gamepad -> AMO velocity command.

    joy_node ──/joy──> joy_to_cmdvel ──/cmd_vel──> cmd_vel_to_amo ──ws:8766──> amo_inference

Run amo_inference with command.source=websocket first (JOYSTICK=1 ./run_amo.sh),
then in the localization container:

    ros2 launch g1_sim_bridge real_teleop.launch.py            # gamepad on /dev/input/js0
    ros2 launch g1_sim_bridge real_teleop.launch.py amo_host:=192.168.123.164

All services use host networking, so amo_host defaults to 127.0.0.1.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    joystick = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('g1_sim_bridge'), '/launch/joystick.launch.py']),
        launch_arguments={'joy_dev': LaunchConfiguration('joy_dev'),
                          'deadman_button': LaunchConfiguration('deadman_button')}.items(),
    )
    bridge = Node(
        package='g1_sim_bridge', executable='cmd_vel_to_amo_node', name='cmd_vel_to_amo',
        parameters=[{'amo_host': LaunchConfiguration('amo_host'),
                     'amo_port': LaunchConfiguration('amo_port')}],
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('joy_dev', default_value='0'),
        DeclareLaunchArgument('deadman_button', default_value='-1',
                              description='Gamepad button to hold for motion (-1 = always on)'),
        DeclareLaunchArgument('amo_host', default_value='127.0.0.1',
                              description='Host of the amo_inference WebSocket server'),
        DeclareLaunchArgument('amo_port', default_value='8766'),
        joystick,
        bridge,
    ])
