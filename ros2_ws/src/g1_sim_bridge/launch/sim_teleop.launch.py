"""Sim teleop + full localization in one shot.

Brings up sim_localization (DLIO + RViz + robot model) plus a way to send
velocity references to the Isaac AMO gait. g1_sim_scene.py --stabilize
subscribes /cmd_vel directly, so no WebSocket bridge is needed in sim:

    keyboard / gamepad ──/cmd_vel──> Isaac AMO stabilizer

The Unitree pad pairs with the REAL robot, so in sim the keyboard is usually the
easier option (teleop:=keyboard, the default).

    ./sim/launch_g1_sim.sh --stabilize                          # host
    ros2 launch g1_sim_bridge sim_teleop.launch.py              # keyboard (default)
    ros2 launch g1_sim_bridge sim_teleop.launch.py teleop:=joystick   # gamepad on js0
    ros2 launch g1_sim_bridge sim_teleop.launch.py teleop:=none       # just localization

teleop:=keyboard does NOT spawn the keyboard node (it needs its own interactive
TTY, which a launch file can't give it) -- run it yourself in a second terminal:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard       # i / j / k / l / ,
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    joy_dev = LaunchConfiguration('joy_dev')
    deadman = LaunchConfiguration('deadman_button')

    sim_loc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('g1_sim_bridge'), '/launch/sim_localization.launch.py']),
    )

    # Gamepad path (teleop:=joystick).
    joystick = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('g1_sim_bridge'), '/launch/joystick.launch.py']),
        launch_arguments={'joy_dev': joy_dev, 'deadman_button': deadman}.items(),
        condition=LaunchConfigurationEquals('teleop', 'joystick'))

    # Keyboard path (teleop:=keyboard): teleop_twist_keyboard needs an interactive
    # TTY, so it is run separately -- just remind the user.
    keyboard_hint = LogInfo(
        condition=LaunchConfigurationEquals('teleop', 'keyboard'),
        msg="[sim_teleop] keyboard mode: run in another terminal -> "
            "ros2 run teleop_twist_keyboard teleop_twist_keyboard")

    return LaunchDescription([
        DeclareLaunchArgument('teleop', default_value='keyboard',
                              description='keyboard | joystick | none'),
        DeclareLaunchArgument('joy_dev', default_value='0',
                              description='joy_node device id (0 -> /dev/input/js0)'),
        DeclareLaunchArgument('deadman_button', default_value='-1',
                              description='Gamepad button to hold for motion (-1 = always on)'),
        sim_loc,
        joystick,
        keyboard_hint,
    ])
