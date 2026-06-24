"""A*+MPC planner bring-up for the Navigation / DLIO stack.

Spawns
------
  a_star_node          2.5D Gaussian-grid A* on /local_voxel_map/obstacles, pose
                       from /dlio/odom_node/odom; publishes /a_star/path.
  mpc_node             CasADi/IPOPT nonlinear MPC tracker; publishes a velocity
                       command as a Twist on /mpc/cmd_vel.
  cmd_vel_to_amo       (optional, bridge:=true) g1_sim_bridge node that forwards
                       /mpc/cmd_vel as {vx,vy,yaw} JSON to the AMO WebSocket
                       server (:8766). The AMO policy is not a ROS 2 process, so
                       this is how velocity reaches the gait.

Run (inside the ROS 2 / localization container, after DLIO + g1_local_map are up):

    ros2 launch a_star_mpc_planner planner.launch.py
    ros2 launch a_star_mpc_planner planner.launch.py bridge:=false   # planner only
    ros2 launch a_star_mpc_planner planner.launch.py amo_host:=127.0.0.1 amo_port:=8766

The whole stack runs on ROS_DOMAIN_ID=42 to match real_localization.launch.py and
isolate it from the ROS 2 Jazzy host (see docs/DLIO_G1_MID360_TUNING.md and the
QoS/transport notes in docs/LOCAL_VOXEL_MAP.md).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('a_star_mpc_planner'),
        'config', 'planner_params_default.yaml')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    bridge = LaunchConfiguration('bridge')
    amo_host = LaunchConfiguration('amo_host')
    amo_port = LaunchConfiguration('amo_port')
    ros_domain_id = LaunchConfiguration('ros_domain_id')
    rviz = LaunchConfiguration('rviz')

    common = [params_file, {'use_sim_time': use_sim_time}]

    a_star_node = Node(
        package='a_star_mpc_planner',
        executable='a_star_node',
        name='a_star_node',
        output='screen',
        parameters=common,
    )

    mpc_node = Node(
        package='a_star_mpc_planner',
        executable='mpc_node',
        name='mpc_node',
        output='screen',
        parameters=common,
    )

    # Forward the MPC's Twist to the AMO WebSocket gait. cmd_vel_topic points at
    # /mpc/cmd_vel (the same bridge also serves teleop on /cmd_vel — run only one
    # source at a time). Caps are set at/above the MPC velocity envelope so they
    # never clip a valid MPC command; AMO applies its own internal limits.
    cmd_vel_to_amo = Node(
        package='g1_sim_bridge',
        executable='cmd_vel_to_amo_node',
        name='cmd_vel_to_amo',
        output='screen',
        condition=IfCondition(bridge),
        parameters=[{
            'cmd_vel_topic': '/mpc/cmd_vel',
            'amo_host': amo_host,
            'amo_port': amo_port,
            'rate_hz': 20.0,
            'max_forward_vel': 0.5,
            'max_lateral_vel': 0.1,
            'max_yaw_rate': 0.8,
            'use_sim_time': use_sim_time,
        }],
    )

    # Optional RViz, OFF by default. real_localization.launch.py already opens
    # the shared g1_dlio.rviz (which now includes the planner displays), so leave
    # this false when running the full stack to avoid two RViz windows. Set
    # rviz:=true only when running the planner STANDALONE for visualization.
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_planner',
        output='screen',
        condition=IfCondition(rviz),
        arguments=['-d', PathJoinSubstitution(
            [FindPackageShare('g1_bringup'), 'rviz', 'g1_dlio.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='YAML parameter file for a_star_node + mpc_node.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock (true in sim, false on the real robot).'),
        DeclareLaunchArgument(
            'bridge', default_value='true',
            description='Also run g1_sim_bridge/cmd_vel_to_amo_node to forward '
                        '/mpc/cmd_vel to the AMO WebSocket (:8766).'),
        DeclareLaunchArgument(
            'amo_host', default_value='127.0.0.1',
            description='Host of the AMO WebSocket server (amo_inference, :8766).'),
        DeclareLaunchArgument(
            'amo_port', default_value='8766',
            description='Port of the AMO WebSocket server.'),
        DeclareLaunchArgument(
            'ros_domain_id', default_value='42',
            description='DDS domain, matching real_localization.launch.py, to '
                        'isolate the stack from the ROS 2 Jazzy host.'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Open RViz (g1_bringup g1_dlio.rviz) for standalone '
                        'planner viz. Leave false when real_localization already '
                        'runs RViz, to avoid two windows.'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', ros_domain_id),
        a_star_node,
        mpc_node,
        cmd_vel_to_amo,
        rviz_node,
    ])
