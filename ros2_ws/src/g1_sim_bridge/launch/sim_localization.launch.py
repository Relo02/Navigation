"""Simulation localization bring-up (DLIO).

Starts the Isaac->DLIO QoS relay and (optionally) DLIO itself, configured for
simulation:

    Isaac Sim --(/livox/lidar    PointCloud2 BEST_EFFORT)--> relay --(/livox/lidar_reliable RELIABLE)--> DLIO
    Isaac Sim --(/livox/imu_raw  Imu        BEST_EFFORT)--> relay --(/livox/imu             RELIABLE)--> DLIO

DLIO consumes the PointCloud2 + Imu directly -- no Livox CustomMsg conversion is
needed (that was a FAST-LIO requirement). The DLIO nodes themselves are brought
up by the shared dlio.launch.py (config_file=dlio_sim.yaml, use_sim_time:=true);
this file just adds the sim-only QoS relay in front of it.

DLIO publishes /dlio/odom_node/{odom,pose,path,pointcloud/deskewed} and
/dlio/map_node/map, and broadcasts TF odom -> base_link -> {livox, livox_imu}.

Run Isaac first (Navigation/sim/launch_g1_sim.sh), then:

    ros2 launch g1_sim_bridge sim_localization.launch.py
    ros2 launch g1_sim_bridge sim_localization.launch.py start_dlio:=false   # relay only
    ros2 launch g1_sim_bridge sim_localization.launch.py rviz:=false         # no RViz

RViz opens by default with the shared g1_dlio.rviz config (from g1_bringup),
with use_sim_time=true so it follows Isaac's /clock.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_dlio = LaunchConfiguration('start_dlio')
    use_sim_time = LaunchConfiguration('use_sim_time')
    stride = LaunchConfiguration('stride')
    rviz = LaunchConfiguration('rviz')
    robot_model = LaunchConfiguration('robot_model')

    declare_start_dlio = DeclareLaunchArgument(
        'start_dlio', default_value='true',
        description='Also launch DLIO (dlio_odom_node + dlio_map_node) with sim params')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='DLIO consumes Isaac /clock when true')
    declare_stride = DeclareLaunchArgument(
        'stride', default_value='1',
        description='Relay decimation: keep every Nth cloud point (1 = all)')
    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz with the shared g1_dlio.rviz config (use_sim_time=true)')
    declare_robot_model = DeclareLaunchArgument(
        'robot_model', default_value='true',
        description='Publish the G1 URDF model (robot_state_publisher) so RViz shows the robot')

    relay_node = Node(
        package='g1_sim_bridge',
        executable='isaac_dlio_qos_relay_node',
        name='isaac_dlio_qos_relay',
        output='screen',
        parameters=[{
            'input_topic': '/livox/lidar',
            'output_topic': '/livox/lidar_reliable',
            'imu_in_topic': '/livox/imu_raw',
            'imu_out_topic': '/livox/imu',
            'stride': stride,
            'use_sim_time': use_sim_time,
        }],
    )

    # Shared DLIO node bring-up, sim variant: read the relay outputs, sim config.
    dlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('g1_sim_bridge'), '/launch/dlio.launch.py']),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'config_file': 'dlio_sim.yaml',
            'pointcloud_topic': '/livox/lidar_reliable',
            'imu_topic': '/livox/imu',
        }.items(),
        condition=IfCondition(start_dlio),
    )

    # RViz with the shared DLIO view (g1_bringup/rviz/g1_dlio.rviz). Sim variant
    # forces use_sim_time=true so displays follow Isaac's /clock (the real-robot
    # launch sets it false). FindPackageShare resolves g1_bringup from the same
    # workspace -- no build dependency needed for a launch substitution.
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_dlio',
        arguments=['-d', PathJoinSubstitution(
            [FindPackageShare('g1_bringup'), 'rviz', 'g1_dlio.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz),
        output='screen',
    )

    # G1 URDF model -> /robot_description + link TFs, riding DLIO's base_link
    # pose. Isaac publishes the live /joint_states (g1_sim_scene.py), so the
    # neutral joint_state_publisher is disabled here.
    robot_model_inc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('g1_bringup'), '/launch/robot_model.launch.py']),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_joint_state_publisher': 'false',
        }.items(),
        condition=IfCondition(robot_model),
    )

    return LaunchDescription([
        declare_start_dlio,
        declare_use_sim_time,
        declare_stride,
        declare_rviz,
        declare_robot_model,
        relay_node,
        dlio,
        rviz_node,
        robot_model_inc,
    ])
