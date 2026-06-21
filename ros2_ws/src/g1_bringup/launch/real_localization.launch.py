"""Real-robot localization bring-up (DLIO + Livox MID-360).

The real-robot counterpart of g1_sim_bridge/sim_localization.launch.py. The stock
Livox SDK driver feeds DLIO directly, except the IMU is rescaled to m/s^2 first:

    livox_ros_driver2 --(/livox/lidar)--> ground_removal --(/livox/lidar_filtered)--> DLIO
    livox_ros_driver2 --(/livox/imu  Imu, g) --> imu_rescale --(/livox/imu_ms2,
                                                  m/s^2)-------------------> DLIO

(ground_removal is optional, arg ground_removal:=false -> DLIO reads raw /livox/lidar)

Key real-robot specifics:
  * xfer_format=0 -> the driver emits a PointCloud2 (PointXYZRTLT) with per-point
    timestamps, which DLIO auto-detects as SensorType::LIVOX and deskews.
  * The MID-360 IMU reports linear acceleration in g (~1.0 at rest); DLIO expects
    m/s^2. imu_rescale (g1_sim_bridge) multiplies accel by 9.80665 -> /livox/imu_ms2,
    else DLIO's gravity removal diverges and crashes.
  * The MID-360 is mounted upside-down. MID360_config.json applies roll:180 to
    the CLOUD (upright); the IMU stays inverted, corrected by DLIO via
    extrinsics/baselink2imu/R = R_x(180) in dlio_mid360_real.yaml -- no driver patch.
  * DLIO must calibrate its IMU + gravity over the first ~3 s, so keep the robot
    STATIONARY at startup (e.g. during the AMO stabilize hold).

The whole stack runs on a dedicated ROS_DOMAIN_ID (default 42) to isolate it from
the ROS 2 Jazzy host: a Jazzy participant on the same domain corrupts CycloneDDS
deserialization of the large deskewed PointCloud2 (serdata.cpp:384 "invalid data
size"), so the cloud silently never arrives. For in-container debugging, export
the SAME domain first:  export ROS_DOMAIN_ID=42

Run (inside the localization container, ws sourced, robot powered + on-network):

    ros2 launch g1_bringup real_localization.launch.py
    ros2 launch g1_bringup real_localization.launch.py rviz:=false robot_model:=false

See docs/DLIO_DEPLOYMENT_TESTING.md (phases 5-7) and docs/DLIO_G1_MID360_TUNING.md.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz = LaunchConfiguration('rviz')
    robot_model = LaunchConfiguration('robot_model')
    livox_config = LaunchConfiguration('livox_config')
    start_map = LaunchConfiguration('start_map')
    local_map = LaunchConfiguration('local_map')
    ground_removal = LaunchConfiguration('ground_removal')
    ros_domain_id = LaunchConfiguration('ros_domain_id')

    default_livox_config = os.path.join(
        get_package_share_directory('livox_ros_driver2'),
        'config', 'MID360_config.json')

    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='true', description='Open RViz (g1_dlio.rviz)')
    declare_robot_model = DeclareLaunchArgument(
        'robot_model', default_value='true',
        description='Show the G1 robot model (RSP + JSP + base_link->pelvis TF)')
    declare_livox_config = DeclareLaunchArgument(
        'livox_config', default_value=default_livox_config,
        description='Livox MID360_config.json (host/lidar IPs + roll:180 mount).')
    declare_start_map = DeclareLaunchArgument(
        'start_map', default_value='true',
        description='Also run dlio_map_node (accumulated /dlio/map_node/map).')
    declare_local_map = DeclareLaunchArgument(
        'local_map', default_value='true',
        description='Run g1_local_map (ground-removed rolling voxel map for the planner).')
    declare_ground_removal = DeclareLaunchArgument(
        'ground_removal', default_value='true',
        description='Run ground_removal between the driver and DLIO so DLIO consumes a '
                    'ground-removed cloud (/livox/lidar_filtered). NOTE: removing the '
                    'ground before a LiDAR-inertial odometry can degrade it (loses the '
                    'pitch/roll/Z constraint) — set false to feed DLIO the raw cloud.')
    declare_domain = DeclareLaunchArgument(
        'ros_domain_id', default_value='42',
        description='Dedicated DDS domain for the whole stack, isolating it from '
                    'the ROS 2 Jazzy host (and anything on domain 0). A Jazzy '
                    'participant on the same domain corrupts CycloneDDS '
                    'deserialization of the large PointCloud2 (serdata.cpp:384). '
                    'Use the same ROS_DOMAIN_ID for any in-container debugging.')

    # Livox MID-360 driver in PointCloud2 mode (xfer_format=0) for DLIO. Mirrors
    # the driver's own msg_MID360_launch.py params, but xfer_format 1 -> 0.
    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[
            {'xfer_format': 0},        # 0 = PointCloud2 (PointXYZRTLT) for DLIO
            {'multi_topic': 0},
            {'data_src': 0},
            {'publish_freq': 10.0},
            {'output_data_type': 0},
            {'frame_id': 'livox_frame'},
            {'lvx_file_path': '/home/livox/livox_test.lvx'},
            {'user_config_path': livox_config},
            {'cmdline_input_bd_code': 'livox0000000001'},
        ],
    )

    # The Livox MID-360 IMU reports linear acceleration in g; DLIO needs m/s^2.
    # Rescale (x9.80665) into /livox/imu_ms2 before DLIO, or its gravity removal
    # diverges and crashes. Angular velocity (rad/s) is passed through.
    imu_rescale = Node(
        package='g1_sim_bridge',
        executable='imu_rescale_node',
        name='imu_rescale',
        output='screen',
        parameters=[{
            'input_topic': '/livox/imu',
            'output_topic': '/livox/imu_ms2',
            'accel_scale': 9.80665,
        }],
    )

    # Optional ground-removal preprocessor: /livox/lidar -> /livox/lidar_filtered.
    # Removes the ground plane per scan (RANSAC) while preserving every point
    # field (incl. per-point timestamp DLIO deskews with). See ground_removal arg.
    ground_removal_node = Node(
        package='g1_local_map',
        executable='ground_removal_node',
        name='ground_removal',
        output='screen',
        parameters=[{
            'input_topic': '/livox/lidar',
            'output_topic': '/livox/lidar_filtered',
        }],
        condition=IfCondition(ground_removal),
    )

    # DLIO reads the filtered cloud when ground_removal is on, else the raw cloud.
    dlio_pointcloud_topic = PythonExpression([
        "'/livox/lidar_filtered' if '", ground_removal, "' == 'true' else '/livox/lidar'"])

    # Shared DLIO node bring-up, real variant: read the driver topics, real config.
    dlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('g1_sim_bridge'), '/launch/dlio.launch.py']),
        launch_arguments={
            'use_sim_time': 'false',
            'config_file': 'dlio_mid360_real.yaml',
            'pointcloud_topic': dlio_pointcloud_topic,
            'imu_topic': '/livox/imu_ms2',
            'start_map': start_map,
        }.items(),
    )

    # Local rolling voxel map: ground-removed obstacle cloud + costmap from the
    # DLIO deskewed scan, to feed the a_star_mpc planner (and RViz). Updates every
    # scan, unlike the keyframe-driven /dlio/map_node/map. Run as a direct Node
    # (NOT an include) so it reliably inherits the SetEnvironmentVariable domain
    # below — an included launch can start on the default domain 0 instead.
    local_map_params = os.path.join(
        get_package_share_directory('g1_local_map'), 'config', 'local_map.yaml')
    local_map_node = Node(
        package='g1_local_map',
        executable='local_voxel_map_node',
        name='local_voxel_map',
        output='screen',
        parameters=[local_map_params, {'use_sim_time': False}],
        condition=IfCondition(local_map),
    )

    robot_model_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('g1_bringup'), '/launch/robot_model.launch.py']),
        launch_arguments={'use_sim_time': 'false'}.items(),
        condition=IfCondition(robot_model),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_dlio',
        arguments=['-d', PathJoinSubstitution(
            [FindPackageShare('g1_bringup'), 'rviz', 'g1_dlio.rviz'])],
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(rviz),
        output='screen',
    )

    return LaunchDescription([
        declare_rviz,
        declare_robot_model,
        declare_livox_config,
        declare_start_map,
        declare_local_map,
        declare_ground_removal,
        declare_domain,
        # Pin the DDS domain for every node this launch spawns (must precede them).
        # Isolates the stack from the ROS 2 Jazzy host (domain 0), whose
        # cross-distro participants corrupt CycloneDDS deserialization of the
        # large PointCloud2. NB: do NOT also set ROS_LOCALHOST_ONLY=1 — with
        # CycloneDDS that disables multicast and caps the domain at ~10
        # participant indices, so this 9-node stack fails with "Failed to find a
        # free participant index". Large-cloud delivery is handled instead by the
        # best-effort readers (RViz CloudRegistered + local_voxel_map cloud sub).
        SetEnvironmentVariable('ROS_DOMAIN_ID', ros_domain_id),
        livox_driver,
        imu_rescale,
        ground_removal_node,
        dlio,
        local_map_node,
        robot_model_launch,
        rviz_node,
    ])
