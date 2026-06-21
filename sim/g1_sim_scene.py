#!/usr/bin/env python3
"""Standalone Isaac Sim scene: Unitree G1 + Livox MID-360 LiDAR + IMU, published
to ROS 2 for the DLIO (direct_lidar_inertial_odometry) localization stack.

This script publishes (consumed by the g1_sim_bridge QoS relay, see below):

    /livox/lidar     sensor_msgs/PointCloud2   frame_id = livox_frame  (RTX lidar)
    /livox/imu_raw   sensor_msgs/Imu           frame_id = livox_frame
    /clock           rosgraph_msgs/Clock

DLIO consumes a plain PointCloud2 + Imu directly (no Livox CustomMsg needed),
but its cloud subscriber is RELIABLE while Isaac's ROS 2 bridge only publishes
BEST_EFFORT. The companion ROS 2 node `g1_sim_bridge` (Humble workspace) is a
thin QoS relay: /livox/lidar -> /livox/lidar_reliable and /livox/imu_raw ->
/livox/imu (both RELIABLE) so DLIO's subscribers match.

Run via launch_g1_sim.sh (which sets RMW_IMPLEMENTATION=rmw_cyclonedds_cpp so it
talks to the rest of the navigation stack), or directly:

    ./isaac-sim/python.sh g1_sim_scene.py [--headless] [--usd PATH] ...
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# CLI -- parsed before SimulationApp so --headless can be honoured.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# The G1 + full-warehouse stage now lives inside this repo (sim/usd/). It is a
# thin wrapper: /World/g1 is a *payload* to the cloud G1 robot
# (.../Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd) and /World/full_warehouse pulls
# the warehouse environment from the Isaac cloud assets -- so the .usd file is
# self-contained (no local sibling assets) and carries NO embedded ROS graphs;
# this script adds the MID-360 lidar + IMU + ROS publishers itself.
# Override with --usd or the ISAAC_G1_STAGE env var.
_DEFAULT_USD = os.environ.get(
    "ISAAC_G1_STAGE",
    os.path.join(_HERE, "usd", "robot_full_warehouse.usd"),
)

parser = argparse.ArgumentParser(description="G1 + MID-360 + IMU -> ROS 2 sim")
parser.add_argument("--usd", default=os.path.normpath(_DEFAULT_USD),
                    help="Stage to open (default: the G1 warehouse stage)")
parser.add_argument("--mount-path", default="/World/g1/torso_link/mid360_link",
                    help="Prim the LiDAR + IMU are parented to")
parser.add_argument("--lidar-config", default="Livox_Mid360",
                    help="RTX lidar config name (file in lidar_configs/)")
parser.add_argument("--lidar-topic", default="/livox/lidar")
# IMU goes to an intermediate topic; g1_sim_bridge republishes it RELIABLE on
# /livox/imu (Isaac's bridge publishes BEST_EFFORT; the relay makes it RELIABLE
# so any reliable consumer matches -- DLIO's IMU sub itself is BEST_EFFORT).
parser.add_argument("--imu-topic", default="/livox/imu_raw")
parser.add_argument("--frame-id", default="livox_frame")
parser.add_argument("--clock-topic", default="/clock")
parser.add_argument("--joint-states-topic", default="/joint_states",
                    help="G1 articulation joint angles -> URDF robot_state_publisher in RViz")
# The G1 is unactuated in this scene -- with physics on and no controller it
# sags/collapses under gravity and DLIO tracks the falling sensor (Z drifts).
# --hold-pose pins the pelvis to the world (fixed base) so the robot stays put
# for a clean static localization test. Turn OFF once AMO drives the joints.
parser.add_argument("--hold-pose", action="store_true",
                    help="Pin the pelvis to the world (fixed base) so the unactuated G1 doesn't collapse")
parser.add_argument("--stabilize", action="store_true",
                    help="Run the Navigation AMO policy (amo/AmoDeployment) so the G1 actively "
                         "stands and stays still. Mutually exclusive with --hold-pose.")
parser.add_argument("--robot-prim", default="/World/g1",
                    help="Articulation root prim for --stabilize (the G1 articulation)")
parser.add_argument("--robot-root", default="/World/g1/pelvis",
                    help="Robot root link prim to pin when --hold-pose is set")
# MID-360 is mounted upside-down on the real G1, but the livox driver already
# rotates cloud+IMU upright at the source, so the rest of the stack sees an
# upright sensor. We therefore mount it upright in sim too (identity by
# default); override if your mid360_link frame needs an extra rotation.
parser.add_argument("--lidar-rpy-deg", default="0,0,0",
                    help="Extra roll,pitch,yaw (deg) applied to the sensor at the mount")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

# enable_motion_bvh: transform each lidar ray by the sensor pose at that ray's
# sample time, not one pose per frame. Without it a moving/walking robot smears
# the world-frame cloud (the "MotionBVH for lidar model not enabled" warning).
simulation_app = SimulationApp({"headless": args.headless, "enable_motion_bvh": True})

# ---------------------------------------------------------------------------
# Imports that only exist inside a running Kit app.
# ---------------------------------------------------------------------------
import carb  # noqa: E402
import omni  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.prims import set_targets  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def _find_articulation_root(stage, start_path: str, fallback: str) -> str:
    """The robot payload root (e.g. /World/g1) is usually a plain Xform; the
    PhysX articulation lives on a child prim. The ROS2 joint-state publisher and
    the tensor articulation view need that exact root, else they spam
    'did not match any articulations'. Walk the subtree for the ArticulationRoot
    API and return its path (fallback to start_path if none found)."""
    start = stage.GetPrimAtPath(start_path)
    if start and start.IsValid():
        for prim in Usd.PrimRange(start):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(prim.GetPath())
    return fallback


def _find_prim_by_name(stage, start_path: str, name: str, fallback: str) -> str:
    """Find a prim by leaf name under start_path. The cloud G1 USD lays the link
    prims out flat under the robot root (/World/g1/mid360_link), not nested like
    the URDF (/World/g1/torso_link/mid360_link), so the default mount path misses
    and the sensor would fall back to an empty Xform at the torso origin."""
    start = stage.GetPrimAtPath(start_path)
    if start and start.IsValid():
        for prim in Usd.PrimRange(start):
            if prim.GetName() == name:
                return str(prim.GetPath())
    return fallback

# RTX lidar + ROS 2 bridge + physics IMU sensor.
for _ext in ("isaacsim.sensors.rtx", "isaacsim.sensors.physics", "isaacsim.ros2.bridge"):
    enable_extension(_ext)
simulation_app.update()

# ---------------------------------------------------------------------------
# Register our lidar_configs/ folder so config="Livox_Mid360" resolves.
# ---------------------------------------------------------------------------
_LIDAR_CFG_DIR = os.path.join(_HERE, "lidar_configs") + "/"
_settings = carb.settings.get_settings()
_PROFILE_KEY = "/app/sensors/nv/lidar/profileBaseFolder"
_folders = list(_settings.get(_PROFILE_KEY) or [])
if _LIDAR_CFG_DIR not in _folders:
    _folders.append(_LIDAR_CFG_DIR)
    _settings.set_string_array(_PROFILE_KEY, _folders)
carb.log_warn(f"[g1_sim] lidar config folder registered: {_LIDAR_CFG_DIR}")

# Per-ray motion compensation for the lidar (belt-and-suspenders with the
# SimulationApp enable_motion_bvh flag above).
_settings.set_bool("/rtx/sceneOptimizationBVH/enableMotion", True)

# ---------------------------------------------------------------------------
# Open the G1 stage.
# ---------------------------------------------------------------------------
if not os.path.isfile(args.usd):
    carb.log_error(f"[g1_sim] stage not found: {args.usd}")
    simulation_app.close()
    sys.exit(1)

ctx = omni.usd.get_context()
ctx.open_stage(args.usd)
simulation_app.update()
stage = ctx.get_stage()
carb.log_warn(f"[g1_sim] opened stage: {args.usd}")

# Resolve the real articulation root under the robot payload (used by the joint
# -state publisher and the --stabilize tensor articulation view).
robot_artic = _find_articulation_root(stage, args.robot_prim, args.robot_prim)
carb.log_warn(f"[g1_sim] articulation root: {robot_artic} (robot_prim={args.robot_prim})")

# ---------------------------------------------------------------------------
# Make sure the mount prim exists (it lives inside the G1 payload; create an
# Xform fallback so the script also works on stages without it).
# ---------------------------------------------------------------------------
# Mount the lidar + IMU at the REAL mid360_link prim (head height, upright). The
# cloud G1 USD has flat link prims, so the nested --mount-path default misses and
# the sensor would otherwise fall back to an empty Xform at the torso origin
# (chest) -- boxed in by the arms/head, where it mostly scans the robot itself.
mount_path = _find_prim_by_name(stage, args.robot_prim, "mid360_link", args.mount_path)
if not stage.GetPrimAtPath(mount_path).IsValid():
    carb.log_warn(f"[g1_sim] mount {mount_path} missing -> creating Xform fallback "
                  f"(lidar may be mis-placed; check the G1 USD link layout)")
    UsdGeom.Xform.Define(stage, mount_path)
else:
    carb.log_warn(f"[g1_sim] lidar/IMU mount prim: {mount_path}")

# Extra orientation at the mount (XYZW order from roll,pitch,yaw).
_r, _p, _y = (float(v) for v in args.lidar_rpy_deg.split(","))
_q = (
    Gf.Rotation(Gf.Vec3d(0, 0, 1), _y)
    * Gf.Rotation(Gf.Vec3d(0, 1, 0), _p)
    * Gf.Rotation(Gf.Vec3d(1, 0, 0), _r)
).GetQuat()
sensor_orient = Gf.Quatd(_q.GetReal(), _q.GetImaginary())

# ---------------------------------------------------------------------------
# RTX LiDAR (Livox MID-360 approximation) -> PointCloud2.
# ---------------------------------------------------------------------------
lidar_path = mount_path + "/mid360_lidar"
# Create the MID-360 as a modern OmniLidar via the replicator path: pass the
# custom JSON profile by NAME in `config`. The profile resolves from the
# `app.sensors.nv.lidar.profileBaseFolder` search folders (registered above +
# symlinked into Isaac's data/lidar_configs by launch_g1_sim.sh), and the lidar
# core plugin builds the 64-emitter MID-360 scan model from it -- a probe of
# this exact call rendered ~40k points/frame against test geometry.
#
# Do NOT pass `sensorModelConfig` as a kwarg (the OmniLidar schema has no such
# attribute -> the command raises "No attribute 'sensorModelConfig'" and aborts)
# and do NOT use force_camera_prim=True: the deprecated camera-prim path renders
# almost nothing here (~379 pts vs 40k). The "Config '...' not found" warning
# that still prints is cosmetic -- it is the USD-asset lookup in _add_reference,
# which custom JSON profiles always miss before the replicator path builds them.
_, lidar_prim = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path=lidar_path,
    parent=None,
    config=args.lidar_config,
    translation=(0.0, 0.0, 0.0),
    orientation=sensor_orient,
)
if lidar_prim is None or not lidar_prim.IsValid():
    carb.log_error(f"[g1_sim] failed to create MID-360 RTX lidar at {lidar_path}")
    simulation_app.close()
    sys.exit(1)
carb.log_warn(f"[g1_sim] MID-360 OmniLidar created at {lidar_path} (config={args.lidar_config})")

# IsaacSensorCreateRtxLidar sets skipDroppingInvalidPoints=True, which KEEPS
# no-return rays in the published cloud (as non-finite / zero points). The real
# MID-360 driver emits only valid returns; DLIO chokes on the invalid ones
# (deskewed points: 0 -> 'free(): invalid next size' heap crash). Force it off so
# the sim cloud contains only real hits, like the hardware.
_skip = lidar_prim.GetAttribute("omni:sensor:Core:skipDroppingInvalidPoints")
if _skip and _skip.IsValid():
    _skip.Set(False)
    carb.log_warn("[g1_sim] skipDroppingInvalidPoints=False (drop no-return rays for DLIO)")
else:
    carb.log_warn("[g1_sim] WARNING: skipDroppingInvalidPoints attr not found on lidar prim")

lidar_render_product = rep.create.render_product(lidar_prim.GetPath(), [1, 1], name="mid360_rp")

pc_writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
pc_writer.initialize(topicName=args.lidar_topic, frameId=args.frame_id)
pc_writer.attach([lidar_render_product])

# A visible debug splat of the cloud in the viewport (no-op when headless).
try:
    dbg = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
    dbg.attach([lidar_render_product])
except Exception as exc:  # pragma: no cover - cosmetic only
    carb.log_warn(f"[g1_sim] debug draw unavailable: {exc}")

carb.log_warn(f"[g1_sim] MID-360 lidar @ {lidar_path} -> {args.lidar_topic} ({args.frame_id})")

# ---------------------------------------------------------------------------
# IMU sensor (co-located with the lidar) -> sensor_msgs/Imu via OmniGraph.
# ---------------------------------------------------------------------------
imu_path = mount_path + "/mid360_imu"
omni.kit.commands.execute(
    "IsaacSensorCreateImuSensor",
    path=imu_path,
    parent=None,
    translation=Gf.Vec3d(0.0, 0.0, 0.0),
    orientation=sensor_orient,
)

GRAPH = "/G1SimGraph"
og.Controller.edit(
    {"graph_path": GRAPH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
            ("PublishIMU", "isaacsim.ros2.bridge.ROS2PublishImu"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            # Joint angles of the G1 articulation -> /joint_states, so the URDF
            # robot_state_publisher (g1_bringup robot_model.launch.py) shows the
            # actual sim pose in RViz instead of a neutral one.
            ("PublishJoints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "ReadIMU.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishJoints.inputs:execIn"),
            ("ReadIMU.outputs:execOut", "PublishIMU.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishIMU.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishJoints.inputs:timeStamp"),
            ("ReadIMU.outputs:linAcc", "PublishIMU.inputs:linearAcceleration"),
            ("ReadIMU.outputs:angVel", "PublishIMU.inputs:angularVelocity"),
            ("ReadIMU.outputs:orientation", "PublishIMU.inputs:orientation"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("ReadIMU.inputs:readGravity", True),
            ("PublishIMU.inputs:topicName", args.imu_topic),
            ("PublishIMU.inputs:frameId", args.frame_id),
            ("PublishClock.inputs:topicName", args.clock_topic),
            ("PublishJoints.inputs:topicName", args.joint_states_topic),
        ],
    },
)
# imuPrim is a USD relationship -> set the target prim explicitly.
set_targets(
    prim=stage.GetPrimAtPath(GRAPH + "/ReadIMU"),
    attribute="inputs:imuPrim",
    target_prim_paths=[imu_path],
)
# targetPrim of the joint-state publisher = the G1 articulation root.
set_targets(
    prim=stage.GetPrimAtPath(GRAPH + "/PublishJoints"),
    attribute="inputs:targetPrim",
    target_prim_paths=[robot_artic],
)
carb.log_warn(f"[g1_sim] IMU @ {imu_path} -> {args.imu_topic} ({args.frame_id}); "
              f"joint states -> {args.joint_states_topic} from {args.robot_prim}")

# ---------------------------------------------------------------------------
# Optional fixed base: pin the pelvis to the world so the unactuated robot does
# not collapse during static localization tests.
# ---------------------------------------------------------------------------
if args.hold_pose and args.stabilize:
    carb.log_error("[g1_sim] --hold-pose and --stabilize are mutually exclusive; "
                   "ignoring --hold-pose (AMO actively stands the robot).")
    args.hold_pose = False

if args.hold_pose:
    if not stage.GetPrimAtPath(args.robot_root).IsValid():
        carb.log_error(f"[g1_sim] --hold-pose: root prim {args.robot_root} not found; "
                       f"robot will collapse. Pass --robot-root <pelvis prim path>.")
    else:
        fj_path = "/World/g1/hold_pose_fixed_joint"
        fj = UsdPhysics.FixedJoint.Define(stage, fj_path)
        # body0 unset = world; body1 = robot root -> pins it at its current pose.
        fj.CreateBody1Rel().SetTargets([args.robot_root])
        carb.log_warn(f"[g1_sim] --hold-pose: pinned {args.robot_root} to world "
                      f"(fixed base). Joints below the pelvis still settle once, "
                      f"then the robot is static.")

# ---------------------------------------------------------------------------
# Run. Sim time drives /clock; use_sim_time:=true on the ROS side.
# ---------------------------------------------------------------------------
_PHYSICS_DT = 1.0 / 200.0
sim_ctx = SimulationContext(physics_dt=_PHYSICS_DT, rendering_dt=1.0 / 60.0,
                            stage_units_in_meters=1.0)
simulation_app.update()
sim_ctx.play()

# --stabilize: actively hold the G1 upright with the Navigation AMO policy. Set
# up AFTER play() so the articulation can initialize against a running timeline.
stabilizer = None
if args.stabilize:
    sys.path.insert(0, _HERE)
    from isaac_amo_stabilizer import Stabilizer  # noqa: E402
    stabilizer = Stabilizer(robot_prim=robot_artic, physics_dt=_PHYSICS_DT,
                            command=(0.0, 0.0, 0.0))
    stabilizer.setup()
    # Tick AMO on every physics step (200 Hz); it decimates to the policy's
    # 50 Hz internally. A physics callback is correctly timed, unlike the
    # render-rate app.update() loop below.
    sim_ctx.add_physics_callback("amo_stabilize", lambda _dt: stabilizer.on_physics_step())
    carb.log_warn("[g1_sim] --stabilize: AMO policy holding the G1 standing pose.")

carb.log_warn("[g1_sim] playing. Publishing /livox/lidar, /livox/imu_raw, /clock. "
              "Run 'ros2 launch g1_sim_bridge sim_localization.launch.py' to get "
              "/livox/lidar_reliable + /livox/imu for DLIO.")

try:
    while simulation_app.is_running():
        simulation_app.update()
finally:
    if stabilizer is not None:
        stabilizer.shutdown()
    sim_ctx.stop()
    simulation_app.close()
