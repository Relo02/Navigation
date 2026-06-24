# Navigation

Deployment layer for the Unitree G1 navigation stack. This folder packages the
robot software into three decoupled Docker images and holds the RoboJuDo AMO
gait policy used for real-robot walking.

## Layout

```
Navigation/
├── docker/
│   ├── Dockerfile.unitree        # Unitree SDK2 (C++) + unitree_ros2, teleop, RViz
│   ├── Dockerfile.localization   # DLIO localization (PCL, Eigen, OpenMP, Livox-SDK2)
│   ├── Dockerfile.amo_policy     # RoboJuDo AMO RL gait (pure-Python DDS, no ROS 2)
│   ├── docker-compose.yml        # the three services
│   ├── run_amo.sh                # launch AMO inference via compose (forwards args)
│   ├── config/amo_g1.yaml        # AMO inference + filter tunables
│   ├── policies/                 # exported model assets (mounted to /workspace/policies)
│   └── shared/                   # scratch shared with the host
├── amo/                          # AMO inference + joint-smoothing code
│   ├── amo_inference.py          # driver: AMO policy + smoothing, 50 Hz loop
│   ├── amo_policy.py             # RoboJuDo AMOPolicy + UnitreeEnv wrapper
│   ├── joint_filters.py          # smoothing filters (layer D) + JointSmoother
│   ├── activation_utils.py       # smoothstep blend / gain-ramp / clamp helpers
│   └── tests/test_joint_filters.py
├── ros2_ws/                      # locally-editable ROS 2 workspace (mounted to /ws)
│   └── src/
│       ├── direct_lidar_inertial_odometry  # DLIO (UCLA VECTR) LiDAR-inertial odometry
│       ├── livox_ros_driver2                # MID-360 ROS 2 driver (tuned config)
│       ├── g1_local_map                     # ground-removed rolling voxel/obstacle map
│       ├── a_star_mpc_planner               # A* + MPC goal→velocity planner
│       ├── g1_sim_bridge                    # IMU rescale, cmd_vel→AMO WS bridge, launches
│       ├── g1_bringup                       # real_localization + planner launches, RViz
│       └── g1_description                   # G1 URDF / robot model
├── policy/
│   └── RoboJuDo -> ../../RoboJuDo # symlink to the RoboJuDo deploy framework
└── docs/
    ├── system_architecture.md    # end-to-end stack (LiDAR → DLIO → MPC → AMO)
    ├── dockerfiles.md            # how the three images fit together
    ├── amo_inference_plan.md     # AMO inference + joint-smoothing filter design
    ├── A_STAR_MPC_PLANNER.md     # the A* + MPC goal→velocity planner
    └── LOCAL_VOXEL_MAP.md        # the ground-removed obstacle source for the planner
```

## The three images

| Image | Base | Role |
|---|---|---|
| **unitree** | `ros:humble-desktop` | ROS 2 ↔ robot bridge: Unitree SDK2, message packages, teleop, joystick, RViz. |
| **localization** | `ros:humble-desktop` | DLIO LiDAR-inertial odometry/localization for the MID-360 (PCL, Eigen, OpenMP, Livox-SDK2). |
| **amo_policy** | `python:3.11-slim-bookworm` | The RoboJuDo **AMO RL gait** that drives the joints, over CycloneDDS via `unitree_sdk2py`. **No ROS 2.** |

They are independent and communicate at run time over CycloneDDS and ROS 2 /
WebSocket — never by sharing a Python environment. See
[docs/system_architecture.md](docs/system_architecture.md) for the end-to-end
data flow and [docs/dockerfiles.md](docs/dockerfiles.md) for the image
framework, shared DDS conventions (`UNITREE_NET_IFACE`, host networking), and
why the stack is split three ways.

## RoboJuDo

[RoboJuDo](https://github.com/HansZ8/RoboJuDo) is a plug-and-play robot
deployment framework (controller / environment / policy, modular and
composable). The AMO gait runs through its `UnitreeEnv` + `AMOPolicy`. It needs
Python ≥ 3.11, `torch`, `mujoco` (mandatory — forward kinematics), and the
Unitree DDS bindings; the `amo_policy` image provides all of these.

`policy/RoboJuDo` is a **symlink** to the RoboJuDo checkout. Ensure the resolved
directory is mounted into the container at run time (the compose file does this).

## Quick start — AMO policy

Use [docker/run_amo.sh](docker/run_amo.sh) — it builds `CYCLONEDDS_URI` from the
NIC, mounts the `amo/` code + RoboJuDo + config, and forwards all args to
`amo_inference.py`:

```bash
cd Navigation/docker

BUILD=1 NET_IF=enp12s0 ./run_amo.sh --observe_only   # build, then dry run (no motor cmds)
NET_IF=enp12s0 ./run_amo.sh                           # run with amo_g1.yaml as-is (stands)
NET_IF=enp12s0 ./run_amo.sh --vx 0.3                  # walk forward at a constant velocity

AUTONOMOUS=1 NET_IF=enp12s0 ./run_amo.sh             # full autonomy: track A*+MPC velocity (WS :8766)
JOYSTICK=1   NET_IF=enp12s0 ./run_amo.sh             # manual: drive with the Unitree pad (hold R1)
```

`AUTONOMOUS=1` sets the command source to `websocket` so the gait tracks the
A\*+MPC planner — the planner stack must be running first (see
[Autonomous navigation](#autonomous-navigation) below). `JOYSTICK=1` drives the
gait from the Unitree pad for teleop / SLAM-mapping a space, **without** the
planner.

Equivalent raw compose call:

```bash
UNITREE_NET_IFACE=enp12s0 docker compose run --rm --service-ports amo_policy \
    python /workspace/amo/amo_inference.py --config /workspace/config/amo_g1.yaml --observe_only
```

`docker compose up amo_policy` drops to a `bash` shell by default — no motors
move on `up`.

### AMO inference + joint smoothing

`amo/amo_inference.py` runs the RoboJuDo AMO gait through a **smoothing stack**
so the joints never snap to the policy reference at activation:

- **Startup (always):** an S-curve blend from the robot's captured posture to the
  first AMO reference, while PD gains ramp soft → full.
- **Per-tick (always):** a clamp capping each joint at `clamp_delta_rad`/tick — a
  hard anti-snap safety rail.
- **Running filter (optional):** an always-on slew/low-pass filter (`ewma` /
  `critdamp`). **Default is `none`** (`filter.kind` in
  [config/amo_g1.yaml](docker/config/amo_g1.yaml)) so the trained gait runs at
  full bandwidth — startup smoothing + clamp still apply.

Tunables are in [docker/config/amo_g1.yaml](docker/config/amo_g1.yaml); CLI flags
override them: `--observe_only`, `--net_if`, `--filter {none,ewma,critdamp}`,
`--vx/--vy/--yaw`. Design notes: [docs/amo_inference_plan.md](docs/amo_inference_plan.md).
Run the filter unit tests:

```bash
cd Navigation/amo && python3 -m pytest tests/ -q
```

### Velocity command input

The AMO policy is a **velocity tracker** — it consumes `(vx, vy, yaw_rate)`, not
goals/waypoints (goal→velocity planning lives upstream in the MPC). The source is
`command.source` in the config:

| `source` | Command comes from | When |
|---|---|---|
| `zero` | always `(0,0,0)` — stand in place | default / safe |
| `constant` | `command.constant` in YAML, or `--vx/--vy/--yaw` | static manual / bench |
| `websocket` | live JSON `{"vx","vy","yaw"}` on `:8766` | **MPC planner** (`AUTONOMOUS=1`), or manual sends |
| `joystick` | Unitree G1 pad via `LowState.wireless_remote` | teleop / SLAM-mapping (`JOYSTICK=1`) |

With `source: websocket` the server accepts any client, so you can drive it
manually:

```bash
echo '{"vx":0.3,"vy":0.0,"yaw":0.1}' | websocat ws://localhost:8766
```

The last value sent is held until the next message. See
[docs/system_architecture.md](docs/system_architecture.md) for how the MPC feeds
this in the full navigation loop.

### Autonomous navigation

Full autonomy = DLIO localization + `g1_local_map` obstacles + the A\*+MPC
planner emitting velocity to the AMO gait. It comes in two ROS 2 launch files
(localization, planner) plus the gait. Hold the robot **still ~3 s** at startup
for DLIO's gravity init, and set `NET_IF` where shown.

**The two ROS 2 launches** (in the `localization` container) can be started
**either** separately **or** together — pick whichever you prefer:

*Option A — two launches by hand (separate terminals, full control/logs):*

```bash
# terminal 1 — DLIO + ground-removed local map (+ RViz):
ros2 launch g1_bringup real_localization.launch.py

# terminal 2 — A* + MPC + the cmd_vel→AMO WS bridge:
ros2 launch a_star_mpc_planner planner.launch.py
```

*Option B — one command, both as separate processes (Ctrl-C stops both):*

```bash
ros2_ws/src/autonomy.sh        # runs both launches; PLANNER_DELAY=3 by default
```

`autonomy.sh` just starts the same two launch files as independent processes
(ROS 2 launch already runs each launch's nodes concurrently — no manual
threading), waits `PLANNER_DELAY` s between them for DLIO init, and tears both
down together on Ctrl-C. It is the all-in-one equivalent of Option A.

**Then start the gait** (in the `amo_policy` container) and send a goal:

```bash
# gait tracking the planner's velocity_target over WS :8766:
AUTONOMOUS=1 NET_IF=enp12s0 ./docker/run_amo.sh
```

Send a goal with RViz's **2D Goal Pose** tool (publishes `/global_goal`) and the
robot walks the planned path, replanning continuously. Full details, topics and
tuning: [docs/A_STAR_MPC_PLANNER.md](docs/A_STAR_MPC_PLANNER.md).

## Building / running the ROS 2 images

```bash
cd Navigation/docker
UNITREE_NET_IFACE=enp12s0 docker compose up unitree        # robot bridge + teleop
UNITREE_NET_IFACE=enp12s0 docker compose up localization   # DLIO localization
```

Both use host networking + IPC so DDS discovery reaches the robot.

The **DLIO workspace is local and editable** at [ros2_ws/](ros2_ws/),
bind-mounted to `/ws`. The DLIO build deps (PCL, Eigen, OpenMP) and Livox-SDK2
are baked into the image, but the workspace itself is built **inside the
container** the first time:

```bash
cd Navigation/docker
docker compose run --rm localization bash
build_ws        # rosdep + colcon; build/ install/ log/ persist on the host (git-ignored)
```

## Docs

- [docs/system_architecture.md](docs/system_architecture.md) — end-to-end stack and data flow.
- [docs/dockerfiles.md](docs/dockerfiles.md) — the three-image framework.
- [docs/simulation_stack.md](docs/simulation_stack.md) — how the stack is wrapped into Isaac Sim (sim front-end).
- [docs/amo_inference_plan.md](docs/amo_inference_plan.md) — AMO inference and the joint-smoothing filter.
- [docs/A_STAR_MPC_PLANNER.md](docs/A_STAR_MPC_PLANNER.md) — the A\* + MPC goal→velocity planner.
- [docs/LOCAL_VOXEL_MAP.md](docs/LOCAL_VOXEL_MAP.md) — the ground-removed obstacle source for the planner.
- [docs/GROUND_SEGMENTATION.md](docs/GROUND_SEGMENTATION.md) — the per-cell local-minimum ground filter.
- [docs/DLIO_G1_MID360_TUNING.md](docs/DLIO_G1_MID360_TUNING.md) — DLIO parameters for the G1 + MID-360.
- [docs/DLIO_DEPLOYMENT_TESTING.md](docs/DLIO_DEPLOYMENT_TESTING.md) — ordered DLIO bring-up + tests.

The Isaac Sim simulation entrypoint lives in [sim/](sim/) — see [sim/README.md](sim/README.md).
