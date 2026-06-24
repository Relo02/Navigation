# System architecture

End-to-end overview of the Unitree G1 autonomous-navigation stack as packaged by
this `Navigation/` deployment layer:

**LiDAR + IMU → DLIO localization → A\* global/local planning → MPC
trajectory tracking → AMO RL walking gait → robot.**

The closed loop runs continuously: *pose + obstacles + goal → A\* path → MPC
velocity → AMO gait → robot moves → new pose*. The navigation layer thinks in
**goals and paths**; the AMO policy thinks only in **velocity commands**
`(vx, vy, yaw_rate)`. The MPC is the bridge between the two.

See also: [dockerfiles.md](dockerfiles.md) (the three images),
[amo_inference_plan.md](amo_inference_plan.md) (the gait + smoothing internals),
[A_STAR_MPC_PLANNER.md](A_STAR_MPC_PLANNER.md) (the goal→velocity planner), and
[LOCAL_VOXEL_MAP.md](LOCAL_VOXEL_MAP.md) (the ground-removed obstacle source).

---

## 1. Closed-loop data flow

```mermaid
flowchart TB
    robot(["Unitree G1 robot"])
    lidar["Livox MID-360<br/>LiDAR + IMU"]

    subgraph PERC["Perception / localization"]
        flio["DLIO<br/>LiDAR-inertial odometry + 3D map"]
        lvm["local_voxel_map<br/>accumulate + gravity-aware<br/>SVD ground removal"]
    end

    subgraph PLAN["Planning + control (ROS 2)"]
        astar["a_star_node<br/>global+local A* on Gaussian grid"]
        mpc["mpc_node (IPOPT)<br/>trajectory optimization"]
        goal["goal source<br/>RViz 2D Goal / fixed goal"]
    end

    subgraph GAIT["AMO gait (this repo: amo/)"]
        amo["amo_inference.py<br/>AMOPolicy + JointSmoother<br/>WS :8766"]
    end

    bridge["unitree bridge<br/>SDK2 / CycloneDDS"]

    lidar -->|raw cloud + imu| flio
    flio -->|deskewed cloud odom| lvm
    lvm -->|/local_voxel_map/obstacles<br/>ground-removed cloud + costmap| astar
    flio -->|odometry /dlio/odom_node/odom| astar
    flio -->|odometry /dlio/odom_node/odom| mpc
    goal -->|/global_goal| astar
    astar -->|/a_star/path| mpc
    mpc -->|velocity_target vx,vy,yaw<br/>WS :8766| amo
    amo -->|29-DoF lowcmd| bridge
    bridge <-->|DDS lowcmd / lowstate| robot
    robot -->|IMU + joint state<br/>DDS lowstate| amo
    robot -->|IMU + LiDAR| lidar
```

The AMO policy is a **velocity-tracking** controller: it never sees the goal. It
receives `(vx, vy, yaw_rate)` from the MPC and produces stable walking that
realizes that velocity. All goal→path→velocity reasoning is upstream in A\*/MPC.

---

## 2. Mapping onto the deployment images

The three `Navigation/docker` images and where each component runs:

```mermaid
flowchart LR
    subgraph loc["localization image<br/>(ros:humble + PCL/Eigen/OpenMP + Livox-SDK2)"]
        flio2["DLIO"]
        lvm2["local_voxel_map<br/>(ground removal)"]
        plan2["A* + MPC planner (ROS 2)*"]
    end

    subgraph amoimg["amo_policy image<br/>(python:3.11, no ROS 2)"]
        amo2["amo_inference.py<br/>AMOPolicy + smoothing"]
    end

    subgraph uni["unitree image<br/>(ros:humble + Unitree SDK2)"]
        teleop["teleop / joystick / RViz<br/>Unitree msgs"]
    end

    robot2(["Unitree G1"])

    flio2 -->|"deskewed cloud (ROS 2)"| lvm2
    flio2 -->|"/dlio/odom_node/odom (ROS 2)"| plan2
    lvm2 -->|"obstacles / costmap (ROS 2)"| plan2
    plan2 -->|"velocity_target (WS :8766)"| amo2
    amo2 <-->|"CycloneDDS lowcmd/lowstate"| robot2
    flio2 <-->|"CycloneDDS (LiDAR/IMU)"| robot2
    teleop <-->|"DDS / ROS 2"| robot2
```

\* The A\*/MPC planner is a ROS 2 package (`a_star_mpc_planner`). It runs in the
ROS 2 environment — co-located with the `localization` image (or its own ROS 2
container). It is **not** a separate `Navigation/docker` image; only
perception, the AMO gait, and the Unitree bridge are packaged here.

| Concern | Image | Transport in | Transport out |
|---|---|---|---|
| DLIO + ground-removed local map | `localization` | CycloneDDS (LiDAR/IMU) | ROS 2 topics |
| A\* + MPC planning | ROS 2 (with `localization`) | ROS 2 (`/dlio/odom_node/odom`, clouds, `/global_goal`) | **WebSocket `:8766`** |
| AMO gait + smoothing | `amo_policy` | WS `:8766` (velocity) + DDS (robot state) | CycloneDDS `rt/lowcmd` |
| Teleop / Unitree bridge / RViz | `unitree` | DDS / ROS 2 | DDS / ROS 2 |

Why the AMO policy is isolated: RoboJuDo needs Python 3.11 + a specific
torch/mujoco/cyclonedds stack that conflicts with the ROS 2 Humble (Python 3.10)
perception stack — so it talks to the rest of the system over the WS bridge and
raw DDS, never by sharing a Python environment. See [dockerfiles.md](dockerfiles.md).

---

## 3. AMO inference internals (the velocity → joints path)

Inside [amo/amo_inference.py](../amo/amo_inference.py), each 50 Hz tick:

```mermaid
flowchart LR
    cmdsrc["command source<br/>zero | constant | websocket(MPC)"] -->|vx,vy,yaw| obs
    state["robot state (DDS lowstate)<br/>IMU, dof_pos, dof_vel"] -->|env_data| obs["build observation"]
    obs --> pol["AMOPolicy<br/>(RoboJuDo torchscript)"]
    pol -->|raw 29-DoF target| sm["JointSmoother<br/>A: startup blend<br/>D: optional filter<br/>C: per-tick clamp"]
    sm -->|cmd_q| step["env.step → set_gains + lowcmd"]
    step -->|CycloneDDS rt/lowcmd| robot3(["Unitree G1"])
```

- **Inputs:** (1) the velocity command `(vx, vy, yaw_rate)` from the command
  source, and (2) the robot's proprioceptive state (IMU orientation/angular
  velocity, joint positions/velocities) over DDS.
- **Command source** is selected by `command.source` in
  [docker/config/amo_g1.yaml](../docker/config/amo_g1.yaml):
  `websocket` = live references from the **MPC planner** (`:8766`); `constant` =
  hardcoded velocities (config or `--vx/--vy/--yaw`); `zero` = stand in place.
- **Smoothing** (`JointSmoother`) guarantees no joint snap at activation: an
  S-curve blend from the captured posture to the first AMO reference + a
  soft→full PD-gain ramp (startup only), plus a per-tick clamp always on, and an
  optional always-on slew filter. Details in [amo_inference_plan.md](amo_inference_plan.md).

---

## 4. Frames & key interfaces

| Interface | Type | Producer → Consumer | Payload |
|---|---|---|---|
| `rt/lowstate` | CycloneDDS | robot → AMO / DLIO | IMU, joint state |
| `rt/lowcmd` | CycloneDDS | AMO → robot | 29-DoF PD targets + gains |
| LiDAR/IMU | CycloneDDS | MID-360 → DLIO | point cloud + IMU |
| `/dlio/odom_node/odom`, TF `odom→base_link` | ROS 2 | DLIO → A\*/MPC | localization odometry |
| deskewed cloud, map | ROS 2 | DLIO → A\*/MPC | planning inputs |
| `/global_goal` | ROS 2 | RViz/goal → A\* | navigation goal |
| `/a_star/path` | ROS 2 | A\* → MPC | global/local path |
| `velocity_target` | WebSocket `:8766` | MPC → AMO | `vx, vy, yaw_rate` |

> Note on the MID-360: it is mounted inverted on the G1, so the IMU needs the
> extrinsic correction. On the real robot this is handled by DLIO's
> `baselink2imu` rotation extrinsic = `R_x(180)` in `dlio_mid360_real.yaml`
> (stock Livox driver) — see the project memory on the inverted IMU before
> trusting odometry while walking. In sim the IMU is upright, so `dlio_sim.yaml`
> uses identity extrinsics.

---

## 5. Bring-up order

1. **Robot + DDS** reachable on the chosen NIC (`UNITREE_NET_IFACE`).
2. **localization** — LiDAR preprocessing + DLIO + `g1_local_map` →
   pose/odometry + ground-removed obstacle cloud:
   `ros2 launch g1_bringup real_localization.launch.py`. Keep the robot
   **stationary for the first ~3 s** so DLIO can run its IMU + gravity
   calibration before it starts moving.
3. **planner** (A\* + MPC) —
   `ros2 launch a_star_mpc_planner planner.launch.py`. Consumes pose
   (`/dlio/odom_node/odom`), obstacles (`/local_voxel_map/obstacles`) and the
   goal (`/global_goal`); emits a `Twist` on `/mpc/cmd_vel` that the bundled
   `cmd_vel_to_amo` bridge forwards to the AMO WS as `velocity_target`.
   *(Steps 2–3 can be started together with `ros2_ws/src/autonomy.sh`.)*
4. **amo_policy** — start the gait and choose how it is driven:
   - **Autonomous:** `AUTONOMOUS=1 NET_IF=… ./docker/run_amo.sh` →
     `command.source=websocket`, tracking the MPC's `velocity_target`.
   - **Manual (no autonomy):** `JOYSTICK=1 …` → drive with the Unitree pad,
     used for teleop / SLAM-mapping a space.
   Staged snap-free activation runs first either way.
5. **Set a goal** — RViz **2D Goal Pose** tool (publishes `/global_goal`), or
   `ros2 topic pub --once /global_goal …`. The robot then walks the planned path.

Dry run first with `NET_IF=eth0 ./run_amo.sh --observe_only` (no motor commands).
Full planner details + tuning: [A_STAR_MPC_PLANNER.md](A_STAR_MPC_PLANNER.md).
