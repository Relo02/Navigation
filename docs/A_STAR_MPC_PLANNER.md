# A\* + MPC local planner (`a_star_mpc_planner`)

The navigation layer that turns a **goal** into the **velocity command** the AMO
gait tracks. It sits between perception (DLIO + `g1_local_map`) and the gait:

```
goal ─► A* (global/local path on a Gaussian cost grid) ─► MPC (IPOPT) ─► velocity ─► AMO
```

Package: `ros2_ws/src/a_star_mpc_planner` · Nodes: `a_star_node`, `mpc_node` ·
Launch: `a_star_mpc_planner/planner.launch.py`.

This planner was ported from the older `G1_navigation` (FAST-LIO) stack and
re-adapted to this repo's DLIO front-end. See
[system_architecture.md](system_architecture.md) for where it sits in the whole
loop and [LOCAL_VOXEL_MAP.md](LOCAL_VOXEL_MAP.md) for the obstacle source it
consumes.

---

## 1. Two nodes + a bridge

| Node | Role |
|---|---|
| `a_star_node` | Builds a robot-centred **2.5D Gaussian cost grid** from the obstacle cloud and runs **A\*** from the current pose to the goal. Publishes `/a_star/path`. |
| `mpc_node` | Tracks `/a_star/path` with a **nonlinear MPC** (CasADi/IPOPT): a 6-DoF unicycle-ish model with actuator lag, obstacle barriers, and a velocity envelope the AMO gait can actually walk. Publishes a velocity `Twist` on `/mpc/cmd_vel`. |
| `cmd_vel_to_amo_node` (in `g1_sim_bridge`) | The AMO policy is **not** a ROS 2 process. This bridge forwards `/mpc/cmd_vel` as `{vx,vy,yaw}` JSON to the AMO **WebSocket** server (`:8766`). `planner.launch.py` starts it automatically (`bridge:=true`). |

The MPC is the bridge between "goals & paths" and "velocity": the AMO policy
never sees the goal — it only tracks `(vx, vy, yaw_rate)`.

---

## 2. Inputs / outputs

**Subscribes** (all in the `odom` frame):

| Topic | Type | From | Used for |
|---|---|---|---|
| `/dlio/odom_node/odom` | `nav_msgs/Odometry` | DLIO | robot pose (converted to PoseStamped internally; BEST_EFFORT) |
| `/local_voxel_map/obstacles` | `sensor_msgs/PointCloud2` | `g1_local_map` | ground-removed obstacle cloud (the cost-grid + MPC obstacle source) |
| `/a_star/path` | `nav_msgs/Path` | `a_star_node` | the path the MPC tracks |
| `/global_goal` | `geometry_msgs/PoseStamped` | RViz / a goal source | navigation goal |

**Publishes:**

| Topic | Type | Meaning |
|---|---|---|
| `/a_star/path` | `nav_msgs/Path` | local A\* path |
| `/a_star/local_goal` | `geometry_msgs/PoseStamped` | current local goal (path end) |
| `/a_star/occupancy_grid` | `nav_msgs/OccupancyGrid` | the inflated Gaussian cost grid (RViz/debug) |
| `/a_star/voxel_grid`, `/a_star/persistent_obstacles` | `PointCloud2` | 3D occupancy + persistent-memory cells (RViz) |
| `/mpc/predicted_path` | `nav_msgs/Path` | MPC's N-step predicted trajectory |
| `/mpc/next_setpoint` | `geometry_msgs/PoseStamped` | lookahead setpoint |
| **`/mpc/cmd_vel`** | `geometry_msgs/Twist` | **the velocity command** → AMO via the WS bridge |
| `/mpc/diagnostics` | `std_msgs/Float64MultiArray` | `[success, cost, solve_ms, avg_ms, fails, security, vx_eff]` |
| `/navigation/state` | `std_msgs/String` | nav state machine: `IDLE`/`NAVIGATING`/`GOAL_REACHED`/`SECURITY`/`STOPPED` |
| `/mpc/predicted_obstacles` | `visualization_msgs/MarkerArray` | obstacle spheres at predicted positions (RViz) |
| `/mpc/obstacle_velocities` | `visualization_msgs/MarkerArray` | velocity-vector ARROW per **dynamic** obstacle cluster (RViz) |
| `/estop` (sub) | `std_msgs/Bool` | latched manual e-stop; the AMO bridge forwards zero velocity while true |

---

## 3. How it was adapted to DLIO

The original planner assumed a FAST-LIO front-end with a pelvis-origin frame
(floor at z≈0) and a raw, ground-carrying scan. The DLIO front-end differs in
three ways that the config and code account for:

1. **Pose is `nav_msgs/Odometry`, not a `PoseStamped`.** Both nodes subscribe to
   `/dlio/odom_node/odom` and repackage it into a PoseStamped (`_odom_cb`). Body
   velocity is still re-estimated by low-pass pose differentiation, so DLIO's
   `twist` convention is not relied upon.

2. **Obstacles are already ground-removed** by `g1_local_map`. So the planner's
   own per-cell ground segmentation is **off** (`ground_segment_en: false`) — the
   incoming points feed the cost layer directly, and the persistent map is fed
   the raw cloud (re-segmenting an already-clean cloud could drop low obstacles).

3. **The odom origin is at the sensor (~1 m above the floor)**, so the floor sits
   near z≈−1 m. An absolute-z rule is therefore meaningless:
   - the 2.5D **step-over is disabled** (`step_over_height: 0.0`) — every obstacle
     in the (already curated) cloud is a full blocker;
   - the MPC obstacle **z-band stays disabled** (the cloud is ground-removed);
   - the **voxel z-band is widened** to `[-2, 2]` (vmap visualization only).
   The Gaussian cost layer itself is **per-cell relative**, so the z offset does
   not affect it.

### The Gaussian cost grid is NOT redundant with the voxel map
`g1_local_map` outputs ground-removed obstacle **points**; the planner inflates
those points into a **cost field** (Gaussian-CDF of distance to the nearest
obstacle) that A\* searches. Different stages, not competing maps — and inflation
happens exactly **once**, here. (`g1_local_map`'s `~/costmap` is a raw,
un-inflated 2D layer for visualization and is not consumed by the planner.)

### Optional DLIO global-map fusion (off by default)
`enable_dlio_map:=true` fuses DLIO's keyframe global map (`/dlio/map_node/map`)
as static context for structure that has left the live FOV. That map is **not**
ground-removed, so the fusion path **strips its floor** with the same per-cell
segmentation (robust to the sensor-origin frame) and caps points above the
sensor (`dlio_map_z_above`) before adding them as obstacles. Off by default:
`g1_local_map` already provides a dense ground-removed cloud.

---

## 4. Running it

In the ROS 2 / `localization` container, **after** DLIO + `g1_local_map` are up
and the robot has been held still ~3 s for DLIO's IMU/gravity init:

```bash
# A* + MPC + the AMO velocity bridge, on ROS_DOMAIN_ID=42 (matches localization):
ros2 launch a_star_mpc_planner planner.launch.py

# planner only, no AMO bridge (e.g. for RViz inspection):
ros2 launch a_star_mpc_planner planner.launch.py bridge:=false

# point the bridge at a remote AMO host:
ros2 launch a_star_mpc_planner planner.launch.py amo_host:=192.168.1.10 amo_port:=8766
```

To start **both** the localization and planner launches at once (as two separate
processes, with the DLIO-init delay and clean Ctrl-C teardown), use the wrapper
`ros2_ws/autonomy.sh` instead of running the two `ros2 launch` lines by hand.

### Reading the logs (DLIO vs planner, separately)

When run by hand (above), each `ros2 launch` already owns its own terminal, so
the DLIO/perception logs and the A\*+MPC planner logs are naturally separate.

`autonomy.sh` runs both from one terminal, so it instead writes each launch's
stdout+stderr to its **own** logfile under `ros2_ws/logs/` rather than
interleaving them on screen. Follow each independently:

```bash
tail -f ros2_ws/logs/localization_latest.log   # DLIO + g1_local_map
tail -f ros2_ws/logs/planner_latest.log        # a_star_node + mpc_node (+ WS bridge)
```

`*_latest.log` symlinks track the newest run; timestamped files are kept beside
them. Env overrides: `LOG_DIR=/path` relocates the logs; `LOG_TO_CONSOLE=1` also
mirrors both streams to the launching terminal (re-interleaving them). The
planner log is where the navigation **state machine** surfaces — `[MPC]`
state transitions and `/navigation/state` (`IDLE` / `NAVIGATING` /
`GOAL_REACHED` / `SECURITY` / `STOPPED`), `[MPC-SECURITY]` escape events,
`[MPC-STOP]` fail-safe stops, and the per-cycle solve line.

Then start the gait in **autonomous** mode (amo_policy container):

```bash
AUTONOMOUS=1 NET_IF=enp12s0 ./docker/run_amo.sh
```

`AUTONOMOUS=1` sets the AMO command source to `websocket`, so it tracks the
velocity the MPC sends. (Contrast `JOYSTICK=1`, which drives the gait from the
Unitree pad for teleop / SLAM-mapping a space **without** autonomous navigation.)

### Setting a goal
Use RViz's **2D Goal Pose** tool — it now publishes to `/global_goal` (the
planner's goal topic) in the `odom` frame. Or publish one directly:

```bash
ros2 topic pub --once /global_goal geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: odom}, pose: {position: {x: 2.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

The planner replans continuously; move the goal anytime. A fixed startup goal can
be set with `goal_x/goal_y` + `wait_for_goal: false` in the config.

### RViz
`g1_bringup/rviz/g1_dlio.rviz` (opened by `real_localization.launch.py`) shows
the DLIO odom/cloud/map, the local voxel map + costmap, and the planner layers:
**AStarPath** (`/a_star/path`), **MPCPredictedPath** (`/mpc/predicted_path`),
**GlobalGoal** (`/global_goal`), and a toggleable **AStarCostGrid**
(`/a_star/occupancy_grid`).

To visualise **dynamic obstacles**, add two `MarkerArray` displays:
`/mpc/predicted_obstacles` (spheres at each obstacle's predicted position) and
`/mpc/obstacle_velocities` (a magenta **velocity-vector arrow + speed label** for
every cluster the tracker classifies as *moving*). Static structure produces no
arrow, so an arrow appearing = something is actually moving; the arrow points
along the tracked velocity and its length is `speed * mpc_vel_arrow_scale`
seconds (default 1 s, so 1 m/s ⇒ a 1 m arrow). Quick check without RViz:
`ros2 topic echo /mpc/obstacle_velocities`.

### Safety e-stop (`s` / `g` keys)
When started via `ros2_ws/autonomy.sh`, that terminal runs a keyboard e-stop
(`g1_sim_bridge/estop_keyboard_node`) that publishes a latched `/estop`:

- **`s` + Enter — STOP**: the `cmd_vel→AMO` bridge forwards **zero** velocity to
  the gait regardless of `/mpc/cmd_vel`; the robot halts and holds in place.
- **`g` + Enter — GO**: releases `/estop`; the bridge resumes forwarding the
  MPC velocity and the robot continues toward the current goal.

It is a *soft, reversible* stop (halt then resume without restarting anything),
and is independent of the automatic fail-safes (MPC zeroes on stale pose/path;
the bridge zeroes on a stale `/mpc/cmd_vel`). The e-stop has top priority at the
bridge, so **while it is engaged the robot will not move even if you send a new
goal** — the planner still replans to the new goal (you'll see the path and
`NAVIGATING` state), but no velocity reaches the gait until you press `g`. Run it
standalone with `ros2 run g1_sim_bridge estop_keyboard_node`.

---

## 5. Tuning

All parameters live in
[`config/planner_params_default.yaml`](../ros2_ws/src/a_star_mpc_planner/config/planner_params_default.yaml),
heavily commented. The knobs you are most likely to touch:

| Param | Default | Note |
|---|---|---|
| `grid_reso` / `grid_half_width` | `0.05` / `20.0` | A\* cell size / local workspace half-extent |
| `grid_std` | `0.2` | Gaussian obstacle inflation spread |
| `obstacle_threshold` | `0.5` | cost above which a cell blocks A\* |
| `replan_rate_hz` | `1.0` | A\* replan rate |
| `mpc_vx_max` / `mpc_vy_max` / `mpc_omega_max` | `0.45` / `0.05` / `0.80` | velocity envelope handed to AMO (keep within what the gait tracks stably) |
| `mpc_obs_r` | `0.55` | robot half-width + margin for the MPC obstacle barrier |
| `mpc_security_threshold` | `0.25` | occupancy at the robot that triggers the escape behaviour |
| `enable_dlio_map` | `false` | fuse DLIO's global map as static context |

**Dependencies:** `a_star_node`/`mpc_node` need `numpy`, `scipy`, and `casadi` in
the ROS 2 / `localization` image. Build the workspace in-container (`build_ws`).

---

## 6. Quick checks

- `ros2 topic echo /a_star/path --once` — A\* is producing a path to the goal.
- `ros2 topic echo /mpc/cmd_vel --once` — the MPC is emitting a velocity.
- `ros2 topic echo /mpc/diagnostics` — `success`/`solve_ms` healthy, `security` 0.
- `ros2 topic echo /navigation/state` — `NAVIGATING` while driving, `GOAL_REACHED`
  on arrival (latched zero-velocity hold), `STOPPED` ⇒ stale pose/path fail-safe.
- Reading logs under `autonomy.sh`: `tail -f ros2_ws/logs/planner_latest.log`
  (A\*+MPC) and `tail -f ros2_ws/logs/localization_latest.log` (DLIO) — see §4.
- AMO log: `command source: websocket :8766` and `connected to AMO WebSocket`
  from the bridge ⇒ velocity is reaching the gait.
- Verify clouds/paths in **RViz**, not `ros2 topic hz` from the host (a fresh CLI
  participant can't pull the large clouds — see [LOCAL_VOXEL_MAP.md](LOCAL_VOXEL_MAP.md) §6).
