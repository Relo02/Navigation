# Local Voxel Grid Map (`g1_local_map`)

How the G1's **local rolling 3D voxel map** is computed, from the DLIO point
cloud to the obstacle representation consumed by the **A\*+MPC planner**.

Package: `ros2_ws/src/g1_local_map` · Node: `local_voxel_map_node` ·
Launched by `g1_bringup/real_localization.launch.py` (arg `local_map:=true`).

---

## 1. Why this node exists

DLIO produces two clouds, neither of which is a good *local obstacle* map:

| Topic | What it is | Why not use it for local avoidance |
|---|---|---|
| `/dlio/map_node/map` | Global SLAM map | Only republishes on a new **keyframe** (~1 m / 45° of motion) → far too laggy; grows unbounded; includes the floor. |
| `/dlio/odom_node/pointcloud/deskewed` | Per-scan registered cloud (~10 Hz) | Fast and in `odom`, but **sparse per scan**, **includes the ground**, and unbounded in the past. |

`local_voxel_map_node` turns the fast deskewed cloud into a **bounded,
robot-centred, ground-removed, temporally-densified** obstacle map that refreshes
every scan and forgets what the robot has passed.

---

## 2. Inputs / outputs

**Subscribes** (BEST_EFFORT — see [§6](#6-qos--transport-notes-important)):
- `/dlio/odom_node/pointcloud/deskewed` — `sensor_msgs/PointCloud2`, **odom** frame, deskewed + registered, ~10 Hz.
- `/dlio/odom_node/odom` — `nav_msgs/Odometry`, robot pose in odom (used only to centre the rolling window).

**Publishes** (in the `odom` frame):
- `~/obstacles` — `PointCloud2` of ground-removed obstacle **voxel centres**. **This is the planner feed** → point the planner's `obstacle_topic` here.
- `~/voxel_grid` — `PointCloud2`, identical to `~/obstacles`, for RViz / 3D queries.
- `~/costmap` — `nav_msgs/OccupancyGrid`, a 2D top-down projection (latched, TRANSIENT_LOCAL).

No TF math is needed: the deskewed cloud is already in `odom`, and the window is
centred on the odom-frame robot position from the odometry message.

---

## 3. The pipeline (per scan)

```
deskewed cloud (odom) ─┐
                       ├─► 1. crop  ─► 2. accumulate (raw, +decay) ─► 3. ground removal ─► 4. publish
odom (robot xyz) ──────┘
```

### 1. Rolling-window crop
Keep only points that are:
- within an **XY box** `±half_width` (default **8 m**) of the robot,
- within a **vertical band** `[robot_z − z_below, robot_z + z_above]` (defaults **1.5 m / 1.5 m**),
- **outside** `min_range` (default **0.4 m**) of the robot — drops self-hits / the body.

> **Why `z_below` must reach below the floor.** DLIO's `baselink2lidar` translation
> is zero, so the odom origin sits at the **sensor**, ~1 m above the ground →
> the floor lands near `z ≈ −1 m` in odom. If `z_below` doesn't reach it, the
> floor is cropped out *before* ground segmentation, which then has no ground
> reference and lets low obstacles/floor leak through. Hence `z_below = 1.5 m`.

### 2. Accumulate raw points into a decaying voxel grid
The cropped **raw** points (floor included) are inserted into a voxel occupancy
grid in the odom frame (`VoxelAccumulator`):
- Each point maps to an integer voxel index `floor(p / voxel_size)` (default **0.10 m**).
- Each occupied voxel stores the **last time it was seen**.
- Every scan, voxels are **pruned** if older than `persistence_s` (default **3 s**)
  or outside the rolling window — bounding memory and forgetting the past.

Keying voxels by absolute odom indices means the grid **never needs
re-centring**; the robot-centred window is enforced purely by the prune step.

> **Why accumulate _before_ removing ground.** A single MID-360 scan is sparse —
> roughly one point per `ground_cell`. Per-cell ground segmentation on one scan
> would treat that lone point as its own ground and drop nearly everything.
> Accumulating ~`persistence_s × 10` scans first gives each cell enough points
> for the floor to be a reliable per-cell minimum. **This ordering is the fix for
> the "empty map" bug — never segment a single scan.**

### 3. Ground removal — per-cell lowest-point segmentation
Run on the **accumulated (dense)** voxel centres (`segment_obstacles`):
1. Bin points into XY cells of size `ground_cell` (default **0.25 m**).
2. The **lowest z** in each cell is the *local ground height* for that cell.
3. A point is an **obstacle** iff its height above its cell's local ground is in
   `(ground_thresh, max_height]` (defaults **0.10 m … 2.0 m**).
   - `≤ ground_thresh` above local ground → **ground**, discarded.
   - `> max_height` above local ground → ceiling / overhead, discarded.

This is **slope- and step-robust** and independent of the absolute floor level
(which drifts slowly with DLIO odom), because ground is estimated *per cell*
rather than as one global plane or a fixed `z` threshold.

Vectorised with `numpy` (`np.minimum.at` over a compact 1-D cell key), so it is
O(points) with no Python loop over cells.

### 4. Publish
The surviving obstacle voxel centres are published as `~/obstacles` and
`~/voxel_grid`; the 2D column projection is published as `~/costmap`. The node
publishes **every scan, even when empty**, so the costmap never goes stale.

---

## 4. The 2D costmap projection

`~/costmap` is an `OccupancyGrid` of `2*half_width / voxel_size` cells per side
(default 160×160 at 0.10 m), centred on the robot:
- any XY cell containing ≥1 obstacle voxel → **100** (occupied),
- all other cells → `costmap_unknown_as` (default **−1 = unknown**; set **0** for free).

It is a convenience/visualization layer; the planner's primary input is the
`~/obstacles` cloud.

---

## 5. Feeding the A\*+MPC planner

The planner's default **`gaussian`** backend subscribes to a raw obstacle
`PointCloud2` on its `obstacle_topic` (BEST_EFFORT, odom frame) and builds its own
Gaussian-inflated cost grid + 2.5D height map internally. So:

- Set the planner's **`obstacle_topic` → `/local_voxel_map/obstacles`**.
- Frame is `odom` (matches the planner's pose frame).
- The planner does its *own* ground segmentation too; pre-removing ground here is
  complementary and keeps the fed cloud small.
- If you want the fed cloud at the planner's grid resolution, set
  `voxel_size:=0.05` (default here is 0.10 m).

---

## 6. QoS / transport notes (important)

These were learned the hard way on this stack (see
[`project_dds_distro_pollution`] notes / `DLIO_DEPLOYMENT_TESTING.md`):

- **Subscriptions are BEST_EFFORT.** DLIO publishes the large deskewed cloud
  `RELIABLE` + `KeepLast(1)`. A *reliable* reader loses the fragment-reassembly
  race against that depth-1 writer and **freezes on the first frame**; a
  best-effort reader takes each burst instead.
- **The bring-up pins `ROS_DOMAIN_ID=42`** to isolate the Humble stack from the
  ROS 2 **Jazzy host**; a cross-distro participant on the same domain corrupts
  CycloneDDS deserialization of the big PointCloud2 (`serdata.cpp:384`). Do
  **not** also set `ROS_LOCALHOST_ONLY=1` — with CycloneDDS that disables
  multicast and caps the domain at ~10 participant indices, so this 9-node stack
  dies with "Failed to find a free participant index". The best-effort readers
  (above) already handle the large cloud without it.
- **Outputs are RELIABLE + KeepLast(5)** so they serve both a reliable subscriber
  (RViz) and a best-effort one (the planner); depth > 1 avoids the same race.
- **Debug from inside the container on `ROS_DOMAIN_ID=42`** — and verify clouds
  via **RViz**, not `ros2 topic hz` (a fresh CLI participant can't pull the large
  cloud even when the co-spawned RViz can).

---

## 7. Parameters (`config/local_map.yaml`)

| Param | Default | Meaning |
|---|---|---|
| `cloud_topic` | `/dlio/odom_node/pointcloud/deskewed` | Input cloud (odom frame) |
| `odom_topic` | `/dlio/odom_node/odom` | Robot pose (window centring) |
| `half_width` | `8.0` m | Rolling-window half-extent |
| `voxel_size` | `0.10` m | Voxel edge = costmap resolution |
| `persistence_s` | `3.0` s | Voxel memory before decay |
| `ground_cell` | `0.25` m | XY cell for local-ground estimation |
| `ground_thresh` | `0.10` m | Height above local ground ⇒ obstacle |
| `max_height` | `2.0` m | Ignore points above this over ground |
| `z_below` / `z_above` | `1.5` / `1.5` m | Vertical crop relative to sensor |
| `min_range` | `0.4` m | Drop returns within this radius (self-hits) |
| `publish_costmap` | `true` | Publish the 2D `~/costmap` |
| `costmap_unknown_as` | `-1` | Non-occupied cell value (−1 unknown / 0 free) |

## 8. Run / verify

```bash
# full stack (DLIO + local map), in the localization container:
ros2 launch g1_bringup real_localization.launch.py        # local_map:=true by default

# standalone, against an already-running DLIO:
ros2 launch g1_local_map local_map.launch.py voxel_size:=0.05

# verify in RViz (LocalVoxelMap display) — NOT via ros2 topic hz from the host.
```
Expect: the floor removed, walls/obstacles within ±8 m shown as orange boxes,
refreshing as the robot moves; `~/costmap` populated.
