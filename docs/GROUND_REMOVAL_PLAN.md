# Ground removal redesign — gravity-aware, SVD plane segmentation

Plan to replace the current ground-removal logic with a method that fits local
planes by **SVD** and classifies them against the **DLIO-estimated gravity
vector**, so the floor is removed correctly while the robot pitches/rolls during
gait and while it traverses surfaces that are **tilted differently** (ramps,
slopes, multi-level floors).

Status: design. No code yet. Supersedes `g1_local_map/ground_removal_node.py`
(to be deleted) and `segment_obstacles()` in `local_voxel_map_node.py`
(to be replaced).

---

## 1. Why the current approach is wrong

There are **two** ground-removal stages today; both are flawed.

### 1a. `ground_removal_node.py` (pre-DLIO RANSAC) — **delete**
```
/livox/lidar (raw, sensor frame) → ground_removal → /livox/lidar_filtered → DLIO
```
- **Assumes sensor-frame z = up.** It picks ground candidates by the lower-z band
  and accepts a plane only if `|n_z| > 0.85`. The MID-360 is on the torso, so the
  sensor frame **pitches and rolls with every step**; the true ground normal is
  not aligned with sensor-z during gait, so the node either misses the floor or
  (worse) accepts a tilted wall/ramp as "horizontal".
- **Single global RANSAC plane.** One plane cannot represent a floor + a ramp +
  a step. On any non-flat terrain it averages them and mis-labels inliers.
- **It feeds DLIO.** The ground is a strong pitch/roll/Z constraint for LiDAR-
  inertial odometry; removing it *before* DLIO degrades the odometry (the node's
  own docstring admits this). Ground removal belongs **downstream** of DLIO, on
  the cloud that feeds the planner — not on DLIO's input.
- 3-point RANSAC on a sparse single scan is noisier than an SVD fit over an
  accumulated patch.

**Decision:** delete this node; DLIO consumes the raw `/livox/lidar`.

### 1b. `segment_obstacles()` in `local_voxel_map_node.py` — **replace**
Per-cell lowest-point: bin XY into `ground_cell` squares, take each cell's lowest
point as local ground, keep points whose height above it is in
`(ground_thresh, max_height]`.
- Operates in the **odom frame** (gravity-aligned by DLIO at init) so it is *not*
  broken by gait tilt — good. But:
- **No plane orientation.** "Height above the lowest point in the cell" is not a
  plane fit. A curb, a box edge, or a sloped cell makes the lowest point
  unrepresentative → ground bleeds into obstacles or vice-versa.
- **Sensitive to the single lowest point** (one stray low return per cell shifts
  the whole cell's ground datum down).
- **No use of gravity beyond the +Z assumption**, which is only as good as DLIO's
  one-shot init gravity estimate; a 1–2° residual tilt biases far cells.

**Decision:** replace with gravity-aware SVD segmentation (below).

---

## 2. Design

### 2.1 Where it runs
Downstream of DLIO, on the **accumulated odom-frame cloud** the local map already
builds (a single MID-360 scan is too sparse — ~1 pt/ground-cell — to fit planes;
accumulation is required, as the local map notes). Implement it as the new
ground stage **inside `local_voxel_map_node.py`**, replacing `segment_obstacles`,
or as a small reusable module imported by it. No new topic in the hot path.

```
/dlio/.../pointcloud/deskewed (odom) ┐
/dlio/.../odom (gravity-aligned)     ├→ local_voxel_map: accumulate → GRAVITY-SVD GROUND REMOVAL → obstacles → voxel grid + costmap
gravity vector g (from DLIO)         ┘
```

### 2.2 The gravity vector from DLIO
DLIO gravity-aligns the odom frame at initialization (`gravity_align_`,
`FromTwoVectors(grav_vec, +Z)` in `odom.cc`), so in the **odom frame**:
```
g_hat ≈ (0, 0, -1)      # "up" is +Z, by construction
```
The cloud and map are published in odom, so we get gravity **for free** as a
constant. Use `g_hat` explicitly (not a hard-coded z index) so the algorithm is
correct if we ever process a non-gravity-aligned frame, and so a future refined
gravity (e.g. from the live odom orientation `q`: `g_sensor = R(q)^T·(0,0,-1)`)
drops in without touching the segmentation.

> **Validate the assumption once:** fit a plane to a known-flat floor patch and
> check its normal is within ~2° of `g_hat`. A larger error means DLIO's init
> gravity is off → fix calibration first (see DLIO_G1_MID360_TUNING §2.2), the
> segmentation can't paper over a tilted world frame.

### 2.3 Algorithm — per-region SVD + gravity classification + region growing
Operates on the accumulated cloud `P` (odom frame) each publish tick.

1. **Tile** P into XY cells of size `cell` (≈0.3–0.5 m — large enough for a
   stable SVD, small enough to follow slope changes). Drop cells with
   `< min_pts` points.
2. **SVD plane per cell.** For cell points `Q`: `c = mean(Q)`;
   `U,S,Vt = svd(Q − c)`; plane normal `n = Vt[2]` (smallest singular vector),
   oriented so `n·g_hat > 0`. Planarity = `S[2] / (S[1] + eps)` (thin → planar).
   Record per cell: `c`, `n`, `S`, point indices.
3. **Ground-candidate test per cell** — a cell is a *candidate* ground patch if:
   - **planar**: `S[2]/S[1] < planarity_max` (e.g. 0.1), and `S[2] < flat_max`;
   - **gravity-aligned within slope tolerance**:
     `angle(n, g_hat) < slope_tol_deg` (e.g. 30–35° — admits ramps/slopes but
     rejects walls); this is the core of "planes tilted differently".
4. **Region-grow the true ground** from seeds, to reject elevated horizontal
   slabs (shelf tops, table tops, the robot's own flat parts):
   - **Seeds**: candidate cells near/under the robot at the expected foot height
     (`robot_z − leg_offset ± seed_band`, robot_z from `/dlio/.../odom`).
   - Grow to 8-neighbour candidate cells whose plane is **continuous** at the
     shared edge: predicted height of cell A's plane at cell B's centre differs
     from B's plane height by `< step_tol` (e.g. 0.08 m). This lets the ground
     surface bend (ramp) while breaking at curbs/steps taller than `step_tol`.
   - The connected component is the **ground manifold** (possibly tilted, multi-
     sloped). Non-connected horizontal slabs above it stay as obstacles.
5. **Point labelling.** For each point, find its cell:
   - cell in the ground manifold → signed distance to that cell's plane along
     `n`: `d = (p − c)·n`. `|d| ≤ ground_band` → **ground (drop)**; `d >
     ground_band` and `≤ max_height` → **obstacle (keep)**; `d < −ground_band`
     (below ground, e.g. noise) → drop.
   - cell not in the manifold (no plane / non-candidate) → **keep** all its points
     up to `max_height` (fail-open: never silently delete unclassified geometry).
6. **Output** the kept (obstacle) points → existing voxel grid + 2D costmap path.

### 2.4 Failsafes (never blank the obstacle cloud)
- Empty / `< min_total` points → pass through unchanged.
- No ground manifold found (no seeds, e.g. robot on a table) → keep everything;
  log a throttled warning. Better a cluttered costmap than a blind one.
- Height cap `max_height` always applied (ignore ceiling/high shelves).
- Per-tick wall-clock budget: cap cell count / subsample dense cells so the node
  keeps up at the cloud rate.

---

## 3. Parameters (new `local_map.yaml` block)
| param | meaning | start |
|---|---|---|
| `ground/cell` | XY tile size for SVD (m) | 0.40 |
| `ground/min_pts` | min points to fit a cell plane | 12 |
| `ground/planarity_max` | `S3/S2` upper bound for "planar" | 0.10 |
| `ground/flat_max` | `S3` upper bound (m), absolute flatness | 0.05 |
| `ground/slope_tol_deg` | max plane↔gravity angle counted as ground | 30 |
| `ground/step_tol` | max height jump across a cell edge to keep growing (m) | 0.08 |
| `ground/ground_band` | \|dist to ground plane\| ≤ this ⇒ ground (m) | 0.06 |
| `ground/seed_band` | foot-height window for seed cells (m) | 0.15 |
| `ground/leg_offset` | robot_z (sensor) → foot height drop (m) | from URDF |
| `ground/max_height` | ignore points this far above ground (m) | 2.0 |

Tune `slope_tol_deg` to the steepest ramp to traverse; `step_tol` to the smallest
obstacle/curb height that must survive (anything ≥ `step_tol` becomes an
obstacle).

---

## 4. Implementation steps
1. **New module** `g1_local_map/ground_segmentation.py` with a pure function
   `segment_ground(xyz, g_hat, robot_z, params) -> obstacle_xyz` (no ROS deps →
   unit-testable). Use numpy; vectorise the per-cell SVD via `np.linalg.svd` on a
   batched covariance, or `np.linalg.eigh` of per-cell 3×3 covariances (faster).
2. **Wire into** `local_voxel_map_node.py`: replace the `segment_obstacles(...)`
   call with `segment_ground(raw_centers, g_hat, robot_z, ...)`; pass
   `g_hat=(0,0,-1)` (odom) and `robot_z` from the cached odom.
3. **Delete** `ground_removal_node.py`, its `console_scripts` entry in
   `g1_local_map/setup.py`, and the `ground_removal` node + `ground_removal:=`
   arg + `/livox/lidar_filtered` wiring in `g1_bringup/real_localization.launch.py`
   (DLIO now subscribes raw `/livox/lidar`). Update `system_architecture.md` and
   `LOCAL_VOXEL_MAP.md` diagrams.
4. **Docs**: fold the new params into `DLIO_G1_MID360_TUNING.md` (§ ground) and
   note in `DLIO_DEPLOYMENT_TESTING.md` Phase 5/6 that there is no pre-DLIO ground
   filter anymore.

---

## 5. Validation
| test | expectation |
|---|---|
| Flat floor, boxes | floor fully removed; boxes/cones kept intact down to `step_tol` |
| Robot pitch/roll during gait | floor stays removed (odom-frame invariance) — diff vs old sensor-frame node |
| Ramp / slope (≤ `slope_tol_deg`) | ramp removed; objects on the ramp kept |
| Curb / step ≥ `step_tol` | step edge survives as an obstacle (region growth breaks) |
| Elevated flat slab (shelf top) | NOT removed (not connected to the foot-level seed) |
| Sparse / empty cloud | pass-through, no crash, throttled warning |
| Gravity check | floor-patch normal within ~2° of `g_hat` |

Compare obstacle clouds / costmaps old-vs-new in RViz on the same bag, and
confirm DLIO odometry is **unchanged/better** now that it gets the raw cloud
(removing 1a should help odometry, not hurt it).

---

## 6. Open questions
- **Gravity source:** start with constant `g_hat=(0,0,-1)` in odom. If a slow
  world-tilt is observed over large maps, switch to per-scan gravity from the
  live odom orientation (already available) — the API is built for it.
- **Cost vs accuracy:** if per-cell SVD is too slow at the cloud rate, fall back
  to per-cell covariance eigendecomposition (3×3, closed form) or coarsen `cell`.
- **Single vs accumulated:** keep it on the accumulated cloud (density). If a
  lower-latency obstacle layer is later needed, a scan-rate variant would need a
  denser sensor model than one MID-360 sweep provides.
