# Ground segmentation — per-cell local-minimum filter

How `g1_local_map` removes the floor from the accumulated DLIO cloud before it
becomes the planner's obstacle map. Code:
[`g1_local_map/ground_segmentation.py`](../ros2_ws/src/g1_local_map/g1_local_map/ground_segmentation.py)
(`segment_ground`), called every scan by `local_voxel_map_node`. See
[LOCAL_VOXEL_MAP.md](LOCAL_VOXEL_MAP.md) for the surrounding pipeline.

> This **replaced** an earlier gravity-aware SVD plane-fit method (documented in
> [GROUND_REMOVAL_PLAN.md](GROUND_REMOVAL_PLAN.md)). That method's planarity /
> flatness *ratios* are inflated by the 0.10 m voxel quantization, so most
> flat-floor cells were rejected as "not planar" and the floor was kept as
> obstacles (the robot boxed itself in). The local-minimum rule needs no such
> ratios and is per-cell relative, so it tolerates voxelization, a tilted/offset
> floor and sensor-height uncertainty without tuning.

---

## 1. Why ground removal happens here (not in DLIO)

DLIO ingests the **raw** cloud — the ground is a pitch/roll/Z constraint its
odometry needs, so removing it upstream degrades localization. The floor is
removed **downstream**, on the temporally-accumulated odom-frame voxel cloud
`local_voxel_map` already maintains. The output (`/local_voxel_map/obstacles`) is
the ground-removed cloud the A\*+MPC planner consumes.

The frame matters: DLIO's odom origin sits at the **sensor** (~1 m above the
floor), so the floor lands near `z ≈ −1 m` and its exact height drifts a little.
Any rule keyed to an absolute floor height is therefore fragile — which is the
whole reason for a **per-cell relative** method.

---

## 2. The algorithm

Input: an `(N,3)` accumulated cloud of **0.10 m voxel centres** in the
gravity-aligned `odom` frame, plus the gravity vector `g_hat` (≈ `(0,0,−1)`) and
the robot/sensor `z`.

```
            ┌─ tile into XY cells (ground_cell) ─┐
points ─────┤  per-cell min height (along -g)    ├─ 3x3 min-pool ─┐
            └─ ignore cells < min_pts            ┘                │
                                                                  ▼
        label each point by rise above its cell's local ground:
          rise ≤ ground_band            → GROUND   (drop)
          ground_band < rise ≤ max_height → OBSTACLE (keep)
          rise > max_height             → CEILING  (drop)
```

1. **Tile** the cloud into XY cells of size `ground_cell` (0.40 m). Heights are
   measured along **gravity-up** (`−g_hat`), so the filter is correct even if
   odom's up isn't exactly `+Z`.
2. **Per-cell local ground = the minimum height** in that cell. Cells with fewer
   than `ground_min_pts` points are ignored when setting ground, so a single
   stray-low point can't define it.
3. **3×3 min-pool** the per-cell minima (pure-numpy shift-and-min). This is the
   key step: a cell that holds *only* a tall obstacle (no floor return of its
   own) is compared against the **surrounding** floor, so the obstacle isn't
   mistaken for its own ground. Empty / under-populated cells are `+∞` and never
   lower a neighbour.
4. **Label** each point by how far it rises above its cell's pooled ground:
   ground (`≤ ground_band`, dropped), obstacle (`ground_band … max_height`,
   kept), or ceiling (`> max_height`, dropped).

**Fail-safe by construction:** an empty / `< ground_min_total` cloud passes
through (capped at `max_height` above the foot), and a cell whose whole 3×3
neighbourhood has no valid ground also fails open (keeps its geometry). Better a
cluttered costmap than a blind one.

It is pure-numpy and vectorised (no Python loop over points or cells, no SciPy),
so it runs every scan well within the ~10 Hz budget.

---

## 3. Parameters

In [`config/local_map.yaml`](../ros2_ws/src/g1_local_map/config/local_map.yaml):

| Param | Default | Meaning |
|---|---|---|
| `ground_cell` | `0.40` m | XY tile size for the local minimum |
| `ground_min_pts` | `12` | min points for a cell to define its ground; **lower** to clear sparse/far floor, **raise** to ignore thin noise |
| `ground_band` | `0.10` m | rise above local ground still counted as ground; must be **≥ `voxel_size`** and **< the smallest obstacle you want to keep** |
| `max_height` | `2.0` m | ignore returns this far above the ground (ceilings/overhangs) |
| `ground_leg_offset` | `1.0` m | sensor-z → foot-height drop; only used to cap fail-open points |
| `ground_min_total` | `200` | below this many points, pass the cloud through |

The legacy SVD knobs (`ground_planarity_max`, `ground_flat_max`,
`ground_slope_tol_deg`, `ground_step_tol`, `ground_seed_band`) are **ignored** by
this filter; they remain in the config only so older param files load.

### Tuning by symptom
- **Floor still kept as obstacles** → raise `ground_band` (toward, but below,
  your smallest real obstacle), and/or lower `ground_min_pts` (far/sparse floor
  cells weren't defining ground).
- **Low real obstacles (curbs, cables) disappear** → lower `ground_band` below
  their height.
- **A gentle ramp is kept** → the within-cell rise exceeds `ground_band`; either
  raise `ground_band` above `tan(slope)·ground_cell` or shrink `ground_cell`.

---

## 4. Guarantees and limitations

**Handles well**
- Flat floor at **any** height / mild tilt (per-cell relative).
- Voxelized clouds (no planarity ratio to inflate).
- Walls, furniture, low obstacles, lone columns (kept via the 3×3 min-pool).
- A raised slab/table with **visible floor in its cells** (lidar sees under it):
  each cell's min is the floor, the slab is well above → kept.

**Known trade-off**
- A **large solid slab that fully occludes the floor beneath it** (the lidar
  can't see under it) has interior cells with no lower neighbour within the 3×3
  window, so its *interior* reads as local ground. Its **edges and anything
  standing on it still register as obstacles**, so the planner still sees a
  blocking footprint. (The previous SVD method kept such slabs whole, at the cost
  of being unusable on the voxelized floor — the trade we made for robustness.)
  Widening the min-pool window would shrink this effect at some compute cost.
- **Steep ramps**: removal is per-cell, so a ramp is removed only up to ~one
  `ground_band` of rise per `ground_cell`. See tuning above.

---

## 5. Heartbeat & verification

`local_voxel_map_node` logs a ~1 Hz heartbeat:

```
in=… cropped=… accum_vox=… cand=… seed=… ground_cells=… obstacles=… | robot=(x,y,z) zband=[…]
```

For this filter `cand`/`seed`/`ground_cells` all report the number of cells that
defined a ground level. The number that matters is **`obstacles` ≪ `accum_vox`**
once the floor clears (otherwise the floor is still being kept). Verify visually
in **RViz** (`LocalVoxelMap`) — the floor should be dark, with orange voxels only
on walls/objects. Run the unit tests with:

```bash
cd ros2_ws/src/g1_local_map && python3 -m pytest test/test_ground_segmentation.py -q
```
