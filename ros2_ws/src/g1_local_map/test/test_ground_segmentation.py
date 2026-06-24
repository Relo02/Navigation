#!/usr/bin/env python3
"""Unit tests for the per-cell local-minimum ground segmentation (no ROS deps).

Pure numpy, so runnable with plain `pytest` or
`python3 test_ground_segmentation.py`.
"""

import numpy as np

from g1_local_map.ground_segmentation import GroundParams, segment_ground

G_HAT = np.array([0.0, 0.0, -1.0])   # odom: up = +Z
ROBOT_Z = 0.0                        # sensor at z=0; floor ~1 m below (leg_offset)
FLOOR_Z = ROBOT_Z - GroundParams().leg_offset   # = -1.0


def _grid(x0, x1, y0, y1, step, z_fn, jitter=0.0, rng=None):
    """Dense XY grid of points whose z = z_fn(x, y) (+ optional gaussian jitter)."""
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    z = z_fn(gx, gy)
    if jitter and rng is not None:
        z = z + rng.normal(0.0, jitter, size=z.shape)
    return np.stack([gx, gy, z], axis=1)


def _flat_floor(rng, floor_z=FLOOR_Z, jitter=0.005):
    return _grid(-4, 4, -4, 4, 0.05, lambda x, y: np.full_like(x, floor_z),
                 jitter=jitter, rng=rng)


def _frac_kept(obs, lo, hi):
    if obs.shape[0] == 0:
        return 0.0
    return float(((obs[:, 2] >= lo) & (obs[:, 2] <= hi)).mean())


def test_flat_floor_removed_boxes_kept():
    rng = np.random.default_rng(0)
    floor = _flat_floor(rng)
    # A solid box (0.3 m tall) standing on the floor, floor visible in its cells.
    boxes = []
    for h in np.arange(FLOOR_Z, FLOOR_Z + 0.30, 0.03):
        boxes.append(_grid(1.0, 1.4, 1.0, 1.4, 0.03, lambda x, y: np.full_like(x, h), rng=rng))
    cloud = np.vstack([floor] + boxes)

    obs = segment_ground(cloud, G_HAT, ROBOT_Z)
    # Floor at FLOOR_Z is fully removed everywhere.
    floor_left = np.sum(np.abs(obs[:, 2] - FLOOR_Z) < 0.04)
    assert floor_left == 0, f"floor not removed: {floor_left} pts"
    # The box (its raised points) survives.
    assert _frac_kept(obs, FLOOR_Z + 0.12, FLOOR_Z + 0.35) > 0.0
    assert obs.shape[0] > 50


def test_offset_floor_removed():
    """Per-cell-relative: a floor at an unexpected height (not robot_z-leg_offset)
    is still removed — the filter doesn't trust the absolute floor height."""
    rng = np.random.default_rng(1)
    floor = _flat_floor(rng, floor_z=-1.6)   # 0.6 m lower than leg_offset implies
    box = []
    for h in np.arange(-1.6, -1.6 + 0.30, 0.03):
        box.append(_grid(0.5, 0.9, 0.5, 0.9, 0.03, lambda x, y: np.full_like(x, h), rng=rng))
    cloud = np.vstack([floor] + box)

    obs = segment_ground(cloud, G_HAT, ROBOT_Z)
    assert np.sum(np.abs(obs[:, 2] + 1.6) < 0.04) == 0, "offset floor not removed"
    assert _frac_kept(obs, -1.6 + 0.12, -1.6 + 0.35) > 0.0


def test_gentle_ramp_removed_object_kept():
    rng = np.random.default_rng(2)
    slope = np.tan(np.deg2rad(5.0))   # gentle ramp (within-cell rise < band)
    ramp = _grid(-4, 4, -4, 4, 0.05, lambda x, y: FLOOR_Z + slope * x,
                 jitter=0.004, rng=rng)
    base = FLOOR_Z + slope * 2.0
    box = []
    for h in np.arange(base, base + 0.30, 0.03):
        box.append(_grid(1.9, 2.2, 0.0, 0.3, 0.03, lambda x, y: np.full_like(x, h), rng=rng))
    cloud = np.vstack([ramp] + box)

    params = GroundParams(ground_band=0.12)
    obs = segment_ground(cloud, G_HAT, ROBOT_Z, params)
    in_box_xy = (obs[:, 0] > 1.5) & (obs[:, 0] < 2.5) & (obs[:, 1] > -0.5) & (obs[:, 1] < 0.6)
    ramp_z = FLOOR_Z + slope * obs[:, 0]
    on_open_ramp = (np.abs(obs[:, 2] - ramp_z) < 0.06) & ~in_box_xy
    # Most of the open ramp is removed (a few cell-edge points may remain).
    assert on_open_ramp.sum() < 0.02 * ramp.shape[0], f"ramp not removed: {on_open_ramp.sum()}"
    # The object on the ramp survives.
    assert (in_box_xy & (obs[:, 2] - ramp_z > 0.12)).sum() > 0, "object on ramp removed"


def test_lone_obstacle_kept_via_minpool():
    """A tall thin column whose cell has no floor return is still kept, because
    the 3x3 min-pool compares it against the surrounding floor."""
    rng = np.random.default_rng(3)
    floor = _flat_floor(rng)
    # Remove floor under the column footprint, then add a tall column there.
    keep = ~((np.abs(floor[:, 0] - 0.0) < 0.2) & (np.abs(floor[:, 1] - 0.0) < 0.2))
    floor = floor[keep]
    col = []
    for h in np.arange(FLOOR_Z + 0.1, FLOOR_Z + 1.0, 0.03):
        col.append(_grid(-0.15, 0.15, -0.15, 0.15, 0.03, lambda x, y: np.full_like(x, h), rng=rng))
    cloud = np.vstack([floor] + col)

    obs = segment_ground(cloud, G_HAT, ROBOT_Z)
    kept_col = np.sum((np.abs(obs[:, 0]) < 0.2) & (np.abs(obs[:, 1]) < 0.2)
                      & (obs[:, 2] > FLOOR_Z + 0.2))
    assert kept_col > 0, "lone column wrongly removed"


def test_table_with_visible_floor_kept():
    """A raised slab with floor visible in its cells (lidar sees under it) is
    kept as obstacle (each cell's min is the floor, the slab is well above)."""
    rng = np.random.default_rng(4)
    floor = _flat_floor(rng)
    shelf = _grid(0.5, 1.5, 0.5, 1.5, 0.04,
                  lambda x, y: np.full_like(x, FLOOR_Z + 0.8), jitter=0.004, rng=rng)
    cloud = np.vstack([floor, shelf])

    obs = segment_ground(cloud, G_HAT, ROBOT_Z)
    kept_shelf = np.sum(np.abs(obs[:, 2] - (FLOOR_Z + 0.8)) < 0.05)
    assert kept_shelf > 0.5 * shelf.shape[0], "elevated slab over visible floor removed"


def test_sparse_passthrough():
    rng = np.random.default_rng(5)
    cloud = rng.uniform(-1, 1, size=(50, 3))   # < min_total
    obs, info = segment_ground(cloud, G_HAT, ROBOT_Z, return_info=True)
    assert info["status"] == "sparse"
    assert obs.shape[0] == 50   # passthrough (all within max_height above foot)


def test_empty_cloud_no_crash():
    obs = segment_ground(np.empty((0, 3)), G_HAT, ROBOT_Z)
    assert obs.shape == (0, 3)


def test_info_fields_present():
    rng = np.random.default_rng(6)
    obs, info = segment_ground(_flat_floor(rng), G_HAT, ROBOT_Z, return_info=True)
    for k in ("status", "n_pts", "n_cells", "candidate_cells", "seed_cells",
              "ground_cells", "manifold_found"):
        assert k in info, f"missing info key {k}"
    assert info["status"] == "ok"
    assert info["ground_cells"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
