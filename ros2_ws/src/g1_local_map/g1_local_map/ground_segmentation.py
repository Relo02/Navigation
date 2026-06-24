#!/usr/bin/env python3
"""
Per-cell local-minimum ground segmentation (pure-numpy, no ROS deps).

Robust, voxelization-friendly ground remover for the DLIO **accumulated
odom-frame voxel cloud** the local map builds. Replaces an earlier gravity-aware
SVD/eigen per-cell plane-fit: that method's planarity/flatness ratios are
inflated by the 0.10 m voxel quantization, so most flat-floor cells were rejected
as "not planar" and the floor was kept as obstacles. The local-minimum rule needs
no such ratios and is per-cell relative, so it tolerates the voxel grid, a
tilted/offset floor and sensor-height uncertainty without tuning.

Pipeline:

  1. Tile the cloud into XY cells of size ``cell``. Each cell's local ground
     height is the **minimum** point height (heights measured along gravity-up,
     ``-g_hat``; that is +Z for DLIO odom). Cells with < ``min_pts`` points are
     ignored when setting ground, so a lone stray-low point can't define it.
  2. **Min-pool** the per-cell minima over a 3x3 neighbourhood, so a cell that
     holds only a tall obstacle (and no floor return) is compared against the
     surrounding floor instead of treating the obstacle as its own ground.
  3. Label each point by how far it rises above its cell's local ground:
     ``<= ground_band`` -> ground (drop); ``ground_band .. max_height`` ->
     obstacle (keep); ``> max_height`` -> ceiling (drop).

Fail-safe: an empty/too-sparse cloud passes through (capped at ``max_height``
above the foot); cells whose 3x3 neighbourhood has no valid ground also fail
open. Better a cluttered costmap than a blind one.

Limitation vs a plane-fit method: a LARGE solid elevated slab that fully occludes
the floor beneath it (e.g. a big tabletop the lidar can't see under) has interior
cells with no lower neighbour within the 3x3 window, so its interior reads as
local ground; its edges — and anything standing on it — still register as
obstacles. For navigation this is an acceptable trade for robustness. Within-cell
slope (ramps) is handled only up to ~``ground_band`` of rise across a cell;
widen ``ground_band`` or shrink ``cell`` for steep ramps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundParams:
    """Tunables for :func:`segment_ground`.

    Used by the local-minimum filter: ``cell``, ``min_pts``, ``ground_band``,
    ``leg_offset``, ``max_height``, ``min_total``. The remaining fields
    (``planarity_max``, ``flat_max``, ``slope_tol_deg``, ``step_tol``,
    ``seed_band``) are retained for config/back-compat and are ignored.
    """

    cell: float = 0.40            # XY tile size for the local-minimum (m)
    min_pts: int = 12             # min points for a cell to define its ground
    planarity_max: float = 0.10   # (unused) legacy SVD planarity bound
    flat_max: float = 0.05        # (unused) legacy SVD absolute flatness (m)
    slope_tol_deg: float = 30.0   # (unused) legacy SVD slope tolerance (deg)
    step_tol: float = 0.08        # (unused) legacy SVD region-grow step (m)
    ground_band: float = 0.06     # rise above local ground still counted as ground (m)
    seed_band: float = 0.15       # (unused) legacy SVD seed window (m)
    leg_offset: float = 1.0       # robot_z (sensor, odom) -> foot height drop (m)
    max_height: float = 2.0       # ignore points this far above ground (m)
    min_total: int = 200          # below this many points, pass the cloud through


def segment_ground(xyz: np.ndarray, g_hat, robot_z: float,
                   params: GroundParams = GroundParams(),
                   return_info: bool = False):
    """Remove the ground from an accumulated odom-frame cloud (local-minimum).

    Args:
        xyz: (N, 3) float array of points in the gravity-aligned **odom** frame.
        g_hat: gravity unit vector (DLIO odom: ~(0, 0, -1) -> up = +Z).
        robot_z: robot/sensor Z in odom; sets the foot height
            (``robot_z - leg_offset``) used to cap fail-open points.
        params: :class:`GroundParams`.
        return_info: also return a diagnostics dict (cell/ground counts).

    Returns:
        ``obstacle_xyz`` (M, 3), or ``(obstacle_xyz, info)`` if ``return_info``.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n_pts = xyz.shape[0]
    foot = float(robot_z) - params.leg_offset

    # "up" = -gravity, normalised. Heights are measured along it, so the filter
    # is correct even if odom gravity isn't exactly +Z.
    up = -np.asarray(g_hat, dtype=np.float64).reshape(3)
    up_norm = np.linalg.norm(up)
    up = up / up_norm if up_norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    def _info(status, ground_cells, n_cells):
        # candidate_cells / seed_cells kept for the node heartbeat's field names.
        return {"status": status, "n_pts": n_pts, "n_cells": n_cells,
                "candidate_cells": ground_cells, "seed_cells": ground_cells,
                "ground_cells": ground_cells, "manifold_found": ground_cells > 0}

    def _passthrough(status):
        if n_pts == 0:
            out = xyz.reshape(0, 3)
        else:
            above = (xyz @ up) - foot
            out = xyz[above <= params.max_height]
        return (out, _info(status, 0, 0)) if return_info else out

    # Failsafe: too few points for reliable per-cell minima -> pass through.
    if n_pts < params.min_total:
        return _passthrough("sparse")

    h = xyz @ up
    # Horizontal cell indices (x, y are horizontal in the gravity-aligned frame).
    gx = np.floor(xyz[:, 0] / params.cell).astype(np.int64)
    gy = np.floor(xyz[:, 1] / params.cell).astype(np.int64)
    gx0, gy0 = int(gx.min()), int(gy.min())
    ix = (gx - gx0).astype(np.intp)
    iy = (gy - gy0).astype(np.intp)
    W = int(ix.max()) + 1
    H = int(iy.max()) + 1

    # Per-cell minimum height + point count.
    cell_min = np.full((W, H), np.inf, dtype=np.float64)
    np.minimum.at(cell_min, (ix, iy), h)
    cell_cnt = np.zeros((W, H), dtype=np.int64)
    np.add.at(cell_cnt, (ix, iy), 1)
    # A lone low point must not define ground: ignore under-populated cells.
    cell_min[cell_cnt < params.min_pts] = np.inf

    # 3x3 neighbourhood min-pool (pure-numpy shift-and-min). Empty / under-
    # populated cells are +inf, so they never lower a neighbour's ground.
    pooled = cell_min.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.full((W, H), np.inf, dtype=np.float64)
            di0, di1 = max(0, -di), W - max(0, di)     # destination row span
            si0, si1 = max(0, di), W - max(0, -di)      # source row span
            dj0, dj1 = max(0, -dj), H - max(0, dj)
            sj0, sj1 = max(0, dj), H - max(0, -dj)
            shifted[di0:di1, dj0:dj1] = cell_min[si0:si1, sj0:sj1]
            pooled = np.minimum(pooled, shifted)

    ground = pooled[ix, iy]
    valid = np.isfinite(ground)
    # Valid cells: rise above local ground. Fail-open cells: rise above the foot.
    above = np.where(valid, h - ground, h - foot)
    keep = (above > params.ground_band) & (above <= params.max_height)

    obstacles = xyz[keep]
    if return_info:
        n_cells = int((cell_cnt > 0).sum())
        ground_cells = int(np.isfinite(cell_min).sum())
        return obstacles, _info("ok", ground_cells, n_cells)
    return obstacles
