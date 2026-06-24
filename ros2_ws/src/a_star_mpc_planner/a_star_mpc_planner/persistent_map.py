"""
persistent_map.py
Sparse world-frame obstacle accumulator for rolling-horizon A* planning.

LiDAR hits are stored as discretised world-frame cell indices so they survive
across planning cycles even when obstacles leave the sensor field of view.
Cells expire after `decay_sec` seconds so dynamic obstacles are eventually
forgotten (set decay_sec=0 to keep everything forever in static environments).

This map is used alongside — not instead of — the FixedGaussianGridMap: the
A* node merges the persistent points with the current LiDAR scan before
calling FixedGaussianGridMap.update(), so the Gaussian inflation and A* cost
pipeline remains completely unchanged.
"""

import numpy as np


class PersistentOccupancyMap:
    """
    Sparse world-frame obstacle accumulator.

    Parameters
    ----------
    grid_reso  : cell size [m] — should match the A* grid resolution
    decay_sec  : seconds before an unconfirmed cell is evicted (0 = never)
    max_cells  : hard cap on stored cells to bound memory usage
    """

    def __init__(
        self,
        grid_reso: float = 0.25,
        decay_sec: float = 30.0,
        max_cells: int   = 50_000,
        obstacle_height: float = 0.30,
    ):
        self._reso      = float(grid_reso)
        self._decay_sec = float(decay_sec)
        self._max_cells = int(max_cells)
        # Height assigned to recalled obstacle points. The A* grid's 2.5D
        # step-over rule treats a cell as steppable (free) when its max-z is
        # below step_over_height (~0.08 m). Returning persistent obstacles at
        # z=0 (the old behaviour) made every accumulated cell look like floor,
        # so they never blocked the planner. Return them clearly above the
        # step-over height so the recalled obstacles actually obstruct A*.
        self._obstacle_height = float(obstacle_height)

        # (ix_world, iy_world) -> last_confirmed_time  [float seconds]
        self._cells: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, lidar_points: np.ndarray | None, now: float) -> None:
        """
        Ingest a new LiDAR scan and evict stale cells.

        Parameters
        ----------
        lidar_points : (N, 3) float array of obstacle points in world frame,
                       or None / empty if the scan is unavailable.
        now          : current time in seconds (use ROS clock for sim compatibility)
        """
        if lidar_points is not None and len(lidar_points) > 0:
            ix = np.round(lidar_points[:, 0] / self._reso).astype(int)
            iy = np.round(lidar_points[:, 1] / self._reso).astype(int)
            for i, j in zip(ix, iy):
                self._cells[(int(i), int(j))] = now

        self._evict(now)

    def get_points_in_window(
        self,
        minx: float, miny: float,
        maxx: float, maxy: float,
    ) -> np.ndarray | None:
        """
        Return world-frame (x, y, 0) obstacle points that fall inside
        [minx, maxx) × [miny, maxy).  Returns None if the window is empty.
        """
        reso = self._reso
        pts  = [
            (ix * reso, iy * reso)
            for (ix, iy) in self._cells
            if minx <= ix * reso < maxx and miny <= iy * reso < maxy
        ]
        if not pts:
            return None
        arr = np.empty((len(pts), 3), dtype=np.float32)
        for k, (x, y) in enumerate(pts):
            arr[k, 0] = x
            arr[k, 1] = y
            arr[k, 2] = self._obstacle_height
        return arr

    @property
    def size(self) -> int:
        return len(self._cells)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict(self, now: float) -> None:
        """Remove expired and excess cells."""
        if self._decay_sec > 0:
            cutoff     = now - self._decay_sec
            stale      = [k for k, t in self._cells.items() if t < cutoff]
            for k in stale:
                del self._cells[k]

        overflow = len(self._cells) - self._max_cells
        if overflow > 0:
            # Drop the oldest entries first
            oldest = sorted(self._cells, key=self._cells.__getitem__)[:overflow]
            for k in oldest:
                del self._cells[k]
