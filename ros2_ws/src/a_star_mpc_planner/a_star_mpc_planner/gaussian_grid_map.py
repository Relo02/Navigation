"""
Fixed-area 2.5D Gaussian grid map for local obstacle mapping.

Key design:
  - Fixed spatial extent (2*half_width x 2*half_width metres) that translates
    rigidly with the robot centre of mass — no growing or shrinking.
  - LiDAR points outside the window are silently ignored.
  - Two parallel layers per cell:
      gmap  — occupancy probability via a Gaussian CDF of XY-distance to the
              nearest obstacle (the original Go2 behaviour).
      hmap  — 2.5D extension: maximum z-height of any hit projected into the
              cell. Used by a_star_planner.py to give cells below the
              step-over threshold a reduced cost (G1 can clear ~8 cm).
              Cells with no hit are NaN.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.stats import norm


class FixedGaussianGridMap:
    """
    A 2.5D Gaussian occupancy grid with a fixed spatial extent.

    Parameters
    ----------
    reso       : float  — cell size [m]
    half_width : float  — half-extent of the square grid [m]
    std        : float  — Gaussian spread applied to each obstacle point [m]
    """

    def __init__(
        self,
        reso: float = 0.25,
        half_width: float = 5.0,
        std: float = 0.5,
        z_min: float = 0.0,
        z_max: float = 2.5,
        ground_segment_height: float = 0.20,
        ground_segment_en: bool = True,
    ):
        self.reso = float(reso)
        self.half_width = float(half_width)
        self.std = float(std)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        # Ground/obstacle segmentation: within each XY cell the lowest point is
        # taken as the local ground; points that rise more than
        # ground_segment_height above it are obstacles. The obstacle costmap is
        # built ONLY from those segmented obstacle points, so the (flat or
        # tilted) floor never becomes an obstacle and the cloud is NOT thrown
        # away — ground stays in hmap/vmap, only the XY cost layer is filtered.
        self.ground_segment_height = float(ground_segment_height)
        self.ground_segment_en = bool(ground_segment_en)

        # Number of cells along each axis — fixed for the lifetime of this object
        self.cells = int(round(2.0 * half_width / reso))
        # Z cell count for the 3D voxel layer
        self.cells_z = max(1, int(round((self.z_max - self.z_min) / self.reso)))

        # XY occupancy map — shape (cells, cells). None until first update().
        self.gmap: np.ndarray | None = None
        # 2.5D max-z layer — shape (cells, cells), NaN where no point projected.
        # Kept around because A*'s step-over rule uses this scalar height per
        # XY cell — cheaper than scanning the full 3D voxel column.
        self.hmap: np.ndarray | None = None
        # 3D occupancy voxel grid — shape (cells, cells, cells_z), float32.
        # 1.0 = at least one hit fell in this voxel, 0.0 = empty. Used for
        # 3D visualisation and downstream manipulation reach queries.
        self.vmap: np.ndarray | None = None

        # World-frame coordinates of the grid origin (bottom-left corner).
        self.minx: float = 0.0
        self.miny: float = 0.0

        # Aliases expected by the A* planner and MPC (mujoco_sim convention)
        self.xw: int = self.cells
        self.yw: int = self.cells
        self.xyreso: float = self.reso

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def segment_obstacles(pts, reso: float, height: float):
        """Return the subset of `pts` (N,3, world frame) classified as obstacles.

        Per (reso-sized) XY cell, the lowest point is the local ground; a point
        is an obstacle if it rises more than `height` above its cell's ground.
        Relative-per-cell → robust to a tilted/offset floor. Used so the
        persistent memory accumulates real obstacles, not floor returns.
        """
        if pts is None or len(pts) == 0:
            return pts
        p = np.asarray(pts, dtype=float)
        if p.shape[1] < 3:
            return p
        ix = np.floor(p[:, 0] / reso).astype(np.int64)
        iy = np.floor(p[:, 1] / reso).astype(np.int64)
        # Combine (ix, iy) into a single key; group to find per-cell min-z.
        key = ix * 1_000_003 + iy
        order = np.argsort(key, kind='stable')
        key_s = key[order]
        z_s = p[order, 2]
        uniq, start = np.unique(key_s, return_index=True)
        cell_min = np.minimum.reduceat(z_s, start)
        cell_min_per_pt = cell_min[np.searchsorted(uniq, key)]
        return p[p[:, 2] > (cell_min_per_pt + height)]

    def update(self, lidar_points, drone_pos, direct_obstacle_points=None) -> bool:
        """
        Rebuild the occupancy grid centred on drone_pos.

        Parameters
        ----------
        lidar_points : (N, 3) float array of LiDAR hits in world frame,
                       or None / empty for an obstacle-free map. These are
                       ground/obstacle SEGMENTED: per XY cell the lowest point is
                       the local ground and only points rising above it by
                       ground_segment_height feed the obstacle (XY) cost layer.
                       The full cloud is still used for hmap/vmap.
        drone_pos    : array-like [x, y, (z)]
        direct_obstacle_points : optional (M, 3) array of points that are ALREADY
                       known to be obstacles (persistent memory, fused maps) and
                       must NOT be re-segmented — they are added straight to the
                       obstacle cost layer.

        Returns
        -------
        True if at least one obstacle point was inside the grid; False otherwise.
        """
        dx = float(drone_pos[0])
        dy = float(drone_pos[1])

        # Re-centre grid origin at drone position
        self.minx = dx - self.half_width
        self.miny = dy - self.half_width
        self.xw = self.cells
        self.yw = self.cells

        # Start with an empty (zero-probability) map, NaN height layer,
        # and empty 3D voxel grid.
        self.gmap = np.zeros((self.cells, self.cells), dtype=np.float32)
        self.hmap = np.full((self.cells, self.cells), np.nan, dtype=np.float32)
        self.vmap = np.zeros(
            (self.cells, self.cells, self.cells_z), dtype=np.float32
        )

        maxx = self.minx + 2.0 * self.half_width
        maxy = self.miny + 2.0 * self.half_width

        # XY positions of points classified as obstacles (segmented live cloud
        # + direct obstacles). The floor stays in hmap/vmap but never here.
        obs_x = np.empty(0, dtype=float)
        obs_y = np.empty(0, dtype=float)

        if lidar_points is not None and len(lidar_points) > 0:
            pts = np.asarray(lidar_points, dtype=float)
            ox = pts[:, 0]
            oy = pts[:, 1]
            # z is optional — accept (N,2) clouds by defaulting to 0
            oz = pts[:, 2] if pts.shape[1] >= 3 else np.zeros_like(ox)

            mask = (ox >= self.minx) & (ox < maxx) & (oy >= self.miny) & (oy < maxy)
            ox = ox[mask]
            oy = oy[mask]
            oz = oz[mask]

            if len(ox) > 0:
                # ── 2.5D layer: max-z per cell ──
                ix_pts = ((ox - self.minx) / self.reso).astype(np.intp)
                iy_pts = ((oy - self.miny) / self.reso).astype(np.intp)
                ix_pts = np.clip(ix_pts, 0, self.cells - 1)
                iy_pts = np.clip(iy_pts, 0, self.cells - 1)
                seed = np.full((self.cells, self.cells), -np.inf, dtype=np.float32)
                np.maximum.at(seed, (ix_pts, iy_pts), oz.astype(np.float32))
                touched = np.isfinite(seed)
                self.hmap[touched] = seed[touched]

                # ── 3D layer: binary occupancy per voxel ──
                iz_pts = ((oz - self.z_min) / self.reso).astype(np.intp)
                iz_pts = np.clip(iz_pts, 0, self.cells_z - 1)
                self.vmap[ix_pts, iy_pts, iz_pts] = 1.0

                # ── Ground/obstacle segmentation ──
                # Local ground per cell = lowest point in that cell. A point is
                # an OBSTACLE if it rises more than ground_segment_height above
                # its cell's ground. Relative-per-cell, so a tilted/offset floor
                # is still classified as ground (each cell judges against its own
                # ground) and only genuine vertical structure becomes obstacle.
                if self.ground_segment_en:
                    ground = np.full((self.cells, self.cells), np.inf, dtype=np.float32)
                    np.minimum.at(ground, (ix_pts, iy_pts), oz.astype(np.float32))
                    # Min-pool over a 3x3 neighbourhood so a single high point in
                    # an otherwise-empty cell is still compared against the
                    # surrounding floor (sparse obstacles aren't lost), while a
                    # tilted floor is still judged locally.
                    ground = minimum_filter(ground, size=3, mode='constant',
                                            cval=np.inf)
                    cell_ground = ground[ix_pts, iy_pts]
                    obs_mask = oz > (cell_ground + self.ground_segment_height)
                    obs_x = ox[obs_mask]
                    obs_y = oy[obs_mask]
                else:
                    obs_x = ox
                    obs_y = oy

        # Append already-known obstacles (persistent memory, fused maps) that
        # must not be segmented away.
        if direct_obstacle_points is not None and len(direct_obstacle_points) > 0:
            dpts = np.asarray(direct_obstacle_points, dtype=float)
            dmask = (
                (dpts[:, 0] >= self.minx) & (dpts[:, 0] < maxx)
                & (dpts[:, 1] >= self.miny) & (dpts[:, 1] < maxy)
            )
            if np.any(dmask):
                obs_x = np.concatenate([obs_x, dpts[dmask, 0]])
                obs_y = np.concatenate([obs_y, dpts[dmask, 1]])

        if len(obs_x) == 0:
            return False  # only ground in view → obstacle-free cost layer

        # ── XY occupancy: Gaussian CDF of distance to the nearest obstacle ──
        # Rasterise obstacles into the grid and use a Euclidean distance
        # transform instead of the old dense (cells, cells, N_obstacles) min
        # broadcast. That broadcast was O(cells² · N) in MEMORY — with a dense
        # cloud (e.g. N≈95 k on an 800×800 grid) it tried to allocate
        # ~450 GiB and crashed the node. distance_transform_edt is O(cells²),
        # INDEPENDENT of the obstacle count, and dedupes points to cells for
        # free. The result matches the old min-distance to within one cell
        # (≈reso·√2), negligible against the inflation std.
        occupied = np.zeros((self.cells, self.cells), dtype=bool)
        ox_idx = ((obs_x - self.minx) / self.reso).astype(np.intp)
        oy_idx = ((obs_y - self.miny) / self.reso).astype(np.intp)
        in_grid = (
            (ox_idx >= 0) & (ox_idx < self.cells)
            & (oy_idx >= 0) & (oy_idx < self.cells)
        )
        occupied[ox_idx[in_grid], oy_idx[in_grid]] = True

        if not occupied.any():
            return False  # all obstacles fell outside the window → no cost layer

        # Distance (in cells) from each cell to the nearest occupied cell, ×reso
        # → metres. EDT computes distance to the nearest *False*-in-inverted, i.e.
        # nearest occupied cell.
        min_dists = distance_transform_edt(~occupied) * self.reso

        self.gmap = (1.0 - norm.cdf(min_dists, 0.0, self.std)).astype(np.float32)
        return True

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_index(self, x: float, y: float):
        """
        World coordinates -> grid indices.
        Returns (ix, iy) inside [0, cells), or (None, None) if outside.
        """
        ix = int((x - self.minx) / self.reso)
        iy = int((y - self.miny) / self.reso)
        if 0 <= ix < self.cells and 0 <= iy < self.cells:
            return ix, iy
        return None, None

    def index_to_world(self, ix: int, iy: int):
        """Grid indices -> world coordinates at cell centre."""
        return (
            ix * self.reso + self.minx,
            iy * self.reso + self.miny,
        )

    def get_probability(self, x: float, y: float) -> float:
        """Obstacle probability at world (x, y); 0.0 if outside grid or not yet updated."""
        if self.gmap is None:
            return 0.0
        ix, iy = self.world_to_index(x, y)
        if ix is None:
            return 0.0
        return float(self.gmap[ix, iy])

    def get_height(self, x: float, y: float) -> float:
        """Max-z of LiDAR hits projected into the cell at (x, y).

        Returns NaN for cells with no hit (the planner should treat those as
        unknown — usually freespace) and for queries outside the grid.
        """
        if self.hmap is None:
            return float("nan")
        ix, iy = self.world_to_index(x, y)
        if ix is None:
            return float("nan")
        return float(self.hmap[ix, iy])

    def get_voxel(self, x: float, y: float, z: float) -> float:
        """Binary occupancy of the 3D voxel at (x, y, z).

        Returns 0.0 for cells outside the grid extent or with no hit.
        """
        if self.vmap is None:
            return 0.0
        ix, iy = self.world_to_index(x, y)
        if ix is None:
            return 0.0
        iz = int((z - self.z_min) / self.reso)
        if not (0 <= iz < self.cells_z):
            return 0.0
        return float(self.vmap[ix, iy, iz])

    # ------------------------------------------------------------------
    # Convenience read-only properties
    # ------------------------------------------------------------------

    @property
    def maxx(self) -> float:
        return self.minx + 2.0 * self.half_width

    @property
    def maxy(self) -> float:
        return self.miny + 2.0 * self.half_width
