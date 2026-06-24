#!/usr/bin/env python3
"""
Local rolling 3D voxel map for G1 navigation — feeds the a_star_mpc planner.

Per incoming scan (~10 Hz):

    /dlio/odom_node/pointcloud/deskewed  (PointCloud2, odom frame, deskewed)
    /dlio/odom_node/odom                 (Odometry, robot pose in odom)
              │
              ▼
    1. crop to a robot-centred rolling window (±half_width, vertical band)
    2. accumulate the RAW cropped points into a voxel grid (odom frame) with
       per-voxel temporal decay  ── this DENSIFIES the sparse MID-360 scans
    3. ground removal on the ACCUMULATED (dense) cloud — gravity-aware per-cell
       SVD/eigen plane segmentation (ground_segmentation.segment_ground; see
       docs/GROUND_REMOVAL_PLAN.md and docs/LOCAL_VOXEL_MAP.md)
    4. publish:
         <ns>/obstacles    PointCloud2    ground-removed obstacle voxel centres,
                                          odom frame → planner `obstacle_topic`
         <ns>/voxel_grid   PointCloud2    same points, for RViz / 3D queries
         <ns>/costmap      OccupancyGrid  2D column projection (quick costmap)

Order matters: ground removal runs AFTER accumulation, never on a single scan.
A lone MID-360 scan has ~1 point per ground cell, so per-cell segmentation would
treat every point as its own ground and drop the whole cloud. Accumulating first
gives each cell enough points for the floor to be the clear minimum.

Why not /dlio/map_node/map? That global SLAM map only republishes on a new
keyframe (~1 m / 45 deg of motion), far too laggy for local avoidance. This node
updates every scan and forgets what the robot walked past (decay).
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

from g1_local_map.ground_segmentation import GroundParams, segment_ground


def read_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract an (N, 3) float64 array of xyz from a PointCloud2 (NaNs dropped).

    Uses read_points (structured array), NOT read_points_numpy: the latter
    asserts EVERY field in the cloud shares one datatype — even when you only
    request x/y/z — which fails on the DLIO deskewed cloud, whose x/y/z (float32)
    sit alongside intensity/timestamp/ring fields of other datatypes
    ("AssertionError: All fields need to have the same datatype"). read_points
    handles mixed-datatype clouds and we pull only the xyz columns.
    """
    rec = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    if not isinstance(rec, np.ndarray):
        # Older sensor_msgs_py returns an iterable of (x, y, z) tuples.
        rec = np.array(list(rec))
    if rec.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if rec.dtype.names:
        return np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
    return rec.astype(np.float64).reshape(-1, 3)


class VoxelAccumulator:
    """Robot-centred rolling voxel occupancy with per-voxel temporal decay.

    Voxels are keyed by integer indices in the odom frame (so the grid never
    needs re-centring); each stores the last time it was hit. Voxels older than
    ``persistence_s`` or outside the rolling window are evicted, bounding memory
    and forgetting obstacles the robot has passed.
    """

    def __init__(self, voxel_size: float, persistence_s: float):
        self.voxel_size = float(voxel_size)
        self.persistence_s = float(persistence_s)
        self._last_seen: dict[tuple[int, int, int], float] = {}

    def update(self, xyz: np.ndarray, now_s: float) -> None:
        if xyz.shape[0]:
            idx = np.floor(xyz / self.voxel_size).astype(np.int64)
            for k in map(tuple, np.unique(idx, axis=0).tolist()):
                self._last_seen[k] = now_s

    def prune(self, now_s: float, center_xy: tuple[float, float], half_width: float) -> None:
        if not self._last_seen:
            return
        cx, cy = center_xy
        reach = (half_width + self.voxel_size) / self.voxel_size
        ix0, iy0 = cx / self.voxel_size, cy / self.voxel_size
        stale = now_s - self.persistence_s
        dead = [
            k for k, t in self._last_seen.items()
            if t < stale or abs(k[0] - ix0) > reach or abs(k[1] - iy0) > reach
        ]
        for k in dead:
            del self._last_seen[k]

    def centers(self) -> np.ndarray:
        """(M, 3) float32 array of occupied voxel centre points in odom frame."""
        if not self._last_seen:
            return np.empty((0, 3), dtype=np.float32)
        keys = np.asarray(list(self._last_seen.keys()), dtype=np.float64)
        return ((keys + 0.5) * self.voxel_size).astype(np.float32)


class LocalVoxelMapNode(Node):
    def __init__(self):
        super().__init__("local_voxel_map")

        p = self.declare_parameter
        self.cloud_topic = p("cloud_topic", "/dlio/odom_node/pointcloud/deskewed").value
        self.odom_topic = p("odom_topic", "/dlio/odom_node/odom").value
        self.obstacles_topic = p("obstacles_topic", "~/obstacles").value
        self.voxel_topic = p("voxel_topic", "~/voxel_grid").value
        self.costmap_topic = p("costmap_topic", "~/costmap").value

        self.half_width = float(p("half_width", 8.0).value)        # rolling window half-extent (m)
        self.voxel_size = float(p("voxel_size", 0.10).value)       # 3D voxel edge (m) = costmap reso
        self.max_height = float(p("max_height", 2.0).value)        # ignore points this far above ground (m)
        self.z_below = float(p("z_below", 1.5).value)              # vertical crop below robot/sensor (m)
        self.z_above = float(p("z_above", 1.5).value)              # vertical crop above robot/sensor (m)
        self.persistence_s = float(p("persistence_s", 3.0).value)  # voxel memory before decay (s)
        self.min_range = float(p("min_range", 0.4).value)          # drop self-hits within this radius (m)
        self.publish_costmap = bool(p("publish_costmap", True).value)
        self.costmap_unknown_as = int(p("costmap_unknown_as", -1).value)  # -1 unknown / 0 free

        # ── per-cell local-minimum ground segmentation (see ground_segmentation.py) ──
        self.ground_params = GroundParams(
            cell=float(p("ground_cell", 0.40).value),            # XY tile for the plane fit (m)
            min_pts=int(p("ground_min_pts", 12).value),          # min points to fit a cell plane
            planarity_max=float(p("ground_planarity_max", 0.10).value),  # sqrt(l0/l1) "planar" bound
            flat_max=float(p("ground_flat_max", 0.05).value),    # sqrt(l0) absolute flatness (m)
            slope_tol_deg=float(p("ground_slope_tol_deg", 30.0).value),  # max plane<->gravity angle (deg)
            step_tol=float(p("ground_step_tol", 0.08).value),    # max edge height jump to keep growing (m)
            ground_band=float(p("ground_band", 0.06).value),     # |dist to plane| <= this => ground (m)
            seed_band=float(p("ground_seed_band", 0.15).value),  # foot-height window for seeds (m)
            leg_offset=float(p("ground_leg_offset", 1.0).value),  # robot_z(sensor) -> foot drop (m)
            max_height=self.max_height,
            min_total=int(p("ground_min_total", 200).value),     # below this, pass cloud through
        )
        # odom is gravity-aligned by DLIO at init, so "up" is +Z and g points down.
        self.g_hat = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        self.frame_id = "odom"
        self.robot_xyz = np.zeros(3, dtype=np.float64)
        self.have_odom = False
        self.accum = VoxelAccumulator(self.voxel_size, self.persistence_s)

        # ── QoS ──────────────────────────────────────────────────────────────
        # Cloud sub BEST_EFFORT: DLIO publishes the large deskewed cloud
        # RELIABLE+KeepLast(1); a reliable reader loses the fragment-reassembly
        # race against that depth-1 writer and freezes on the first frame, so we
        # take each burst best-effort instead.
        be_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        )
        # Outputs RELIABLE+KeepLast(5): serves a RELIABLE subscriber (RViz) AND a
        # BEST_EFFORT one (the planner); depth>1 avoids the same reassembly race.
        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
        )
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST, depth=1,
        )

        self.obstacle_pub = self.create_publisher(PointCloud2, self.obstacles_topic, pub_qos)
        self.voxel_pub = self.create_publisher(PointCloud2, self.voxel_topic, pub_qos)
        self.costmap_pub = (
            self.create_publisher(OccupancyGrid, self.costmap_topic, latched_qos)
            if self.publish_costmap else None
        )
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, be_qos)
        self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, be_qos)

        self.get_logger().info(
            f"local_voxel_map up | cloud={self.cloud_topic} odom={self.odom_topic} | "
            f"window=±{self.half_width:.1f}m voxel={self.voxel_size:.2f}m "
            f"ground(local-min cell={self.ground_params.cell:.2f}m band={self.ground_params.ground_band:.2f}m "
            f"min_pts={self.ground_params.min_pts}) persistence={self.persistence_s:.1f}s"
        )

    def _on_odom(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        self.robot_xyz = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        self.have_odom = True

    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.have_odom:
            return  # need the robot position to centre the rolling window
        self.frame_id = msg.header.frame_id or "odom"
        now_s = self.get_clock().now().nanoseconds * 1e-9

        xyz = read_xyz(msg)
        cx, cy, cz = self.robot_xyz

        # 1. rolling-window crop (XY box + vertical band relative to robot/sensor).
        if xyz.shape[0]:
            dx = xyz[:, 0] - cx
            dy = xyz[:, 1] - cy
            in_box = (np.abs(dx) <= self.half_width) & (np.abs(dy) <= self.half_width)
            in_z = (xyz[:, 2] >= cz - self.z_below) & (xyz[:, 2] <= cz + self.z_above)
            in_range = (dx * dx + dy * dy) >= (self.min_range * self.min_range)
            cropped = xyz[in_box & in_z & in_range]
        else:
            cropped = xyz

        # 2. accumulate RAW points (floor included) + decay.
        if cropped.shape[0]:
            self.accum.update(cropped, now_s)
        self.accum.prune(now_s, (cx, cy), self.half_width)
        raw_centers = self.accum.centers()

        # 3. gravity-aware SVD ground removal on the dense accumulated cloud.
        if raw_centers.shape[0]:
            obstacles, seg_info = segment_ground(
                raw_centers, self.g_hat, cz, self.ground_params, return_info=True)
        else:
            obstacles, seg_info = raw_centers, {"status": "empty", "manifold_found": False,
                                                "ground_cells": 0, "candidate_cells": 0}

        # Throttled warning when no ground manifold is found (e.g. robot on a table,
        # bad init gravity): we then keep everything rather than blank the cloud.
        if raw_centers.shape[0] and not seg_info["manifold_found"]:
            self.get_logger().warning(
                f"ground segmentation found no manifold ({seg_info['status']}, "
                f"candidates={seg_info['candidate_cells']}) — passing cloud through",
                throttle_duration_sec=5.0)

        # Pipeline heartbeat (~1 Hz) — shows where data is lost if the map looks empty.
        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick <= 5 or self._tick % 10 == 0:
            self.get_logger().info(
                f"in={xyz.shape[0]} cropped={cropped.shape[0]} "
                f"accum_vox={raw_centers.shape[0]} obstacles={obstacles.shape[0]} "
                f"cand={seg_info['candidate_cells']} seed={seg_info['seed_cells']} "
                f"ground_cells={seg_info['ground_cells']} | "
                f"robot=({cx:.2f},{cy:.2f},{cz:.2f}) zband=[{cz - self.z_below:.2f},{cz + self.z_above:.2f}]")

        # 4. publish every scan (even when empty) so the costmap stays live.
        header = Header(stamp=msg.header.stamp, frame_id=self.frame_id)
        cloud = pc2.create_cloud_xyz32(header, obstacles)
        self.obstacle_pub.publish(cloud)
        self.voxel_pub.publish(cloud)
        if self.costmap_pub is not None:
            self.costmap_pub.publish(self._build_costmap(obstacles, header, (cx, cy)))

    def _build_costmap(self, centers: np.ndarray, header: Header,
                       center_xy: tuple[float, float]) -> OccupancyGrid:
        """Project occupied voxels down to a 2D OccupancyGrid (column = occupied)."""
        reso = self.voxel_size
        n = int(round(2.0 * self.half_width / reso))
        cx, cy = center_xy
        origin_x = (np.floor(cx / reso) * reso) - self.half_width
        origin_y = (np.floor(cy / reso) * reso) - self.half_width

        grid = np.full((n, n), self.costmap_unknown_as, dtype=np.int8)
        if centers.shape[0]:
            ix = np.floor((centers[:, 0] - origin_x) / reso).astype(np.int64)
            iy = np.floor((centers[:, 1] - origin_y) / reso).astype(np.int64)
            ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
            grid[iy[ok], ix[ok]] = 100  # row-major: row=y, col=x

        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = reso
        msg.info.width = n
        msg.info.height = n
        msg.info.origin.position = Point(x=float(origin_x), y=float(origin_y), z=0.0)
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = LocalVoxelMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
