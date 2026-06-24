"""
A* planner ROS2 node for the Unitree G1 humanoid (Navigation / DLIO stack).

Ported from the G1_navigation planner and re-adapted to the Navigation
deployment layer, whose perception front-end is DLIO + g1_local_map:
  - pose comes from DLIO odometry (nav_msgs/Odometry on /dlio/odom_node/odom),
    not a PoseStamped on /g1/pose. It is converted to a PoseStamped internally
    so the rest of the planner is unchanged. Frame is `odom`.
  - obstacles come from g1_local_map's ground-removed, rolling, robot-centred
    obstacle cloud (/local_voxel_map/obstacles, odom frame, ~10 Hz). Because
    that cloud is ALREADY ground-removed, the planner's own per-cell ground
    segmentation is left OFF by default (config: ground_segment_en=false) — the
    incoming points are treated as obstacles directly, and the persistent map
    is fed the raw cloud (no re-segmentation that could drop low obstacles).
  - the DLIO odom origin sits at the SENSOR (~1 m above the floor), so the floor
    lands near z≈-1 m. The voxel z-band is widened accordingly; the gmap cost
    layer is per-cell relative so it is unaffected by the z offset.
  - the underlying grid map is 2.5D: each cell stores max-z of hits so short
    obstacles (< step-over threshold) can be traversed.

Static map fusion
-----------------
The planner can still fuse a static global obstacle layer into its local grid:
  - enable_slam_map:=true → a generic 2D OccupancyGrid (/map). Backend-agnostic
    helper (e.g. a Nav2 map server); not auto-enabled, defaults off.
  - enable_dlio_map:=true → DLIO's global 3D map (/dlio/map_node/map) fused as
    static context. Defaults OFF (g1_local_map already gives a dense
    ground-removed cloud); the fusion path strips the map's floor first, since
    the DLIO global map is the raw accumulated cloud and still contains ground.

Data flow:
  /local_voxel_map/obstacles ──┐
  persistent_map             ──┼──► FixedGaussianGridMap ──► A* ──► /a_star/path
  (optional 2D /map cells)   ──┘

Architecture
------------
  Subscribes:
    /dlio/odom_node/odom       nav_msgs/Odometry — robot pose (converted to
                                                  PoseStamped; frame `odom`)
    /local_voxel_map/obstacles PointCloud2       — ground-removed obstacle cloud
    /global_goal               PoseStamped       — runtime global goal override
    /map                       OccupancyGrid     — generic 2D static map
                                                  (only when enable_slam_map=true)

  Publishes:
    /a_star/path              nav_msgs/Path             — local A* path
    /a_star/local_goal        geometry_msgs/PoseStamped — current local goal
    /a_star/occupancy_grid    nav_msgs/OccupancyGrid    — Gaussian grid map
    /a_star/grid_raw          std_msgs/Float32MultiArray — raw grid + meta
    /a_star/height_map        std_msgs/Float32MultiArray — 2.5D max-z per cell
"""

import math

import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, Header

from a_star_mpc_planner.a_star_planner import AStarPlanner
from a_star_mpc_planner.external_grid_map import parse_costmap_raw
from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap
from a_star_mpc_planner.persistent_map import PersistentOccupancyMap
from a_star_mpc_planner.slam_map_utils import extract_slam_obstacle_points


class AStarNode(Node):

    def __init__(self):
        super().__init__('a_star_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('goal_x',                5.0)
        self.declare_parameter('goal_y',                5.0)
        self.declare_parameter('goal_z',                0.0)
        self.declare_parameter('wait_for_goal',       False)
        self.declare_parameter('grid_reso',             0.25)
        self.declare_parameter('grid_half_width',       5.0)
        self.declare_parameter('grid_std',              0.4)
        self.declare_parameter('obstacle_threshold',    0.5)
        self.declare_parameter('obstacle_cost_weight', 10.0)
        self.declare_parameter('replan_rate_hz',        2.0)
        self.declare_parameter('goal_reached_radius',   0.3)
        self.declare_parameter('duplicate_goal_xy_tolerance', 0.05)
        self.declare_parameter('duplicate_goal_z_tolerance', 0.05)
        self.declare_parameter('max_lidar_range',       6.0)
        # Pose source. The Navigation stack localizes with DLIO, which publishes
        # nav_msgs/Odometry on /dlio/odom_node/odom (frame `odom`). It is
        # converted to a PoseStamped internally so the rest of the planner is
        # unchanged. Override for sim or a different localization backend.
        self.declare_parameter('odom_topic', '/dlio/odom_node/odom')
        # Source of obstacle points. Default is g1_local_map's ground-removed,
        # rolling, robot-centred obstacle cloud (odom frame, ~10 Hz). It is a
        # PointCloud2 with x/y/z fields in the odom frame, so the rest of the
        # planner is agnostic to the source. Because this cloud is already
        # ground-removed, ground_segment_en is left off in the config.
        self.declare_parameter('obstacle_topic', '/local_voxel_map/obstacles')
        self.declare_parameter('planning_height',       0.0)
        self.declare_parameter('map_decay_sec',        30.0)
        self.declare_parameter('map_max_cells',     50_000)
        # Persistent-map obstacle band + recall height. The persistent map
        # extends obstacle memory beyond g1_local_map's short rolling window by
        # accumulating obstacle cells over map_decay_sec. Only points in
        # [persistent_z_min, persistent_z_max] are accumulated so the floor is not
        # remembered as an obstacle, and recalled cells are returned at
        # persistent_obstacle_height (> step_over_height) so they actually block
        # A*. Set persistent_z_min > persistent_z_max to disable the band.
        self.declare_parameter('persistent_z_min',          0.15)
        self.declare_parameter('persistent_z_max',          2.0)
        self.declare_parameter('persistent_obstacle_height', 0.30)
        self.declare_parameter('ground_segment_height', 0.3)
        self.declare_parameter('ground_segment_en',     True)
        self.declare_parameter('ground_removal_en',     False)
        self.declare_parameter('ground_plane_margin',   0.18)   # m above/below plane = ground
        self.declare_parameter('ground_candidate_z_max', 0.5)   # only fit plane among low points
        self.declare_parameter('ground_min_normal_z',   0.85)   # plane must be ~horizontal
        self.declare_parameter('ground_ransac_iters',   40)
        # 2.5D step-over rule
        self.declare_parameter('step_over_height',      0.08)
        self.declare_parameter('step_over_cost_scale',  0.25)
        # 3D voxel grid Z extent
        self.declare_parameter('voxel_z_min',           0.4)
        self.declare_parameter('voxel_z_max',           2.5)
        # ── Generic 2D OccupancyGrid fusion ───────────────────────────
        # Set enable_slam_map:=true to fuse occupied cells from a 2D
        # OccupancyGrid on /map (any backend-agnostic source, e.g. a Nav2
        # map server) into the local planning grid each replan cycle. Not
        # auto-enabled by any mapping_source; defaults off.
        self.declare_parameter('enable_slam_map',              False)
        self.declare_parameter('slam_map_topic',               '/map')
        self.declare_parameter('slam_map_occupied_threshold',  50)
        self.declare_parameter('slam_map_unknown_is_obstacle', False)
        self.declare_parameter('slam_map_max_age_sec',         5.0)
        # ── DLIO 3D map fusion ─────────────────────────────────────────
        # Set enable_dlio_map:=true to fuse DLIO's global 3D map
        # (/dlio/map_node/map) as static obstacle context (walls, furniture)
        # alongside the live g1_local_map cloud. The planner crops it to the
        # local planning window and adds the result as direct obstacles, so
        # structure already mapped by DLIO blocks A* even when it has left the
        # live sensor's FOV. Off by default: g1_local_map already provides a
        # dense, ground-removed, robot-centred obstacle cloud.
        #
        # Unlike /local_voxel_map/obstacles, the DLIO global map is NOT
        # ground-removed and updates only on a new keyframe (~1 m / 45° of
        # motion), so the fusion path below strips its floor with the same
        # per-cell segmentation the live grid uses (see _extract_dlio_*).
        self.declare_parameter('enable_dlio_map',     False)
        self.declare_parameter('dlio_map_topic',      '/dlio/map_node/map')
        self.declare_parameter('dlio_map_max_age_sec', 30.0)
        # Ceiling/overhang cap for fused DLIO-map points, expressed RELATIVE to
        # the robot's current odom z (the sensor height). The planner is 2D, so
        # any point — including ceilings and overhangs — projects to an XY
        # obstacle; drop everything more than this above the sensor so a ceiling
        # is not fused as a phantom wall. The floor below is removed by per-cell
        # segmentation, which is robust to the sensor-origin frame.
        self.declare_parameter('dlio_map_z_above', 1.0)
        # ── Costmap backend selector ──────────────────────────────────
        # 'gaussian'      : default. Build the local cost grid from the live
        #                   cloud + persistent memory + fused maps (above).
        # 'external_grid' : plan directly on a pre-computed cost grid published
        #                   by another node (the nvblox ESDF adapter on
        #                   external_costmap_topic, grid_raw layout). In this
        #                   mode the Gaussian live-cloud / persistent-map /
        #                   DLIO-map fusion paths are all DISABLED so the
        #                   external costmap is the single source of obstacles
        #                   (no double inflation). MPC still gets discrete
        #                   obstacles from the adapter's /nvblox_g1/obstacle_points.
        self.declare_parameter('costmap_backend',         'gaussian')
        self.declare_parameter('external_costmap_topic',  '/nvblox_g1/costmap_raw')
        self.declare_parameter('external_costmap_max_age_sec', 2.0)

        self._goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value,
        ])
        self._wait_for_goal = bool(self.get_parameter('wait_for_goal').value)
        self._goal_initialized = not self._wait_for_goal

        self._planning_height = float(self.get_parameter('planning_height').value)
        self._goal_reached_radius = float(self.get_parameter('goal_reached_radius').value)
        self._duplicate_goal_xy_tolerance = float(
            self.get_parameter('duplicate_goal_xy_tolerance').value
        )
        self._duplicate_goal_z_tolerance = float(
            self.get_parameter('duplicate_goal_z_tolerance').value
        )
        self._max_lidar_range = float(self.get_parameter('max_lidar_range').value)
        self._obstacle_topic = str(self.get_parameter('obstacle_topic').value)

        # Costmap backend: 'gaussian' (build grid from cloud) or 'external_grid'
        # (plan on a pre-computed cost grid, e.g. the nvblox ESDF adapter).
        self._costmap_backend = str(self.get_parameter('costmap_backend').value).strip()
        if self._costmap_backend not in ('gaussian', 'external_grid'):
            self.get_logger().warning(
                f"unknown costmap_backend={self._costmap_backend!r}; "
                "falling back to 'gaussian'"
            )
            self._costmap_backend = 'gaussian'
        self._external_costmap_max_age = float(
            self.get_parameter('external_costmap_max_age_sec').value
        )
        # External cost grid state (external_grid backend only).
        self._external_grid = None          # ExternalGridMap | None
        self._external_grid_t: float = 0.0

        # SLAM map fusion config
        self._enable_slam_map = bool(self.get_parameter('enable_slam_map').value)
        self._slam_map_occ_thr = int(self.get_parameter('slam_map_occupied_threshold').value)
        self._slam_map_unknown_obstacle = bool(
            self.get_parameter('slam_map_unknown_is_obstacle').value
        )
        self._slam_map_max_age = float(self.get_parameter('slam_map_max_age_sec').value)

        # ── Algorithm objects ─────────────────────────────────────────
        self._grid_map = FixedGaussianGridMap(
            reso=float(self.get_parameter('grid_reso').value),
            half_width=float(self.get_parameter('grid_half_width').value),
            std=float(self.get_parameter('grid_std').value),
            z_min=float(self.get_parameter('voxel_z_min').value),
            z_max=float(self.get_parameter('voxel_z_max').value),
            ground_segment_height=float(self.get_parameter('ground_segment_height').value),
            ground_segment_en=bool(self.get_parameter('ground_segment_en').value),
        )
        self._planner = AStarPlanner(
            obstacle_threshold=float(self.get_parameter('obstacle_threshold').value),
            obstacle_cost_weight=float(self.get_parameter('obstacle_cost_weight').value),
            step_over_height=float(self.get_parameter('step_over_height').value),
            step_over_cost_scale=float(self.get_parameter('step_over_cost_scale').value),
        )
        self._persistent_z_min = float(self.get_parameter('persistent_z_min').value)
        self._persistent_z_max = float(self.get_parameter('persistent_z_max').value)
        self._ground_removal_en = bool(self.get_parameter('ground_removal_en').value)
        self._ground_margin = float(self.get_parameter('ground_plane_margin').value)
        self._ground_cand_z_max = float(self.get_parameter('ground_candidate_z_max').value)
        self._ground_min_nz = float(self.get_parameter('ground_min_normal_z').value)
        self._ground_iters = int(self.get_parameter('ground_ransac_iters').value)
        self._ground_rng = np.random.default_rng(0)
        self._persistent_map = PersistentOccupancyMap(
            grid_reso=float(self.get_parameter('grid_reso').value),
            decay_sec=float(self.get_parameter('map_decay_sec').value),
            max_cells=int(self.get_parameter('map_max_cells').value),
            obstacle_height=float(self.get_parameter('persistent_obstacle_height').value),
        )

        # ── State ─────────────────────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._lidar_points: np.ndarray | None = None
        self._goal_reached = False
        # Frame used for all published topics (path, grids, etc.). Set from the
        # first DLIO odometry header (frame `odom`), so the planner outputs match
        # the obstacle cloud and MPC. Default protects against a race where
        # publishing starts before the first odom message arrives.
        self._pose_frame: str = 'odom'

        # SLAM map state (slam_toolbox OccupancyGrid)
        self._slam_map: OccupancyGrid | None = None
        self._slam_map_t: float = 0.0

        # DLIO 3D map state (global /dlio/map_node/map, odom frame, keyframe-rate)
        self._enable_dlio_map = bool(self.get_parameter('enable_dlio_map').value)
        self._dlio_map_max_age = float(self.get_parameter('dlio_map_max_age_sec').value)
        self._dlio_map_z_above = float(self.get_parameter('dlio_map_z_above').value)
        self._dlio_map_pts: np.ndarray | None = None   # (N, 3) xyz in odom frame
        self._dlio_map_t: float = 0.0

        # ── TF2 (used for SLAM map frame → planner frame transform) ───
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── QoS ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────
        # Pose: DLIO odometry. Use BEST_EFFORT so a reliable DLIO publisher is
        # still readable (a best-effort reader can read a reliable writer, not
        # vice-versa) and the stack stays robust to the QoS DLIO ships with.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, odom_qos)
        self.get_logger().info(f'pose source: {odom_topic} (Odometry → PoseStamped)')
        self.create_subscription(PoseStamped, '/global_goal',          self._goal_cb,         10)

        external = self._costmap_backend == 'external_grid'
        if external:
            # Plan on a pre-computed cost grid; the Gaussian live-cloud,
            # persistent-map and DLIO-map fusion paths are all OFF (the
            # external costmap is the single obstacle source — no double
            # inflation). MPC reads discrete obstacles from the adapter.
            ext_topic = str(self.get_parameter('external_costmap_topic').value)
            self.create_subscription(
                Float32MultiArray, ext_topic, self._external_costmap_cb, 10)
            self._enable_slam_map = False
            self._enable_dlio_map = False
            self.get_logger().info(
                f"costmap backend: external_grid  topic={ext_topic}  "
                f"max_age={self._external_costmap_max_age}s "
                "(Gaussian live-cloud/persistent/DLIO-map fusion disabled)"
            )
        else:
            self.create_subscription(
                PointCloud2, self._obstacle_topic, self._lidar_cb, sensor_qos)
            self.get_logger().info(
                f"costmap backend: gaussian  obstacle source: {self._obstacle_topic}")

        if self._enable_slam_map:
            slam_map_topic = str(self.get_parameter('slam_map_topic').value)
            # TRANSIENT_LOCAL so a late-joining subscriber gets the last map
            # published by slam_toolbox even before the first new scan arrives.
            slam_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(OccupancyGrid, slam_map_topic, self._slam_map_cb, slam_qos)
            self.get_logger().info(
                f'SLAM map fusion: enabled  topic={slam_map_topic}  '
                f'occ_threshold={self._slam_map_occ_thr}  '
                f'unknown_is_obstacle={self._slam_map_unknown_obstacle}  '
                f'max_age={self._slam_map_max_age}s'
            )
        else:
            self.get_logger().info('SLAM map fusion: disabled (enable_slam_map=false)')

        if self._enable_dlio_map:
            dlio_map_topic = str(self.get_parameter('dlio_map_topic').value)
            # Use BEST_EFFORT for the dense global map cloud; it can be large
            # and we prefer throughput over guaranteed delivery.
            dlio_map_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(
                PointCloud2, dlio_map_topic, self._dlio_map_cb, dlio_map_qos
            )
            self.get_logger().info(
                f'DLIO 3D map fusion: enabled  topic={dlio_map_topic}  '
                f'max_age={self._dlio_map_max_age}s'
            )
        else:
            self.get_logger().info('DLIO 3D map fusion: disabled (enable_dlio_map=false)')

        # ── Publishers ────────────────────────────────────────────────
        self._path_pub   = self.create_publisher(Path,               '/a_star/path',                10)
        self._lgpal_pub  = self.create_publisher(PoseStamped,        '/a_star/local_goal',          10)
        self._grid_pub   = self.create_publisher(OccupancyGrid,      '/a_star/occupancy_grid',      10)
        self._raw_pub    = self.create_publisher(Float32MultiArray,  '/a_star/grid_raw',            10)
        self._pmap_pub   = self.create_publisher(PointCloud2,        '/a_star/persistent_obstacles', 10)
        # 2.5D max-z layer for downstream consumers (find_safe_stop_pose, etc.)
        self._height_pub = self.create_publisher(Float32MultiArray,  '/a_star/height_map',          10)
        # 3D voxel occupancy as a point cloud (one point per occupied voxel
        # centre). Lets RViz / Foxglove render the full 3D obstacle layer
        # cheaply, and exposes voxelised occupancy to host-side consumers
        # without shipping a 160k-element dense array.
        self._voxel_pub  = self.create_publisher(PointCloud2,        '/a_star/voxel_grid',          10)

        # ── Replan timer ──────────────────────────────────────────────
        rate = float(self.get_parameter('replan_rate_hz').value)
        self.create_timer(1.0 / rate, self._replan_cb)

        if self._goal_initialized:
            self.get_logger().info(
                f'A* node ready | startup goal=({self._goal[0]:.1f}, {self._goal[1]:.1f}) '
                f'| grid={2*self._grid_map.half_width:.0f}m @ {self._grid_map.reso}m/cell'
            )
        else:
            self.get_logger().info(
                f'A* node ready | waiting for /global_goal '
                f'| grid={2*self._grid_map.half_width:.0f}m @ {self._grid_map.reso}m/cell'
            )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        """Convert DLIO odometry into the PoseStamped the planner expects.

        DLIO publishes nav_msgs/Odometry; the planner only needs the pose, so we
        repackage header + pose.pose into a PoseStamped and reuse _pose_cb. The
        header frame (`odom`) drives the frame_id of every published topic.
        """
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self._pose_cb(ps)

    def _pose_cb(self, msg: PoseStamped):
        self._pose = msg
        self._pose_frame = msg.header.frame_id or 'odom'

    def _goal_cb(self, msg: PoseStamped):
        new_goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        if self._goal_initialized:
            xy_delta = float(np.linalg.norm(new_goal[:2] - self._goal[:2]))
            z_delta = abs(float(new_goal[2] - self._goal[2]))
            if (
                xy_delta <= self._duplicate_goal_xy_tolerance
                and z_delta <= self._duplicate_goal_z_tolerance
            ):
                self.get_logger().debug(
                    f'[A*] Ignoring duplicate global goal: '
                    f'xy_delta={xy_delta:.3f} m z_delta={z_delta:.3f} m'
                )
                return

        self._goal = new_goal
        self._goal_reached = False
        self._goal_initialized = True

        self.get_logger().info(
            f'[A*] New global goal: ({self._goal[0]:.2f}, {self._goal[1]:.2f}, {self._goal[2]:.2f})'
        )

    def _remove_ground_plane(self, pts: np.ndarray) -> np.ndarray:
        """Remove the dominant near-horizontal plane (the floor) via RANSAC.

        A flat z-cut cannot separate floor from obstacles when the floor return
        is tilted/noisy (and drifts in z). Fitting the plane and
        dropping points within ``ground_plane_margin`` of it is robust to that
        tilt/offset. Points well above the plane (real obstacles) are kept.
        Returns the cloud unchanged if no good ground plane is found.
        """
        n_pts = len(pts)
        if n_pts < 50:
            return pts
        # Fit the plane only among low points so a tall wall can't dominate it.
        z = pts[:, 2]
        cand = pts[z < (z.min() + self._ground_cand_z_max)]
        if len(cand) < 30:
            return pts

        best_inliers = -1
        best_plane = None
        m = len(cand)
        for _ in range(self._ground_iters):
            tri = cand[self._ground_rng.choice(m, 3, replace=False)]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = np.linalg.norm(normal)
            if nn < 1e-9:
                continue
            normal = normal / nn
            if abs(normal[2]) < self._ground_min_nz:   # must be ~horizontal
                continue
            d = -float(normal.dot(tri[0]))
            inliers = int(np.count_nonzero(
                np.abs(cand @ normal + d) < self._ground_margin))
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal, d)

        if best_plane is None:
            return pts
        normal, d = best_plane
        ground = np.abs(pts @ normal + d) < self._ground_margin
        kept = pts[~ground]
        return kept if len(kept) > 0 else pts

    def _lidar_cb(self, msg: PointCloud2):
        """Parse the scan; PRESERVE the last valid cloud on transient empties.

        Failure modes distinguished in the log:
          - "no_points"     : message contained zero points (rotary sweep idle)
          - "out_of_range"  : every point was beyond max_lidar_range from robot
          - "no_pose"       : we can't range-filter yet because pose hasn't arrived
        In any of those cases we keep the previous self._lidar_points so a
        single bad cycle doesn't tank planning. The persistent map still
        eventually decays stale cells.
        """
        try:
            points = list(point_cloud2.read_points(msg, skip_nans=True))
        except Exception as e:
            self.get_logger().warn(f'Lidar parsing error: {e}')
            return

        # type: np.ndarray | None
        new_pts = None
        if points:
            arr = np.array([(p[0], p[1], p[2]) for p in points], dtype=float)
            if self._pose is not None:
                px = self._pose.pose.position.x
                py = self._pose.pose.position.y
                dists = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                arr = arr[dists < self._max_lidar_range]
            # Drop the (tilted/noisy) ground plane so it is not treated as an
            # obstacle by either the live grid or the persistent map.
            if self._ground_removal_en and len(arr) > 0:
                arr = self._remove_ground_plane(arr)
            if len(arr) > 0:
                new_pts = arr
            else:
                # Either all points were beyond max_lidar_range, or pose
                # filter rejected them all. Either way the scan IS live,
                # it just doesn't contribute new local obstacles.
                self.get_logger().info(
                    '[A*-LIDAR] all points filtered (range or pose)',
                    throttle_duration_sec=3.0,
                )
        else:
            self.get_logger().info(
                '[A*-LIDAR] empty scan received — keeping last valid cloud',
                throttle_duration_sec=3.0,
            )

        # Update the persistent map ONLY when we received real points;
        # writing an empty/None scan would still trigger eviction, which
        # is fine, but we don't want it to silently delete useful state
        # on every empty rotary tick.
        #
        # Feed the persistent map only SEGMENTED obstacles (same per-cell height
        # segmentation the grid uses) so the floor is never remembered as a
        # blocker — even a tilted floor, which a flat z-band could not separate.
        # The full cloud is still kept in self._lidar_points; the grid segments
        # it live each cycle.
        now = self.get_clock().now().nanoseconds * 1e-9
        if new_pts is not None:
            # Feed the persistent map only SEGMENTED obstacles so the floor is
            # never remembered as a blocker. When the obstacle source is already
            # ground-removed (g1_local_map's /local_voxel_map/obstacles,
            # ground_segment_en=false), re-segmenting per-cell would treat a low
            # obstacle's own lowest point as ground and drop it — so feed the raw
            # cloud directly in that case.
            if self._grid_map.ground_segment_en:
                obstacle_pts = FixedGaussianGridMap.segment_obstacles(
                    new_pts,
                    float(self.get_parameter('grid_reso').value),
                    float(self.get_parameter('ground_segment_height').value),
                )
            else:
                obstacle_pts = new_pts
            self._persistent_map.update(
                obstacle_pts if obstacle_pts is not None and len(obstacle_pts) > 0
                else None, now)
            self._lidar_points = new_pts
        else:
            # Just decay the persistent map; don't clobber self._lidar_points.
            self._persistent_map.update(None, now)

    def _slam_map_cb(self, msg: OccupancyGrid) -> None:
        """Cache the latest SLAM map from slam_toolbox."""
        self._slam_map = msg
        self._slam_map_t = self.get_clock().now().nanoseconds * 1e-9

    def _dlio_map_cb(self, msg: PointCloud2) -> None:
        """Cache DLIO's global 3D map (/dlio/map_node/map) as an (N,3) array."""
        try:
            pts = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
        except Exception as exc:
            self.get_logger().warning(f'[A*-DLIO] map parse error: {exc}', throttle_duration_sec=5.0)
            return
        if pts:
            self._dlio_map_pts = np.array([(p[0], p[1], p[2]) for p in pts], dtype=np.float32)
            self._dlio_map_t   = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().debug(
                f'[A*-DLIO] map updated: {len(self._dlio_map_pts)} pts',
            )

    def _external_costmap_cb(self, msg: Float32MultiArray) -> None:
        """Decode a pre-computed cost grid (external_grid backend).

        Parsing failures (truncated / malformed payload) keep the last valid
        grid so a single bad message does not stall planning; the staleness
        guard in _replan_external eventually trips if they keep failing.
        """
        try:
            grid = parse_costmap_raw(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f'[A*-EXT] costmap parse error: {exc} — keeping last grid',
                throttle_duration_sec=5.0,
            )
            return
        self._external_grid = grid
        self._external_grid_t = self.get_clock().now().nanoseconds * 1e-9

    # ── SLAM map fusion ───────────────────────────────────────────────

    def _extract_slam_obstacle_points(self, robot_xy: np.ndarray) -> np.ndarray | None:
        """Resolve the map→odom TF then delegate to slam_map_utils."""
        if not self._enable_slam_map or self._slam_map is None:
            return None

        map_frame = self._slam_map.header.frame_id or 'map'

        # Look up map → odom transform; fall back to identity (map == odom)
        # when slam_toolbox is not running or TF is not yet available.
        dx, dy, yaw = 0.0, 0.0, 0.0
        try:
            tf = self._tf_buffer.lookup_transform(
                self._pose_frame,
                map_frame,
                rclpy.time.Time(),
                Duration(seconds=0.1),
            )
            dx = tf.transform.translation.x
            dy = tf.transform.translation.y
            qx = tf.transform.rotation.x
            qy = tf.transform.rotation.y
            qz = tf.transform.rotation.z
            qw = tf.transform.rotation.w
            yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
        except tf2_ros.LookupException:
            pass  # map == odom; identity is correct
        except Exception as exc:
            self.get_logger().warning(
                f'[A*-SLAM] TF lookup failed: {exc!r} — assuming map==odom',
                throttle_duration_sec=5.0,
            )

        now = self.get_clock().now().nanoseconds * 1e-9
        return extract_slam_obstacle_points(
            slam_map=self._slam_map,
            slam_map_t=self._slam_map_t,
            now=now,
            slam_map_max_age=self._slam_map_max_age,
            slam_map_occ_thr=self._slam_map_occ_thr,
            slam_map_unknown_obstacle=self._slam_map_unknown_obstacle,
            robot_xy=robot_xy,
            grid_half_width=self._grid_map.half_width,
            planning_height=self._planning_height,
            pose_frame=self._pose_frame,
            tf_dx=dx,
            tf_dy=dy,
            tf_yaw=yaw,
            logger=self.get_logger(),
        )

    # ── DLIO 3D map fusion ──────────────────────────────────────────────

    def _extract_dlio_obstacle_points(self, robot_xy: np.ndarray) -> np.ndarray | None:
        """Crop DLIO's global map to the window and strip its floor.

        Returns an (M, 3) array of obstacle xyz points within grid_half_width of
        the robot, or None if no map is available / the map is stale / nothing
        survives. These are fused as DIRECT obstacles, so the floor MUST be
        removed first — unlike /local_voxel_map/obstacles, DLIO's global map
        (/dlio/map_node/map) is the raw accumulated cloud and still contains the
        ground.

        Floor removal uses the same gravity-agnostic, per-cell segmentation the
        live grid uses (FixedGaussianGridMap.segment_obstacles): each XY cell's
        lowest point is its local ground, points rising above it by
        ground_segment_height are obstacles. Relative-per-cell, so it is robust
        to the DLIO sensor-origin frame (floor at z≈-1 m) and to map tilt/drift —
        a fixed z-band would either keep the floor or clip low obstacles here.
        A ceiling/overhang cap (dlio_map_z_above, relative to the sensor z)
        prevents ceilings from projecting into the 2D planner as phantom walls.
        """
        if not self._enable_dlio_map or self._dlio_map_pts is None:
            return None

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._dlio_map_t > self._dlio_map_max_age:
            self.get_logger().info(
                '[A*-DLIO] map stale — skipping fusion',
                throttle_duration_sec=5.0,
            )
            return None

        pts  = self._dlio_map_pts
        hw   = self._grid_map.half_width
        # Sensor z in odom (floor sits ~1 m below this); used for the ceiling cap.
        robot_z = float(self._pose.pose.position.z) if self._pose is not None else 0.0
        mask = (
            (np.abs(pts[:, 0] - robot_xy[0]) < hw) &
            (np.abs(pts[:, 1] - robot_xy[1]) < hw) &
            (pts[:, 2] <= robot_z + self._dlio_map_z_above)
        )
        cropped = pts[mask]
        if len(cropped) == 0:
            return None

        # Drop the floor (relative-per-cell), keeping only real vertical structure.
        obstacles = FixedGaussianGridMap.segment_obstacles(
            cropped,
            float(self.get_parameter('grid_reso').value),
            float(self.get_parameter('ground_segment_height').value),
        )
        return obstacles if obstacles is not None and len(obstacles) > 0 else None

    # ── Replan ────────────────────────────────────────────────────────

    def _replan_cb(self):
        """
        Replan every timer tick (now at 10 Hz for continuous online adaptation).

        KEY: This callback runs at the configured replan_rate_hz (default 10 Hz).
        At EACH call:
          1. We use the LATEST robot pose from self._pose
          2. We use the LATEST LiDAR scan from self._lidar_points
          3. Local occupancy grid is RE-CENTERED on robot's CURRENT position
          4. A* is RE-RUN from robot's CURRENT position to goal
          5. New path is PUBLISHED for MPC to track

        This ensures that as the robot moves, the planning continuously adapts
        to new sensor data and pose, rather than following a stale plan.
        """
        if self._costmap_backend == 'external_grid':
            self._replan_external()
            return

        # DEBUG: show what data is (or isn't) available.
        # Block ONLY if pose is missing OR (live lidar is missing AND we have
        # no persistent obstacle memory either). With the persistent_map fallback
        # we can keep replanning across transient empty scans — important on a
        # rotary lidar whose sweep occasionally returns zero hits within range.
        has_live = self._lidar_points is not None
        has_persistent = self._persistent_map.size > 0
        if self._pose is None or (not has_live and not has_persistent):
            self.get_logger().warn(
                f'[A*-DEBUG] Blocked: pose={self._pose is not None}  '
                f'lidar={has_live}  persistent_cells={self._persistent_map.size}',
                throttle_duration_sec=3.0,
            )
            return

        if not self._goal_initialized:
            self.get_logger().info(
                '[A*] Waiting for first goal on /global_goal ...',
                throttle_duration_sec=5.0,
            )
            return

        drone_xyz = np.array([
            self._pose.pose.position.x,
            self._pose.pose.position.y,
            self._pose.pose.position.z,
        ])
        drone_xy = drone_xyz[:2]

        # DEBUG: log the actual pose being used for planning
        self.get_logger().info(
            f'[A*-DEBUG] Planning from pose=({drone_xy[0]:.4f}, {drone_xy[1]:.4f})  '
            f'goal=({self._goal[0]:.2f}, {self._goal[1]:.2f})',
            throttle_duration_sec=2.0,
        )

        # Goal-reached guard — stop replanning once close enough
        dist_to_goal = float(np.linalg.norm(drone_xy - self._goal[:2]))
        if dist_to_goal <= self._goal_reached_radius:
            if not self._goal_reached:
                self.get_logger().info(
                    f'[A*] Goal reached! dist={dist_to_goal:.2f} m'
                )
                self._goal_reached = True
            return
        self._goal_reached = False

        # === ONLINE REPLANNING: Update grid centered on CURRENT robot position ===
        # The live LiDAR cloud is passed WHOLE and ground/obstacle segmented
        # inside the grid map (per-cell height). Obstacles that are already
        # confirmed — persistent memory + fused SLAM/DLIO maps — are passed
        # as DIRECT obstacles so they bypass segmentation and always block,
        # representing walls that have left the live sensor FOV.
        hw = self._grid_map.half_width
        direct_list = []
        persistent_pts = self._persistent_map.get_points_in_window(
            drone_xy[0] - hw, drone_xy[1] - hw,
            drone_xy[0] + hw, drone_xy[1] + hw,
        )
        if persistent_pts is not None:
            direct_list.append(persistent_pts)

        # Fuse SLAM map occupied cells (slam_toolbox 2D OccupancyGrid).
        slam_pts = self._extract_slam_obstacle_points(drone_xy)
        if slam_pts is not None:
            direct_list.append(slam_pts)
            self.get_logger().debug(
                f'[A*-SLAM] fused {len(slam_pts)} cells from SLAM map',
                throttle_duration_sec=2.0,
            )

        # Fuse DLIO's global 3D map (cropped to local window, floor stripped) —
        # static context from structures no longer in the live sensor's FOV.
        dlio_pts = self._extract_dlio_obstacle_points(drone_xy)
        if dlio_pts is not None:
            direct_list.append(dlio_pts)
            self.get_logger().debug(
                f'[A*-DLIO] fused {len(dlio_pts)} pts from DLIO global map',
                throttle_duration_sec=2.0,
            )

        direct_obstacles = np.vstack(direct_list) if direct_list else None
        self._grid_map.update(
            self._lidar_points, drone_xy,
            direct_obstacle_points=direct_obstacles)

        stamp = self.get_clock().now().to_msg()

        # === ONLINE REPLANNING: Run A* from CURRENT robot position ===
        # Plan directly to the current global goal; do not route through
        # nav-graph waypoints, so a previous explored graph node cannot block
        # or pull the robot backward when a new goal is sent.
        path = self._planner.plan(self._grid_map, drone_xy, self._goal[:2])

        if path:
            # Publish path
            path_msg = Path()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = self._pose_frame
            for wx, wy in path:
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = self._pose_frame
                pose.pose.position.x = float(wx)
                pose.pose.position.y = float(wy)
                pose.pose.position.z = self._goal[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self._path_pub.publish(path_msg)

            # Publish local goal (last waypoint)
            local_goal_msg = PoseStamped()
            local_goal_msg.header.stamp = stamp
            local_goal_msg.header.frame_id = self._pose_frame
            local_goal_msg.pose.position.x = float(path[-1][0])
            local_goal_msg.pose.position.y = float(path[-1][1])
            local_goal_msg.pose.position.z = self._goal[2]
            local_goal_msg.pose.orientation.w = 1.0
            self._lgpal_pub.publish(local_goal_msg)

            self.get_logger().info(
                f'[A*] NAVIGATING  dist_to_goal={dist_to_goal:.2f} m  '
                f'robot=({drone_xy[0]:.2f},{drone_xy[1]:.2f})  '
                f'goal=({self._goal[0]:.2f},{self._goal[1]:.2f})  '
                f'path={len(path)} wpts  local_goal=({path[-1][0]:.2f},{path[-1][1]:.2f})',
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().warn('[A*] No path found', throttle_duration_sec=2.0)

        # Publish occupancy grid (correct ROS row-major: x=column, y=row -> transpose)
        if self._grid_map.gmap is not None:
            ogm = OccupancyGrid()
            ogm.header.stamp = stamp
            ogm.header.frame_id = self._pose_frame
            ogm.info.resolution = self._grid_map.reso
            ogm.info.width = self._grid_map.cells
            ogm.info.height = self._grid_map.cells
            ogm.info.origin.position.x = self._grid_map.minx
            ogm.info.origin.position.y = self._grid_map.miny
            ogm.info.origin.orientation.w = 1.0
            scaled = (self._grid_map.gmap.T.flatten() * 100.0).clip(0, 100).astype(np.int8)
            ogm.data = scaled.tolist()
            self._grid_pub.publish(ogm)

            # Publish raw grid for MPC node
            raw_msg = Float32MultiArray()
            gm = self._grid_map
            meta = [float(gm.minx), float(gm.miny), float(gm.reso), float(gm.cells)]
            raw_msg.data = meta + gm.gmap.flatten(order='C').astype(np.float32).tolist()
            self._raw_pub.publish(raw_msg)

            # Publish 2.5D max-z layer for find_safe_stop_pose. Same row-major
            # layout as grid_raw so a host-side consumer can decode both with
            # identical meta. NaN sentinel for untouched cells.
            if gm.hmap is not None:
                height_msg = Float32MultiArray()
                height_msg.data = meta + gm.hmap.flatten(order='C').astype(np.float32).tolist()
                self._height_pub.publish(height_msg)

            # Publish 3D voxel layer as a sparse PointCloud2 — one point per
            # occupied voxel centre. RViz renders this with the same
            # PointCloud2 display class used for the lidar cloud, just at
            # the cell centres for crisp axis-aligned cubes.
            if gm.vmap is not None:
                occ = np.argwhere(gm.vmap > 0.0)
                if occ.size > 0:
                    half = gm.reso * 0.5
                    wx = occ[:, 0].astype(np.float32) * gm.reso + gm.minx + half
                    wy = occ[:, 1].astype(np.float32) * gm.reso + gm.miny + half
                    wz = occ[:, 2].astype(np.float32) * gm.reso + gm.z_min + half
                    voxel_pts = np.stack([wx, wy, wz], axis=1).tolist()
                    hdr_v = Header(stamp=stamp, frame_id=self._pose_frame)
                    self._voxel_pub.publish(
                        point_cloud2.create_cloud_xyz32(hdr_v, voxel_pts)
                    )

        # Publish all persistent obstacle cells as a PointCloud2 for RViz
        all_cells = self._persistent_map.get_points_in_window(
            -1e9, -1e9, 1e9, 1e9  # unbounded — publish everything stored
        )
        if all_cells is not None:
            hdr = Header(stamp=stamp, frame_id=self._pose_frame)
            pc = point_cloud2.create_cloud_xyz32(hdr, all_cells[:, :3].tolist())
            self._pmap_pub.publish(pc)


    # ── External-grid replanning (external_grid backend) ───────────────

    def _replan_external(self):
        """Replan on a pre-computed cost grid (e.g. the nvblox ESDF adapter).

        Mirrors the Gaussian replan tail (path / local-goal / occupancy-grid
        publication) but skips all live-cloud / persistent-map / fusion work —
        the external grid IS the costmap. MPC obstacle constraints come from the
        adapter's discrete /nvblox_g1/obstacle_points, not from this node.
        """
        if self._pose is None or self._external_grid is None:
            self.get_logger().warn(
                f'[A*-EXT] Blocked: pose={self._pose is not None}  '
                f'external_grid={self._external_grid is not None}',
                throttle_duration_sec=3.0,
            )
            return
        if not self._goal_initialized:
            self.get_logger().info(
                '[A*-EXT] Waiting for first goal on /global_goal ...',
                throttle_duration_sec=5.0,
            )
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._external_grid_t > self._external_costmap_max_age:
            self.get_logger().warn(
                '[A*-EXT] external costmap stale — holding (no replan)',
                throttle_duration_sec=3.0,
            )
            return

        grid = self._external_grid
        drone_xy = np.array([
            self._pose.pose.position.x,
            self._pose.pose.position.y,
        ])

        # Goal-reached guard — identical to the Gaussian path.
        dist_to_goal = float(np.linalg.norm(drone_xy - self._goal[:2]))
        if dist_to_goal <= self._goal_reached_radius:
            if not self._goal_reached:
                self.get_logger().info(f'[A*-EXT] Goal reached! dist={dist_to_goal:.2f} m')
                self._goal_reached = True
            return
        self._goal_reached = False

        stamp = self.get_clock().now().to_msg()
        path = self._planner.plan(grid, drone_xy, self._goal[:2])

        if path:
            path_msg = Path()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = self._pose_frame
            for wx, wy in path:
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = self._pose_frame
                pose.pose.position.x = float(wx)
                pose.pose.position.y = float(wy)
                pose.pose.position.z = self._goal[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self._path_pub.publish(path_msg)

            local_goal_msg = PoseStamped()
            local_goal_msg.header.stamp = stamp
            local_goal_msg.header.frame_id = self._pose_frame
            local_goal_msg.pose.position.x = float(path[-1][0])
            local_goal_msg.pose.position.y = float(path[-1][1])
            local_goal_msg.pose.position.z = self._goal[2]
            local_goal_msg.pose.orientation.w = 1.0
            self._lgpal_pub.publish(local_goal_msg)

            self.get_logger().info(
                f'[A*-EXT] NAVIGATING  dist_to_goal={dist_to_goal:.2f} m  '
                f'robot=({drone_xy[0]:.2f},{drone_xy[1]:.2f})  '
                f'goal=({self._goal[0]:.2f},{self._goal[1]:.2f})  path={len(path)} wpts',
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().warn('[A*-EXT] No path found', throttle_duration_sec=2.0)

        # Re-publish the external grid as an OccupancyGrid for RViz/debug so the
        # same /a_star/occupancy_grid display works regardless of backend.
        if grid.gmap is not None:
            ogm = OccupancyGrid()
            ogm.header.stamp = stamp
            ogm.header.frame_id = self._pose_frame
            ogm.info.resolution = grid.reso
            ogm.info.width = grid.cells
            ogm.info.height = grid.cells
            ogm.info.origin.position.x = grid.minx
            ogm.info.origin.position.y = grid.miny
            ogm.info.origin.orientation.w = 1.0
            scaled = (grid.gmap.T.flatten() * 100.0).clip(0, 100).astype(np.int8)
            ogm.data = scaled.tolist()
            self._grid_pub.publish(ogm)


def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
