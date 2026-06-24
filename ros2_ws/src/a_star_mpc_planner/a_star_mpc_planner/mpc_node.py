"""
MPC tracker ROS2 node for the Unitree G1 humanoid (Navigation / DLIO stack).

Ported from the G1_navigation planner and re-adapted to the Navigation
deployment layer:
  - pose comes from DLIO odometry (nav_msgs/Odometry on /dlio/odom_node/odom),
    converted to a PoseStamped internally (frame `odom`). Body-frame velocity is
    still estimated by low-pass pose differentiation, so DLIO's twist convention
    is not relied upon.
  - obstacles come from g1_local_map's ground-removed obstacle cloud
    (/local_voxel_map/obstacles, odom frame). It is already ground-removed, so
    the MPC z-band stays disabled and the security grid runs with ground
    segmentation off (ground_segment_en=false).
  - the velocity command is published as a Twist on /mpc/cmd_vel. The Navigation
    AMO policy is NOT a ROS 2 process: g1_sim_bridge/cmd_vel_to_amo_node forwards
    that Twist as {vx,vy,yaw} JSON to the AMO WebSocket server (:8766). See
    docs/system_architecture.md.

Architecture
------------
  Subscribes:
    /dlio/odom_node/odom       nav_msgs/Odometry — robot pose (→ PoseStamped)
    /local_voxel_map/obstacles PointCloud2       — ground-removed obstacle cloud
    /a_star/path               nav_msgs/Path     — local A* path

  Publishes:
    /mpc/predicted_path  nav_msgs/Path               — N-step MPC predicted trajectory
    /mpc/next_setpoint   geometry_msgs/PoseStamped   — lookahead setpoint
    /mpc/cmd_vel         geometry_msgs/Twist         — velocity command (→ AMO via WS bridge)
    /mpc/diagnostics     std_msgs/Float64MultiArray  — [success, cost, solve_ms, avg_ms,
                                                        fails, security, vx_eff]

  Control flow (at mpc_rate_hz):
    1. Build 6-D state [px, py, yaw, vx, vy, wz] from pose history.
    2. Check LiDAR scan age; mark stale if too old.
    3. Predict obstacle positions at horizon-midpoint via frame-to-frame tracking.
    4. Solve MPCTracker -> predicted state trajectory.
    5. Adaptive velocity limits: reduce on high failure rate, recover when healthy.
    6. Walk predicted trajectory to find lookahead setpoint; ramp down near goal.
    7. Publish setpoint, predicted path, cmd_vel, diagnostics.
"""

import math
from collections import deque
from typing import Optional

import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap
from a_star_mpc_planner.mpc_tracker import MPCConfig, MPCTracker

# CubicSpline for path smoothing (fix #5); graceful fallback if unavailable
try:
    from scipy.interpolate import CubicSpline as _CubicSpline
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _yaw_to_quat(yaw: float) -> tuple:
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))  # (qx, qy, qz, qw)


class MPCNode(Node):

    def __init__(self):
        super().__init__('mpc_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('mpc_N',            30)
        self.declare_parameter('mpc_dt',           0.1)
        self.declare_parameter('mpc_tau_v',        0.12)   # actuator lag — fix #3/#4
        self.declare_parameter('mpc_tau_w',        0.10)
        self.declare_parameter('mpc_vx_max',       1.0)
        self.declare_parameter('mpc_vy_max',       0.5)
        self.declare_parameter('mpc_omega_max',    1.5)
        self.declare_parameter('mpc_v_ref',        0.5)
        self.declare_parameter('mpc_Q_x',          20.0)
        self.declare_parameter('mpc_Q_y',          20.0)
        self.declare_parameter('mpc_Q_yaw',         0.5)
        self.declare_parameter('mpc_Q_terminal',   50.0)
        self.declare_parameter('mpc_R_vx',          1.0)
        self.declare_parameter('mpc_R_vy',          1.0)
        self.declare_parameter('mpc_R_omega',       0.5)
        self.declare_parameter('mpc_R_jerk',       0.2)
        self.declare_parameter('mpc_W_obs_sigmoid',       500.0)
        self.declare_parameter('mpc_obs_alpha',           8.0)
        self.declare_parameter('mpc_obs_r',               0.5)
        self.declare_parameter('mpc_max_obs_constraints', 15)
        self.declare_parameter('mpc_obs_check_radius',    2.0)
        self.declare_parameter('mpc_max_iter',   100)
        self.declare_parameter('mpc_warm_start',  True)
        self.declare_parameter('mpc_rate_hz',       2.0)
        self.declare_parameter('mpc_lookahead_dist', 0.5)
        self.declare_parameter('max_lidar_range',  6.0)
        self.declare_parameter('mpc_path_resample_ds', 0.20)
        self.declare_parameter('mpc_path_smooth_window', 5)
        self.declare_parameter('mpc_setpoint_alpha', 0.35)
        self.declare_parameter('mpc_setpoint_max_step', 0.30)
        self.declare_parameter('mpc_setpoint_reset_dist', 1.25)

        # Velocity estimation (#3/#4)
        self.declare_parameter('vel_filter_alpha', 0.30)

        # LiDAR staleness (#6)
        self.declare_parameter('lidar_max_age_sec', 0.30)
        # Pose source: DLIO odometry (nav_msgs/Odometry), converted internally.
        self.declare_parameter('odom_topic', '/dlio/odom_node/odom')
        # Obstacle topic. Default = g1_local_map's ground-removed obstacle cloud
        # (odom frame). It is already ground-removed, so obstacle_z_min/_max stay
        # disabled (below) and the security grid runs ground_segment_en=false.
        self.declare_parameter('obstacle_topic', '/local_voxel_map/obstacles')
        # Ground/ceiling removal for the obstacle cloud. The MPC collapses the
        # cloud to 2D (obs_2d = points[:, :2]), so any floor return becomes a
        # phantom obstacle the robot would brake/swerve for. Keep only obstacles
        # in [obstacle_z_min, obstacle_z_max] when enabled.
        #
        # DISABLED by default (z_min > z_max → no filtering): the Navigation
        # obstacle source /local_voxel_map/obstacles is already ground-removed by
        # g1_local_map, so a second z-band here would only delete genuine
        # obstacles. (Also note the DLIO odom frame is sensor-origin — floor at
        # z≈-1 m — so a floor-at-0 z-band would be wrong anyway.) Leave disabled
        # unless you point obstacle_topic at a raw, ground-carrying cloud.
        self.declare_parameter('obstacle_z_min', 1.0)
        self.declare_parameter('obstacle_z_max', -1.0)

        # Dynamic obstacle prediction (#10)
        self.declare_parameter('obs_predict_frac', 0.50)  # fraction of horizon

        # Adaptive velocity limits (#9)
        self.declare_parameter('adaptive_vel_limits', True)

        # Security protocol
        self.declare_parameter('grid_reso',                  0.25)
        self.declare_parameter('grid_half_width',            5.0)
        self.declare_parameter('grid_std',                   0.2)
        self.declare_parameter('mpc_security_threshold',     0.35)
        self.declare_parameter('mpc_security_escape_radius', 3.0)
        # Ground segmentation for the security grid. The obstacle cloud from
        # g1_local_map is already ground-removed, so leave this off — per-cell
        # re-segmentation of an already-clean cloud could drop low obstacles and
        # blind the in-obstacle security check.
        self.declare_parameter('ground_segment_en',          False)

        # ── Build MPCConfig and tracker ───────────────────────────────
        cfg = MPCConfig(
            N             = int(self.get_parameter('mpc_N').value),
            dt            = float(self.get_parameter('mpc_dt').value),
            tau_v         = float(self.get_parameter('mpc_tau_v').value),
            tau_w         = float(self.get_parameter('mpc_tau_w').value),
            vx_max        = float(self.get_parameter('mpc_vx_max').value),
            vy_max        = float(self.get_parameter('mpc_vy_max').value),
            omega_max     = float(self.get_parameter('mpc_omega_max').value),
            v_ref         = float(self.get_parameter('mpc_v_ref').value),
            Q_x           = float(self.get_parameter('mpc_Q_x').value),
            Q_y           = float(self.get_parameter('mpc_Q_y').value),
            Q_yaw         = float(self.get_parameter('mpc_Q_yaw').value),
            Q_terminal    = float(self.get_parameter('mpc_Q_terminal').value),
            R_vx          = float(self.get_parameter('mpc_R_vx').value),
            R_vy          = float(self.get_parameter('mpc_R_vy').value),
            R_omega       = float(self.get_parameter('mpc_R_omega').value),
            R_jerk        = float(self.get_parameter('mpc_R_jerk').value),
            W_obs_sigmoid       = float(self.get_parameter('mpc_W_obs_sigmoid').value),
            obs_alpha           = float(self.get_parameter('mpc_obs_alpha').value),
            obs_r               = float(self.get_parameter('mpc_obs_r').value),
            max_obs_constraints = int(self.get_parameter('mpc_max_obs_constraints').value),
            obs_check_radius    = float(self.get_parameter('mpc_obs_check_radius').value),
            max_iter      = int(self.get_parameter('mpc_max_iter').value),
            warm_start    = bool(self.get_parameter('mpc_warm_start').value),
        )
        self._tracker = MPCTracker(config=cfg)
        self._cfg = cfg

        # ── Security protocol ─────────────────────────────────────────
        self._security_threshold    = float(self.get_parameter('mpc_security_threshold').value)
        self._security_escape_radius = float(self.get_parameter('mpc_security_escape_radius').value)
        self._grid = FixedGaussianGridMap(
            reso=float(self.get_parameter('grid_reso').value),
            half_width=float(self.get_parameter('grid_half_width').value),
            std=float(self.get_parameter('grid_std').value),
            ground_segment_en=bool(self.get_parameter('ground_segment_en').value),
        )
        self._security_mode: bool = False

        self._max_lidar_range     = float(self.get_parameter('max_lidar_range').value)
        self._lookahead_dist      = float(self.get_parameter('mpc_lookahead_dist').value)
        self._path_resample_ds    = float(self.get_parameter('mpc_path_resample_ds').value)
        self._path_smooth_window  = int(self.get_parameter('mpc_path_smooth_window').value)
        self._setpoint_alpha      = float(self.get_parameter('mpc_setpoint_alpha').value)
        self._setpoint_max_step   = float(self.get_parameter('mpc_setpoint_max_step').value)
        self._setpoint_reset_dist = float(self.get_parameter('mpc_setpoint_reset_dist').value)

        # ── Velocity estimation (#3/#4) ───────────────────────────────
        self._vel_filter_alpha = float(self.get_parameter('vel_filter_alpha').value)
        self._prev_pose_sec: Optional[float] = None
        self._prev_pose_xy:  Optional[np.ndarray] = None
        self._prev_pose_yaw: float = 0.0
        self._vx_est = 0.0
        self._vy_est = 0.0
        self._wz_est = 0.0

        # ── LiDAR staleness (#6) ──────────────────────────────────────
        self._lidar_max_age_sec = float(self.get_parameter('lidar_max_age_sec').value)
        self._last_scan_stamp: Optional[rclpy.time.Time] = None
        self._obstacle_topic = str(self.get_parameter('obstacle_topic').value)
        self._obstacle_z_min = float(self.get_parameter('obstacle_z_min').value)
        self._obstacle_z_max = float(self.get_parameter('obstacle_z_max').value)

        # ── Dynamic obstacle prediction (#10) ─────────────────────────
        self._obs_predict_frac = float(self.get_parameter('obs_predict_frac').value)
        self._prev_obs_pts:  Optional[np.ndarray] = None   # (M, 2) selected last cycle
        self._prev_obs_time: Optional[float]      = None   # perf_counter seconds

        # ── Adaptive velocity limits (#9) ─────────────────────────────
        self._adaptive_enabled  = bool(self.get_parameter('adaptive_vel_limits').value)
        self._cfg_vx_max        = cfg.vx_max    # configured ceiling
        self._cfg_vy_max        = cfg.vy_max
        self._cfg_omega_max     = cfg.omega_max
        self._adaptive_vx_max   = cfg.vx_max    # current effective limit
        self._adaptive_vy_max   = cfg.vy_max
        self._adaptive_omega_max = cfg.omega_max
        self._recent_solves: deque = deque(maxlen=20)

        # ── Subscriptions state ───────────────────────────────────────
        self._pose: PoseStamped | None = None
        self._yaw = 0.0
        self._a_star_path: list | None = None
        self._a_star_path_raw_len = 0
        self._lidar_points: np.ndarray | None = None
        self._setpoint_filtered_xy:  np.ndarray | None = None
        self._setpoint_filtered_yaw: float | None = None

        # Logging counters
        self._solve_count    = 0
        self._fail_count     = 0
        self._total_solve_ms = 0.0

        # ── QoS ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ───────────────────────────────────────────────
        # Pose: DLIO odometry. BEST_EFFORT so a reliable DLIO publisher is still
        # readable and the stack is robust to whatever QoS DLIO ships with.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, odom_qos)
        self.get_logger().info(f'pose source: {odom_topic} (Odometry → PoseStamped)')
        self.create_subscription(PointCloud2, self._obstacle_topic,      self._lidar_cb, sensor_qos)
        self.get_logger().info(f"obstacle source: {self._obstacle_topic}")
        self.create_subscription(Path,        '/a_star/path',            self._path_cb,  10)

        # ── Publishers ────────────────────────────────────────────────
        self._pred_path_pub   = self.create_publisher(Path,                '/mpc/predicted_path',      10)
        self._setpoint_pub    = self.create_publisher(PoseStamped,         '/mpc/next_setpoint',       10)
        self._obs_markers_pub = self.create_publisher(MarkerArray,         '/mpc/predicted_obstacles', 10)
        # /mpc/cmd_vel: the instantaneous velocity command the MPC wants the
        # robot to apply right now. Sourced from x_pred[1, 3:6] — the predicted
        # state-velocity at the first horizon step (≈ what the MPC's first
        # control input would produce under the identified actuator-lag model).
        # cmd_vel_to_ws_node forwards this over the WS bridge as a
        # velocity_target message, which walking_policy.py consumes in place
        # of the heading-first P-controller.
        self._cmd_vel_pub     = self.create_publisher(Twist,               '/mpc/cmd_vel',        10)
        self._diagnostics_pub = self.create_publisher(Float64MultiArray,   '/mpc/diagnostics',    10)

        # ── Solve timer ───────────────────────────────────────────────
        rate = float(self.get_parameter('mpc_rate_hz').value)
        self.create_timer(1.0 / rate, self._solve_cb)

        self.get_logger().info('MPC node ready')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        """Repackage DLIO odometry as the PoseStamped the tracker expects.

        Only the pose is used here; body-frame velocity is re-estimated from the
        pose stream in _pose_cb rather than taken from msg.twist, so DLIO's twist
        frame/convention does not have to be trusted.
        """
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self._pose_cb(ps)

    def _pose_cb(self, msg: PoseStamped):
        """Update pose and estimate body-frame velocity via low-pass pose differentiation."""
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        yaw_new = _quat_to_yaw(qx, qy, qz, qw)
        x_new   = msg.pose.position.x
        y_new   = msg.pose.position.y

        # ── Velocity estimation (#3/#4) ───────────────────────────────
        if (self._prev_pose_sec is not None and
                self._prev_pose_xy is not None):
            dt_pose = now_sec - self._prev_pose_sec
            if 0.01 < dt_pose < 0.5:  # ignore stale or too-fast updates
                dx_w = (x_new - self._prev_pose_xy[0]) / dt_pose
                dy_w = (y_new - self._prev_pose_xy[1]) / dt_pose

                # World → body-frame rotation
                cy = math.cos(yaw_new)
                sy = math.sin(yaw_new)
                vx_raw =  dx_w * cy + dy_w * sy
                vy_raw = -dx_w * sy + dy_w * cy

                # Wrap-aware yaw rate
                dyaw_raw = math.atan2(
                    math.sin(yaw_new - self._prev_pose_yaw),
                    math.cos(yaw_new - self._prev_pose_yaw),
                ) / dt_pose

                # Exponential moving average low-pass filter
                a = self._vel_filter_alpha
                self._vx_est = (1.0 - a) * self._vx_est + a * vx_raw
                self._vy_est = (1.0 - a) * self._vy_est + a * vy_raw
                self._wz_est = (1.0 - a) * self._wz_est + a * dyaw_raw

        self._prev_pose_sec = now_sec
        self._prev_pose_xy  = np.array([x_new, y_new], dtype=float)
        self._prev_pose_yaw = yaw_new

        self._pose = msg
        self._yaw  = yaw_new

        self.get_logger().info(
            f'[MPC-DEBUG] odom pose received: '
            f'pos=({x_new:.4f}, {y_new:.4f})  '
            f'yaw={math.degrees(yaw_new):.1f} deg  '
            f'vel_body=({self._vx_est:.2f}, {self._vy_est:.2f}, {self._wz_est:.2f})',
            throttle_duration_sec=2.0,
        )

    def _lidar_cb(self, msg: PointCloud2):
        """Parse LiDAR points, filter by range, record scan timestamp (#6).

        PRESERVES the previous self._lidar_points on a transient empty/
        all-filtered scan — the staleness check downstream will still
        time them out via msg.header.stamp if the situation persists.
        Wiping to None on every empty scan made the MPC alternate
        between thousands of points and zero, causing the planner to
        skip obstacles on every other cycle even when the sensor was
        actually live.
        """
        # Stamp is updated EVERY message (including empty ones from
        # bridge_node) so scan_age tracks topic liveness, not just
        # non-empty-message age.
        self._last_scan_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

        try:
            points = list(point_cloud2.read_points(msg, skip_nans=True))
        except Exception as e:
            self.get_logger().warn(f'Lidar error: {e}')
            return

        if not points:
            self.get_logger().info(
                '[MPC-LIDAR] empty scan — keeping last valid cloud',
                throttle_duration_sec=3.0,
            )
            return

        arr = np.array([(p[0], p[1], p[2]) for p in points], dtype=float)

        # Ground/ceiling removal: drop points outside the obstacle z-band so the
        # floor (and ceiling) never reach the 2D obstacle projection. Active only
        # when enabled (z_min <= z_max). Left disabled for the default
        # /local_voxel_map/obstacles source, which g1_local_map has already
        # ground-removed — enable it only for a raw, ground-carrying cloud.
        if self._obstacle_z_min <= self._obstacle_z_max:
            zmask = (arr[:, 2] >= self._obstacle_z_min) & (arr[:, 2] <= self._obstacle_z_max)
            arr = arr[zmask]

        if self._pose is not None and len(arr) > 0:
            px = self._pose.pose.position.x
            py = self._pose.pose.position.y
            dists = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
            arr = arr[dists < self._max_lidar_range]

        if len(arr) > 0:
            self._lidar_points = arr
        else:
            self.get_logger().info(
                '[MPC-LIDAR] all points filtered by z-band/range — keeping last valid cloud',
                throttle_duration_sec=3.0,
            )

    def _path_cb(self, msg: Path):
        """Store and smooth the latest A* path."""
        if msg.poses:
            raw_path = [
                (p.pose.position.x, p.pose.position.y, p.pose.position.z)
                for p in msg.poses
            ]
            self._a_star_path_raw_len = len(raw_path)
            smoothed_path = self._smooth_resample_path(raw_path)
            self._a_star_path = smoothed_path

            did_reset = False
            if self._setpoint_filtered_xy is None or self._setpoint_filtered_yaw is None:
                did_reset = True
            else:
                idx = 1 if len(smoothed_path) > 1 else 0
                anchor = np.array([float(smoothed_path[idx][0]),
                                   float(smoothed_path[idx][1])], dtype=float)
                if float(np.linalg.norm(anchor - self._setpoint_filtered_xy)) > self._setpoint_reset_dist:
                    did_reset = True

            if did_reset:
                self._setpoint_filtered_xy  = None
                self._setpoint_filtered_yaw = None

            self.get_logger().info(
                f'[MPC] Received NEW A* path: {len(raw_path)} raw → '
                f'{len(self._a_star_path)} resampled | reset_filter={did_reset}',
                throttle_duration_sec=1.0,
            )
        else:
            self._a_star_path = None
            self._a_star_path_raw_len = 0
            self._setpoint_filtered_xy  = None
            self._setpoint_filtered_yaw = None

    # ── Path smoothing (fix #5 — CubicSpline replaces moving average) ──

    def _smooth_resample_path(self, path_xyz: list) -> list:
        """
        Resample A* path and smooth with CubicSpline (fix #5).

        Falls back to linear interpolation when scipy is unavailable or the
        path is too short for a cubic fit (< 4 waypoints).  Endpoints are
        always preserved exactly.
        """
        if not path_xyz or len(path_xyz) < 2:
            return path_xyz

        xy = np.array([(p[0], p[1]) for p in path_xyz], dtype=float)

        # Remove repeated consecutive points
        dxy  = np.diff(xy, axis=0)
        keep = np.hstack(([True], np.linalg.norm(dxy, axis=1) > 1e-4))
        xy   = xy[keep]
        if len(xy) < 2:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z)]

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(arc[-1])
        if total <= 1e-6:
            z = float(path_xyz[-1][2])
            return [(float(xy[0, 0]), float(xy[0, 1]), z),
                    (float(xy[-1, 0]), float(xy[-1, 1]), z)]

        ds = max(self._path_resample_ds, 1e-2)
        s  = np.arange(0.0, total + 1e-9, ds)
        if s[-1] < total:
            s = np.append(s, total)

        # ── CubicSpline smoothing (fix #5) ────────────────────────────
        if _SCIPY_OK and len(xy) >= 4:
            # Ensure strictly increasing arc (numerical safety)
            arc_safe = arc.copy()
            for i in range(1, len(arc_safe)):
                if arc_safe[i] <= arc_safe[i - 1]:
                    arc_safe[i] = arc_safe[i - 1] + 1e-6

            cs_x = _CubicSpline(arc_safe, xy[:, 0])
            cs_y = _CubicSpline(arc_safe, xy[:, 1])
            x_s  = cs_x(s)
            y_s  = cs_y(s)
            # Pin endpoints to the original path exactly
            x_s[0]  = xy[0, 0];  y_s[0]  = xy[0, 1]
            x_s[-1] = xy[-1, 0]; y_s[-1] = xy[-1, 1]
        else:
            # Fallback: linear interpolation (also used when scipy is absent)
            x_s = np.interp(s, arc, xy[:, 0])
            y_s = np.interp(s, arc, xy[:, 1])

        z = float(path_xyz[-1][2])
        return [(float(x_s[i]), float(y_s[i]), z) for i in range(len(s))]

    # ── Dynamic obstacle prediction (#10) ─────────────────────────────

    def _predict_obs_positions(
        self,
        obs_2d:       np.ndarray,
        predict_sec:  float,
        current_time: float,
    ) -> np.ndarray:
        """
        Predict where each obstacle will be at predict_sec in the future.

        Uses nearest-neighbour matching between consecutive selected-obstacle
        sets to estimate per-point velocity.  Matches with implausible speed
        (> 3 m/s) or large positional jump are ignored.

        Modifies self._prev_obs_pts / self._prev_obs_time as a side-effect.
        """
        predicted = obs_2d.copy()

        if (self._prev_obs_pts is not None and
                self._prev_obs_time is not None and
                len(self._prev_obs_pts) > 0):
            frame_dt = current_time - self._prev_obs_time
            if 0.05 < frame_dt < 1.0:
                for i, pt in enumerate(obs_2d):
                    dists = np.linalg.norm(self._prev_obs_pts - pt, axis=1)
                    j = int(np.argmin(dists))
                    if dists[j] < 0.5:                         # plausible correspondence
                        vel   = (pt - self._prev_obs_pts[j]) / frame_dt
                        speed = float(np.linalg.norm(vel))
                        if speed < 3.0:                        # cap at brisk walking speed
                            predicted[i] = pt + vel * predict_sec

        self._prev_obs_pts  = obs_2d.copy()
        self._prev_obs_time = current_time
        return predicted

    # ── Security escape helper ─────────────────────────────────────────

    def _find_escape_target(
        self,
        grid:     FixedGaussianGridMap,
        robot_xy: np.ndarray,
    ) -> Optional[np.ndarray]:
        """BFS to find the nearest free cell outside the inflated obstacle zone."""
        ix0, iy0 = grid.world_to_index(float(robot_xy[0]), float(robot_xy[1]))
        if ix0 is None:
            return None

        visited: set = set()
        queue:   deque = deque()
        queue.append((ix0, iy0))
        visited.add((ix0, iy0))

        while queue:
            ix, iy = queue.popleft()
            if float(grid.gmap[ix, iy]) < self._security_threshold:
                wx, wy = grid.index_to_world(ix, iy)
                return np.array([wx, wy], dtype=float)
            for dix in (-1, 0, 1):
                for diy in (-1, 0, 1):
                    if dix == 0 and diy == 0:
                        continue
                    nix, niy = ix + dix, iy + diy
                    if (nix, niy) in visited:
                        continue
                    if not (0 <= nix < grid.cells and 0 <= niy < grid.cells):
                        continue
                    wx, wy = grid.index_to_world(nix, niy)
                    if np.hypot(wx - robot_xy[0], wy - robot_xy[1]) > self._security_escape_radius:
                        continue
                    visited.add((nix, niy))
                    queue.append((nix, niy))

        return None

    # ── Predicted obstacle visualization ──────────────────────────────

    def _publish_obstacle_markers(
        self,
        obs_2d: Optional[np.ndarray],
        pose:   Optional[PoseStamped],
    ) -> None:
        """Publish predicted obstacle positions as a RViz MarkerArray.

        Each obstacle cluster centroid is shown as a sphere at the
        predicted future position, coloured by proximity to the robot
        (green → yellow → red as distance decreases past obs_r).
        A DELETE_ALL marker is emitted first so stale markers disappear
        when the obstacle count drops between cycles.
        """
        ma = MarkerArray()

        # Clear previous frame's markers
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        ma.markers.append(del_marker)

        if obs_2d is None or len(obs_2d) == 0 or pose is None:
            self._obs_markers_pub.publish(ma)
            return

        frame_id  = pose.header.frame_id or 'odom'
        stamp     = self.get_clock().now().to_msg()
        robot_xy  = np.array([pose.pose.position.x, pose.pose.position.y])
        obs_r     = self._cfg.obs_r

        # Cluster nearby obstacle points (keep the max_obs_constraints
        # closest centroids to avoid flooding RViz with thousands of markers).
        dists  = np.linalg.norm(obs_2d - robot_xy, axis=1)
        n_show = min(self._cfg.max_obs_constraints, len(obs_2d))
        near_idx = np.argsort(dists)[:n_show]

        for i, idx in enumerate(near_idx):
            wx, wy = float(obs_2d[idx, 0]), float(obs_2d[idx, 1])
            d      = float(dists[idx])

            # Colour by proximity: green > 2×r, yellow ~1.5×r, red ≤ r
            ratio = max(0.0, min(1.0, 1.0 - (d - obs_r) / (obs_r + 1e-6)))
            r_c   = ratio
            g_c   = 1.0 - ratio * 0.7

            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp    = stamp
            m.ns              = 'predicted_obstacles'
            m.id              = i
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.5   # waist height for visibility
            m.pose.orientation.w = 1.0
            m.scale.x         = obs_r * 2.0
            m.scale.y         = obs_r * 2.0
            m.scale.z         = obs_r * 2.0
            m.color.r         = r_c
            m.color.g         = g_c
            m.color.b         = 0.0
            m.color.a         = 0.65
            m.lifetime.sec    = 0
            m.lifetime.nanosec = int(0.5e9)   # 0.5 s — fades if MPC stops publishing
            ma.markers.append(m)

        self._obs_markers_pub.publish(ma)

    # ── Main solve callback ────────────────────────────────────────────

    def _solve_cb(self):
        if self._pose is None or self._a_star_path is None:
            self.get_logger().warn('[MPC] Waiting for pose and path…', throttle_duration_sec=5.0)
            return

        # ── 6-D state (#3/#4) ─────────────────────────────────────────
        state = np.array([
            self._pose.pose.position.x,
            self._pose.pose.position.y,
            self._yaw,
            self._vx_est,
            self._vy_est,
            self._wz_est,
        ])

        # ── LiDAR staleness check (#6) ────────────────────────────────
        obs_2d: Optional[np.ndarray] = None
        scan_age_sec = 0.0
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            if self._last_scan_stamp is not None:
                now_ros   = self.get_clock().now()
                scan_age_sec = (
                    now_ros - self._last_scan_stamp
                ).nanoseconds * 1e-9

            if scan_age_sec <= self._lidar_max_age_sec:
                obs_2d = self._lidar_points[:, :2]
            else:
                self.get_logger().warn(
                    f'[MPC] LiDAR scan stale ({scan_age_sec*1e3:.0f} ms > '
                    f'{self._lidar_max_age_sec*1e3:.0f} ms) — skipping obstacles',
                    throttle_duration_sec=1.0,
                )

        # ── Dynamic obstacle prediction (#10) ─────────────────────────
        if obs_2d is not None and len(obs_2d) > 0:
            predict_sec = self._obs_predict_frac * self._cfg.N * self._cfg.dt
            import time as _time_mod
            obs_2d = self._predict_obs_positions(
                obs_2d, predict_sec, _time_mod.perf_counter()
            )

        # ── Publish predicted obstacle markers ────────────────────────
        self._publish_obstacle_markers(obs_2d, self._pose)

        # ── Obstacle proximity log (world frame + robot-relative) ─────
        robot_xy  = state[:2]
        robot_yaw = state[2]
        cy, sy    = math.cos(robot_yaw), math.sin(robot_yaw)

        if obs_2d is not None and len(obs_2d) > 0:
            dists   = np.linalg.norm(obs_2d - robot_xy, axis=1)
            n_near  = min(5, len(obs_2d))
            near_idx = np.argsort(dists)[:n_near]
            parts   = []
            for idx in near_idx:
                wx, wy = float(obs_2d[idx, 0]), float(obs_2d[idx, 1])
                dx_w   = wx - float(robot_xy[0])
                dy_w   = wy - float(robot_xy[1])
                # World-delta → body frame
                dx_b   =  dx_w * cy + dy_w * sy
                dy_b   = -dx_w * sy + dy_w * cy
                d      = float(dists[idx])
                bearing_deg = math.degrees(math.atan2(dy_b, dx_b))
                parts.append(
                    f'world=({wx:.2f},{wy:.2f}) '
                    f'body=({dx_b:+.2f},{dy_b:+.2f}) '
                    f'd={d:.2f}m bear={bearing_deg:+.0f}°'
                )
            self.get_logger().warn(
                f'[MPC-OBS] {len(obs_2d)} pts in range | '
                f'nearest {n_near}: ' + ' | '.join(parts),
                throttle_duration_sec=0.5,
            )
        else:
            self.get_logger().warn(
                '[MPC-OBS] NO obstacles fed to MPC this cycle '
                f'(stale={scan_age_sec*1e3:.0f}ms, lidar_pts='
                f'{len(self._lidar_points) if self._lidar_points is not None else 0})',
                throttle_duration_sec=0.5,
            )

        # ── Security protocol ─────────────────────────────────────────
        in_inflated    = False
        escape_target: Optional[np.ndarray] = None
        occ_at_robot   = 0.0
        if self._lidar_points is not None and len(self._lidar_points) > 0:
            self._grid.update(self._lidar_points, state[:2])
            occ_at_robot = self._grid.get_probability(float(state[0]), float(state[1]))
            if occ_at_robot >= self._security_threshold:
                in_inflated    = True
                escape_target  = self._find_escape_target(self._grid, state[:2])

        prev_security    = self._security_mode
        self._security_mode = in_inflated
        if in_inflated and not prev_security:
            self._tracker._prev_u = None
            self._tracker._prev_x = None
            self._setpoint_filtered_xy  = None
            self._setpoint_filtered_yaw = None

        mpc_path = self._a_star_path
        if in_inflated:
            z = float(self._a_star_path[-1][2]) if self._a_star_path else 0.0
            if escape_target is not None:
                mpc_path = [
                    (float(state[0]), float(state[1]), z),
                    (float(escape_target[0]), float(escape_target[1]), z),
                ]
                self.get_logger().warn(
                    f'[MPC-SECURITY] occ={occ_at_robot:.3f} — escape → '
                    f'({escape_target[0]:.2f}, {escape_target[1]:.2f})',
                    throttle_duration_sec=0.5,
                )
            else:
                self.get_logger().warn(
                    f'[MPC-SECURITY] occ={occ_at_robot:.3f} — no free cell, holding A* path',
                    throttle_duration_sec=0.5,
                )

        # ── Solve MPC ─────────────────────────────────────────────────
        result = self._tracker.solve(state, mpc_path, obstacle_points_2d=obs_2d)
        result.security_mode = in_inflated

        self._solve_count    += 1
        self._total_solve_ms += result.solve_time_ms
        if not result.success:
            self._fail_count += 1

        # ── Adaptive velocity limits (#9) ─────────────────────────────
        if self._adaptive_enabled:
            self._recent_solves.append(result.success)
            if len(self._recent_solves) >= 10:
                fail_rate = self._recent_solves.count(False) / len(self._recent_solves)
                if fail_rate > 0.30:
                    # Too many failures — reduce velocity ceiling by 10 %
                    new_vx = max(0.15, self._adaptive_vx_max * 0.90)
                    if new_vx < self._adaptive_vx_max:
                        self._adaptive_vx_max = new_vx
                        self._tracker.update_velocity_limits(vx_max=new_vx)
                        self.get_logger().warn(
                            f'[MPC-ADAPTIVE] High failure rate ({fail_rate:.0%}) — '
                            f'reducing vx_max to {new_vx:.2f} m/s',
                            throttle_duration_sec=2.0,
                        )
                elif fail_rate < 0.05 and self._adaptive_vx_max < self._cfg_vx_max:
                    # Healthy — recover velocity ceiling by 5 %
                    new_vx = min(self._cfg_vx_max, self._adaptive_vx_max * 1.05)
                    self._adaptive_vx_max = new_vx
                    self._tracker.update_velocity_limits(vx_max=new_vx)

        # ── Publish predicted trajectory ──────────────────────────────
        if result.x_pred is not None:
            pred_path        = Path()
            pred_path.header = self._pose.header
            for i in range(len(result.x_pred)):
                p                    = PoseStamped()
                p.header             = self._pose.header
                p.pose.position.x    = float(result.x_pred[i, 0])
                p.pose.position.y    = float(result.x_pred[i, 1])
                p.pose.position.z    = self._pose.pose.position.z
                q                    = _yaw_to_quat(float(result.x_pred[i, 2]))
                p.pose.orientation.x = q[0]
                p.pose.orientation.y = q[1]
                p.pose.orientation.z = q[2]
                p.pose.orientation.w = q[3]
                pred_path.poses.append(p)
            self._pred_path_pub.publish(pred_path)

            # ── Lookahead setpoint with near-goal ramp-down (#8) ──────
            robot_pos   = state[:2]
            path_end    = np.array(self._a_star_path[-1][:2], dtype=float)
            dist_to_end = float(np.linalg.norm(path_end - robot_pos))
            eff_lookahead = min(self._lookahead_dist, max(0.3, dist_to_end * 0.5))

            lookahead_idx = len(result.x_pred) - 1
            found = False
            for i in range(1, len(result.x_pred)):
                if float(np.linalg.norm(result.x_pred[i, :2] - robot_pos)) >= eff_lookahead:
                    lookahead_idx = i
                    found = True
                    break

            if found:
                nxt_xy  = result.x_pred[lookahead_idx, :2]
                nxt_yaw = float(result.x_pred[lookahead_idx, 2])
            else:
                last_wp = self._a_star_path[-1]
                nxt_xy  = np.array([float(last_wp[0]), float(last_wp[1])])
                nxt_yaw = self._yaw

            # Setpoint low-pass filter
            nxt_xy = np.asarray(nxt_xy, dtype=float)
            if self._setpoint_filtered_xy is None:
                self._setpoint_filtered_xy  = nxt_xy.copy()
                self._setpoint_filtered_yaw = nxt_yaw
            else:
                jump      = nxt_xy - self._setpoint_filtered_xy
                jump_norm = float(np.linalg.norm(jump))
                if self._setpoint_max_step > 0.0 and jump_norm > self._setpoint_max_step:
                    nxt_xy = self._setpoint_filtered_xy + jump / (jump_norm + 1e-9) * self._setpoint_max_step
                alpha = float(np.clip(self._setpoint_alpha, 0.0, 1.0))
                self._setpoint_filtered_xy = (1.0 - alpha) * self._setpoint_filtered_xy + alpha * nxt_xy
                yaw_err = math.atan2(
                    math.sin(nxt_yaw - self._setpoint_filtered_yaw),
                    math.cos(nxt_yaw - self._setpoint_filtered_yaw),
                )
                self._setpoint_filtered_yaw = self._setpoint_filtered_yaw + alpha * yaw_err
            nxt_xy  = self._setpoint_filtered_xy
            nxt_yaw = self._setpoint_filtered_yaw

            setpoint                    = PoseStamped()
            setpoint.header             = self._pose.header
            setpoint.pose.position.x    = float(nxt_xy[0])
            setpoint.pose.position.y    = float(nxt_xy[1])
            setpoint.pose.position.z    = self._pose.pose.position.z
            q                           = _yaw_to_quat(nxt_yaw)
            setpoint.pose.orientation.x = q[0]
            setpoint.pose.orientation.y = q[1]
            setpoint.pose.orientation.z = q[2]
            setpoint.pose.orientation.w = q[3]
            self._setpoint_pub.publish(setpoint)

            # ── Velocity command (Twist on /mpc/cmd_vel) ──────────────
            # x_pred is (N+1, 6) with state = [px, py, yaw, vx, vy, wz].
            # x_pred[1] is the state predicted one step into the future
            # (after applying the optimal first control), so its velocity
            # fields are a near-faithful proxy for that control under the
            # actuator-lag model. Failed solves return x_pred=x_ref, which
            # carries the reference velocity profile — still a sensible
            # fallback (it tracks the A* path at v_ref), so we publish even
            # on result.success=False rather than going silent.
            cmd_vel = Twist()
            if result.x_pred is not None and len(result.x_pred) >= 2:
                cmd_vel.linear.x  = float(result.x_pred[1, 3])
                cmd_vel.linear.y  = float(result.x_pred[1, 4])
                cmd_vel.angular.z = float(result.x_pred[1, 5])
            self._cmd_vel_pub.publish(cmd_vel)

            self.get_logger().info(
                f'[MPC] #{self._solve_count:04d} '
                f'ok={result.success} '
                f'cost={result.cost:8.1f} '
                f'solve={result.solve_time_ms:5.1f} ms '
                f'avg={self._total_solve_ms / self._solve_count:5.1f} ms  '
                f'fails={self._fail_count}  '
                f'security={self._security_mode}  '
                f'vx_eff={self._adaptive_vx_max:.2f}  '
                f'scan_age={scan_age_sec*1e3:.0f} ms  '
                f'path_wpts={len(self._a_star_path)}(raw={self._a_star_path_raw_len})  '
                f'robot=[{state[0]:.2f},{state[1]:.2f}] '
                f'setpt=[{nxt_xy[0]:.2f},{nxt_xy[1]:.2f}]',
                throttle_duration_sec=0.5,
            )

        # ── Diagnostics ───────────────────────────────────────────────
        diag      = Float64MultiArray()
        diag.data = [
            float(1 if result.success else 0),
            result.cost,
            result.solve_time_ms,
            float(self._total_solve_ms / max(self._solve_count, 1)),
            float(self._fail_count),
            float(1 if result.security_mode else 0),
            float(self._adaptive_vx_max),   # [6] current adaptive vx limit
        ]
        self._diagnostics_pub.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
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
