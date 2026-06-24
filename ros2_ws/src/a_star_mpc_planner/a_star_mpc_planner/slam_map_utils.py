"""
slam_map_utils.py — pure-Python SLAM map fusion helpers.

No ROS2 / rclpy imports: this module is importable in plain pytest without a
running ROS2 environment.  AStarNode delegates all OccupancyGrid crop-and-
transform work here so the logic is unit-testable in isolation.
"""

import math

import numpy as np


def extract_slam_obstacle_points(
    slam_map,
    slam_map_t: float,
    now: float,
    slam_map_max_age: float,
    slam_map_occ_thr: int,
    slam_map_unknown_obstacle: bool,
    robot_xy: np.ndarray,
    grid_half_width: float,
    planning_height: float,
    pose_frame: str,
    tf_dx: float = 0.0,
    tf_dy: float = 0.0,
    tf_yaw: float = 0.0,
    logger=None,
) -> "np.ndarray | None":
    """Crop occupied SLAM-map cells around the robot into the planner frame.

    Parameters
    ----------
    slam_map      : nav_msgs/OccupancyGrid (or duck-typed SimpleNamespace)
    slam_map_t    : monotonic timestamp (s) when the map was last received
    now           : current monotonic time (s)
    slam_map_max_age : drop stale maps older than this many seconds
    slam_map_occ_thr : occupancy value threshold [0..100]; cells >= this
                        are treated as obstacles
    slam_map_unknown_obstacle : if True, unknown cells (-1) are obstacles
    robot_xy      : (2,) array — robot position in the planner (odom) frame
    grid_half_width : local planning window half-size in metres
    planning_height : Z value assigned to all extracted obstacle points
    pose_frame    : name of the planner frame (only used for logging)
    tf_dx, tf_dy  : translation component of map→odom transform
    tf_yaw        : yaw component of map→odom transform (radians)
    logger        : optional logger with .warning() method

    Returns
    -------
    Nx3 float32 array of obstacle point centres in the planner frame,
    or None when the map is unavailable, stale, or contains no occupied cells.
    """
    def _warn(msg):
        if logger is not None:
            logger.warning(msg, throttle_duration_sec=5.0)

    if slam_map is None:
        return None

    if now - slam_map_t > slam_map_max_age:
        _warn('[A*-SLAM] map is stale — skipping fusion this cycle')
        return None

    reso = slam_map.info.resolution
    if reso <= 0.0:
        return None

    ox     = slam_map.info.origin.position.x
    oy     = slam_map.info.origin.position.y
    width  = slam_map.info.width
    height = slam_map.info.height

    # ── Inverse transform: robot position in map frame ────────────────
    # map→odom is (dx, dy, yaw); inverse is odom→map: rotate by -yaw then
    # subtract (dx, dy).
    cos_inv = math.cos(-tf_yaw)
    sin_inv = math.sin(-tf_yaw)
    rx_map = cos_inv * (robot_xy[0] - tf_dx) - sin_inv * (robot_xy[1] - tf_dy)
    ry_map = sin_inv * (robot_xy[0] - tf_dx) + cos_inv * (robot_xy[1] - tf_dy)

    # ── Crop local window in cell indices ─────────────────────────────
    col_min = max(0,      int((rx_map - grid_half_width - ox) / reso))
    col_max = min(width,  int((rx_map + grid_half_width - ox) / reso) + 1)
    row_min = max(0,      int((ry_map - grid_half_width - oy) / reso))
    row_max = min(height, int((ry_map + grid_half_width - oy) / reso) + 1)

    if col_min >= col_max or row_min >= row_max:
        return None

    # OccupancyGrid data: int8, 0=free, 100=occupied, -1=unknown.
    # Cast to int16 to handle the -1 sentinel without sign-extension issues
    # when comparing against a positive threshold.
    raw = slam_map.data
    if isinstance(raw, (bytes, bytearray)):
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.int16)
    else:
        arr = np.array(raw, dtype=np.int16)
    grid = arr.reshape(height, width)
    crop = grid[row_min:row_max, col_min:col_max]

    if slam_map_unknown_obstacle:
        mask = (crop >= slam_map_occ_thr) | (crop < 0)
    else:
        mask = crop >= slam_map_occ_thr

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    # ── Cell centres in map frame ─────────────────────────────────────
    wx_map = (col_min + cols + 0.5) * reso + ox
    wy_map = (row_min + rows + 0.5) * reso + oy

    # ── Forward transform: map frame → planner (odom) frame ──────────
    cos_fw = math.cos(tf_yaw)
    sin_fw = math.sin(tf_yaw)
    wx = cos_fw * wx_map - sin_fw * wy_map + tf_dx
    wy = sin_fw * wx_map + cos_fw * wy_map + tf_dy
    wz = np.full(len(rows), planning_height, dtype=np.float32)

    return np.stack([wx, wy, wz], axis=1).astype(np.float32)
