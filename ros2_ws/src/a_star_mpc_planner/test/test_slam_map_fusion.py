"""
Unit tests for slam_map_utils.extract_slam_obstacle_points.

Pure Python — no ROS2 / rclpy runtime required.  All tests run with plain
``pytest`` from the workspace root.

Covered:
  1.  Occupied cells above threshold become local obstacle points.
  2.  Cells below threshold are ignored.
  3.  Unknown cells (-1) ignored when slam_map_unknown_is_obstacle=False.
  4.  Unknown cells become obstacles when slam_map_unknown_is_obstacle=True.
  5.  Only cells inside the local planning window are extracted.
  6.  A stale map (age > max_age_sec) returns None.
  7.  No map available (None) returns None.
  8.  planning_height is applied as Z coordinate.
  9.  Identity TF (map==odom) leaves coordinates unchanged.
  10. Pure translation TF shifts coordinates correctly.
  11. 90° rotation TF transforms coordinates correctly.
  12. Multiple occupied cells are all returned.
  13. Output shape is always Nx3.
"""

import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

# Make the package importable without a colcon install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from a_star_mpc_planner.slam_map_utils import extract_slam_obstacle_points  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(width, height, reso, ox, oy, data, frame_id="map"):
    """Return a SimpleNamespace that duck-types nav_msgs/OccupancyGrid."""
    g = types.SimpleNamespace()
    g.header = types.SimpleNamespace(frame_id=frame_id)
    g.info = types.SimpleNamespace(
        resolution=reso,
        width=width,
        height=height,
        origin=types.SimpleNamespace(
            position=types.SimpleNamespace(x=ox, y=oy, z=0.0)
        ),
    )
    # Store as bytes, matching the serialised ROS field used in the real node.
    g.data = bytes(np.array(data, dtype=np.int8).tobytes())
    return g


def _call(
    grid,
    robot_xy=(0.0, 0.0),
    now=100.0,
    slam_map_t=99.0,       # 1 s ago → fresh by default
    max_age=5.0,
    occ_thr=50,
    unknown_obstacle=False,
    half_width=5.0,
    planning_height=0.0,
    tf_dx=0.0,
    tf_dy=0.0,
    tf_yaw=0.0,
):
    """Thin wrapper so tests don't have to spell out every keyword."""
    return extract_slam_obstacle_points(
        slam_map=grid,
        slam_map_t=slam_map_t,
        now=now,
        slam_map_max_age=max_age,
        slam_map_occ_thr=occ_thr,
        slam_map_unknown_obstacle=unknown_obstacle,
        robot_xy=np.asarray(robot_xy, dtype=float),
        grid_half_width=half_width,
        planning_height=planning_height,
        pose_frame="odom",
        tf_dx=tf_dx,
        tf_dy=tf_dy,
        tf_yaw=tf_yaw,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSlamMapFusion(unittest.TestCase):

    # ── None-return guards ──────────────────────────────────────────────

    def test_no_map_returns_none(self):
        result = extract_slam_obstacle_points(
            slam_map=None,
            slam_map_t=99.0, now=100.0, slam_map_max_age=5.0,
            slam_map_occ_thr=50, slam_map_unknown_obstacle=False,
            robot_xy=np.array([0.0, 0.0]), grid_half_width=5.0,
            planning_height=0.0, pose_frame="odom",
        )
        self.assertIsNone(result)

    def test_stale_map_returns_none(self):
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, [100] * 100)
        result = _call(grid, now=100.0, slam_map_t=90.0, max_age=5.0)
        self.assertIsNone(result)

    def test_fresh_map_not_dropped(self):
        """Map 4.9 s old (< max_age 5 s) is not dropped."""
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, [100] * 100)
        result = _call(grid, now=100.0, slam_map_t=95.1, max_age=5.0)
        self.assertIsNotNone(result)

    def test_all_free_returns_none(self):
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, [0] * 100)
        self.assertIsNone(_call(grid))

    # ── Threshold filtering ─────────────────────────────────────────────

    def test_occupied_above_threshold_extracted(self):
        """Cells with value >= occ_threshold become obstacle points."""
        # 10×10, 0.1 m/cell, origin (-0.5, -0.5).
        # Mark cell (row=5, col=5): centre at (-0.5 + 5.5*0.1, same) = (0.05, 0.05).
        data = [0] * 100
        data[5 * 10 + 5] = 100
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid)
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(float(pts[0, 0]), 0.05, places=5)
        self.assertAlmostEqual(float(pts[0, 1]), 0.05, places=5)

    def test_exactly_at_threshold_extracted(self):
        data = [0] * 100
        data[50] = 50          # exactly at threshold
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid, occ_thr=50)
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 1)

    def test_below_threshold_ignored(self):
        data = [0] * 100
        data[50] = 49          # one below threshold
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        self.assertIsNone(_call(grid, occ_thr=50))

    # ── Unknown-cell handling ───────────────────────────────────────────

    def test_unknown_ignored_by_default(self):
        data = [0] * 100
        data[50] = -1
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        self.assertIsNone(_call(grid, unknown_obstacle=False))

    def test_unknown_treated_as_obstacle_when_enabled(self):
        data = [0] * 100
        data[50] = -1
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid, unknown_obstacle=True)
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 1)

    # ── Window cropping ─────────────────────────────────────────────────

    def test_cells_outside_window_excluded(self):
        """Only cells inside the local half-width window are returned."""
        # 100×100 grid, 0.1 m/cell, origin (-5, -5) → world spans [-5, 5]².
        # Robot at (0, 0), half_width=1.0 → window [-1, 1]².
        # Cell (row=50, col=50): centre = (-5 + 50.5*0.1, same) = (0.05, 0.05) — inside.
        # Cell (row=0,  col=0):  centre = (-5 + 0.5*0.1,  same) = (-4.95, -4.95) — outside.
        w, h = 100, 100
        data = [0] * (w * h)
        data[50 * w + 50] = 100   # inside  — world (0.05, 0.05)
        data[0  * w + 0]  = 100   # outside — world (-4.95, -4.95)
        grid = _make_grid(w, h, 0.1, -5.0, -5.0, data)
        pts = _call(grid, half_width=1.0)
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 1)
        self.assertAlmostEqual(float(pts[0, 0]), 0.05, places=4)
        self.assertAlmostEqual(float(pts[0, 1]), 0.05, places=4)

    # ── Z / planning height ─────────────────────────────────────────────

    def test_planning_height_applied(self):
        data = [0] * 100
        data[50] = 100
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid, planning_height=0.75)
        self.assertIsNotNone(pts)
        self.assertAlmostEqual(float(pts[0, 2]), 0.75, places=5)

    # ── TF transform ────────────────────────────────────────────────────

    def test_identity_tf_coordinates_unchanged(self):
        """With identity TF (map==odom) world coordinates are unchanged."""
        data = [0] * 100
        data[5 * 10 + 5] = 100
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid, tf_dx=0.0, tf_dy=0.0, tf_yaw=0.0)
        self.assertIsNotNone(pts)
        self.assertAlmostEqual(float(pts[0, 0]), 0.05, places=5)
        self.assertAlmostEqual(float(pts[0, 1]), 0.05, places=5)

    def test_pure_translation_tf(self):
        """Pure TF translation shifts obstacle coordinates correctly."""
        # Occupied cell centre in map frame: (0.05, 0.05).
        # map→odom translation (3, 2) → expected odom (3.05, 2.05).
        # Robot is at odom (3, 2) so map-frame robot = (0, 0) → within window.
        data = [0] * 100
        data[5 * 10 + 5] = 100
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid, robot_xy=(3.0, 2.0), tf_dx=3.0, tf_dy=2.0,
                    half_width=5.0)
        self.assertIsNotNone(pts)
        self.assertAlmostEqual(float(pts[0, 0]), 3.05, places=4)
        self.assertAlmostEqual(float(pts[0, 1]), 2.05, places=4)

    def test_rotation_90deg_ccw(self):
        """90° CCW map→odom rotation correctly transforms coordinates.

        Occupied cell at map (1.05, 0.05).
        After 90° CCW rotation (yaw = π/2):
            x_odom = cos(π/2)*1.05 - sin(π/2)*0.05 + 0 = -0.05
            y_odom = sin(π/2)*1.05 + cos(π/2)*0.05 + 0 =  1.05
        """
        # 20×10 grid, 0.1 m/cell, origin (0, -0.5).
        # Cell (row=5, col=10): centre x = 0 + 10.5*0.1 = 1.05,
        #                        centre y = -0.5 + 5.5*0.1 = 0.05.
        w, h = 20, 10
        data = [0] * (w * h)
        data[5 * w + 10] = 100
        grid = _make_grid(w, h, 0.1, 0.0, -0.5, data)
        pts = _call(grid, robot_xy=(0.0, 0.0), tf_yaw=math.pi / 2,
                    half_width=5.0)
        self.assertIsNotNone(pts)
        self.assertAlmostEqual(float(pts[0, 0]), -0.05, places=4)
        self.assertAlmostEqual(float(pts[0, 1]),  1.05, places=4)

    # ── Multi-cell and shape ────────────────────────────────────────────

    def test_multiple_occupied_cells_all_returned(self):
        data = [0] * 100
        for idx in [11, 22, 33, 44, 55]:
            data[idx] = 100
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid, half_width=5.0)
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 5)

    def test_output_shape_is_nx3(self):
        data = [0] * 100
        data[50] = 100
        data[60] = 100
        grid = _make_grid(10, 10, 0.1, -0.5, -0.5, data)
        pts = _call(grid)
        self.assertIsNotNone(pts)
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
