"""
Unit tests for the A* external_grid costmap backend.

Covers external_grid_map (parse_costmap_raw / ExternalGridMap coordinate
helpers) and AStarPlanner running on synthetic ESDF-derived cost grids.

Pure Python — no ROS2 runtime. Run with plain ``pytest`` from the workspace
root.

Covered:
  parse_costmap_raw / ExternalGridMap
    1.  round-trips the grid_raw flat layout (meta + [ix*cells+iy]).
    2.  world_to_index / index_to_world are inverse on the cell grid.
    3.  world_to_index returns (None, None) outside the grid.
    4.  too-short payload raises ValueError.
    5.  length inconsistent with declared cells raises ValueError.
  AStarPlanner on external grids
    6.  free grid → a path from start to goal.
    7.  full wall across the grid → no path (None).
    8.  a gap in the wall → path exists and threads the gap.
    9.  inflated obstacle → path avoids all cells at/above obstacle_threshold.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from a_star_mpc_planner.a_star_planner import AStarPlanner  # noqa: E402
from a_star_mpc_planner.external_grid_map import (  # noqa: E402
    ExternalGridMap,
    parse_costmap_raw,
)


def make_raw(cost: np.ndarray, minx=0.0, miny=0.0, reso=1.0):
    """Pack a (cells, cells) [ix, iy] cost grid into the grid_raw payload."""
    cells = cost.shape[0]
    meta = [float(minx), float(miny), float(reso), float(cells)]
    return meta + cost.flatten(order='C').astype(np.float32).tolist()


class TestParseCostmapRaw(unittest.TestCase):

    def test_roundtrip(self):
        cells = 5
        cost = np.random.default_rng(0).random((cells, cells)).astype(np.float32)
        grid = parse_costmap_raw(make_raw(cost, minx=-2.0, miny=1.0, reso=0.1))
        self.assertEqual(grid.cells, cells)
        self.assertAlmostEqual(grid.minx, -2.0, places=5)
        self.assertAlmostEqual(grid.miny, 1.0, places=5)
        self.assertAlmostEqual(grid.reso, 0.1, places=5)
        np.testing.assert_allclose(grid.gmap, cost, atol=1e-6)
        self.assertIsNone(grid.hmap)

    def test_coord_inverse(self):
        grid = ExternalGridMap(
            np.zeros((10, 10), np.float32), minx=-1.0, miny=2.0, reso=0.25)
        for ix, iy in [(0, 0), (3, 7), (9, 9)]:
            wx, wy = grid.index_to_world(ix, iy)
            self.assertEqual(grid.world_to_index(wx + 1e-6, wy + 1e-6), (ix, iy))

    def test_outside_grid(self):
        # Matches FixedGaussianGridMap.world_to_index: int() truncates toward
        # zero, so a fully-outside coordinate needs |offset| >= reso.
        grid = ExternalGridMap(np.zeros((4, 4), np.float32), 0.0, 0.0, 1.0)
        self.assertEqual(grid.world_to_index(-1.5, 0.0), (None, None))
        self.assertEqual(grid.world_to_index(0.0, 99.0), (None, None))

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            parse_costmap_raw([0.0, 0.0, 1.0])

    def test_length_mismatch_raises(self):
        # declares cells=3 (needs 9 cost values) but supplies 4
        with self.assertRaises(ValueError):
            parse_costmap_raw([0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0])


class TestAStarOnExternalGrid(unittest.TestCase):

    def setUp(self):
        self.planner = AStarPlanner(obstacle_threshold=0.5, obstacle_cost_weight=10.0)

    def _grid(self, cost):
        return parse_costmap_raw(make_raw(cost))

    def test_free_grid_has_path(self):
        grid = self._grid(np.zeros((10, 10), np.float32))
        path = self.planner.plan(grid, (0.0, 0.0), (9.0, 9.0))
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 1)
        # ends at (or adjacent to) the goal cell
        self.assertAlmostEqual(path[-1][0], 9.0, delta=1.0)
        self.assertAlmostEqual(path[-1][1], 9.0, delta=1.0)

    def test_full_wall_blocks(self):
        cost = np.zeros((10, 10), np.float32)
        cost[5, :] = 1.0  # lethal wall spanning all iy at ix=5
        grid = self._grid(cost)
        path = self.planner.plan(grid, (0.0, 0.0), (9.0, 9.0))
        self.assertIsNone(path)

    def test_wall_with_gap_threads(self):
        cost = np.zeros((10, 10), np.float32)
        cost[5, :] = 1.0
        cost[5, 8] = 0.0  # open a single free cell in the wall
        grid = self._grid(cost)
        path = self.planner.plan(grid, (0.0, 0.0), (9.0, 9.0))
        self.assertIsNotNone(path)
        # every waypoint must sit on a traversable (sub-threshold) cell
        for wx, wy in path:
            ix, iy = grid.world_to_index(wx, wy)
            self.assertIsNotNone(ix)
            self.assertLess(float(grid.gmap[ix, iy]), 0.5)
        # the path must cross the wall column through the gap row (iy=8)
        crossed = [(ix, iy) for ix, iy in
                   (grid.world_to_index(wx, wy) for wx, wy in path) if ix == 5]
        self.assertTrue(crossed)
        self.assertTrue(all(iy == 8 for _, iy in crossed))

    def test_inflated_obstacle_avoided(self):
        # Lethal core with a linearly-decaying inflation ring (ESDF-like).
        cost = np.zeros((15, 15), np.float32)
        cx, cy = 7, 7
        for ix in range(15):
            for iy in range(15):
                d = np.hypot(ix - cx, iy - cy)
                if d <= 1.0:
                    cost[ix, iy] = 1.0
                elif d < 4.0:
                    cost[ix, iy] = float(np.clip((4.0 - d) / 3.0, 0.0, 1.0))
        grid = self._grid(cost)
        path = self.planner.plan(grid, (0.0, 0.0), (14.0, 14.0))
        self.assertIsNotNone(path)
        for wx, wy in path:
            ix, iy = grid.world_to_index(wx, wy)
            self.assertLess(float(grid.gmap[ix, iy]), 0.5,
                            f'path crosses inflated cell ({ix},{iy})')


if __name__ == '__main__':
    unittest.main()
