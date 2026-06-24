"""
External costmap adapter for the A* planner.

In ``costmap_backend:=external_grid`` mode the planner does not build its own
Gaussian grid from the live cloud — it plans directly on a pre-computed cost
grid supplied by another node (the nvblox ESDF adapter, ``/nvblox_g1/costmap_raw``).
``ExternalGridMap`` exposes exactly the subset of the ``FixedGaussianGridMap``
interface that ``AStarPlanner`` touches:

    attributes : gmap, hmap, cells, reso, minx, miny, half_width, xw, yw
    methods    : world_to_index(x, y), index_to_world(ix, iy)

so the planner is agnostic to which backend produced the grid.

``hmap`` is ``None`` here: an ESDF slice carries no 2.5D max-z, so the
step-over rule is disabled and a lethal cell is a hard block. That is the
intended behaviour — the ESDF adapter already chose a humanoid-appropriate
slice height band, so anything it reports as an obstacle should block.

Grid payload contract (shared with a_star_node ``/a_star/grid_raw`` and
g1_mapping.esdf_costmap.build_costmap_raw):

    data[0:4]  = [minx, miny, resolution, cells]
    data[4: ]  = cost grid, C-order flatten of a (cells, cells) array indexed
                 [ix, iy]; i.e. value at (ix, iy) = data[4 + ix*cells + iy].
"""

from __future__ import annotations

import numpy as np


class ExternalGridMap:
    """A read-only cost grid that quacks like FixedGaussianGridMap for A*."""

    def __init__(self, gmap: np.ndarray, minx: float, miny: float, reso: float):
        if gmap.ndim != 2 or gmap.shape[0] != gmap.shape[1]:
            raise ValueError(f'gmap must be square 2D, got {gmap.shape}')
        self.gmap: np.ndarray = gmap.astype(np.float32, copy=False)
        self.hmap = None  # no 2.5D layer from an ESDF slice → no step-over
        self.cells = int(gmap.shape[0])
        self.reso = float(reso)
        self.minx = float(minx)
        self.miny = float(miny)
        self.half_width = self.cells * self.reso * 0.5
        # Aliases some consumers expect (mujoco_sim convention).
        self.xw = self.cells
        self.yw = self.cells
        self.xyreso = self.reso

    def world_to_index(self, x: float, y: float):
        """World coords → (ix, iy) inside [0, cells), or (None, None)."""
        ix = int((x - self.minx) / self.reso)
        iy = int((y - self.miny) / self.reso)
        if 0 <= ix < self.cells and 0 <= iy < self.cells:
            return ix, iy
        return None, None

    def index_to_world(self, ix: int, iy: int):
        """(ix, iy) → world coords at the cell corner (matches the Gaussian grid)."""
        return (ix * self.reso + self.minx, iy * self.reso + self.miny)

    @property
    def maxx(self) -> float:
        return self.minx + self.cells * self.reso

    @property
    def maxy(self) -> float:
        return self.miny + self.cells * self.reso


def parse_costmap_raw(data) -> ExternalGridMap:
    """Decode a ``/nvblox_g1/costmap_raw`` Float32MultiArray payload.

    Parameters
    ----------
    data : sequence of float — ``[minx, miny, reso, cells] + flattened_cost``.

    Returns
    -------
    ExternalGridMap

    Raises
    ------
    ValueError if the payload is too short or its length is inconsistent with
    the declared ``cells`` (a malformed / truncated message).
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.size < 4:
        raise ValueError(f'costmap_raw too short: {arr.size} < 4 (no meta)')
    minx = float(arr[0])
    miny = float(arr[1])
    reso = float(arr[2])
    cells = int(round(float(arr[3])))
    if cells <= 0:
        raise ValueError(f'costmap_raw declares non-positive cells={cells}')
    expected = 4 + cells * cells
    if arr.size != expected:
        raise ValueError(
            f'costmap_raw length {arr.size} != 4 + cells^2 ({expected}) '
            f'for cells={cells}'
        )
    gmap = arr[4:].reshape(cells, cells)  # C-order → [ix, iy]
    return ExternalGridMap(gmap, minx, miny, reso)
