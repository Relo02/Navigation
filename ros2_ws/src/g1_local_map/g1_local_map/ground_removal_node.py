#!/usr/bin/env python3
"""
Ground-removal preprocessor for the Livox MID-360 cloud.

A standalone node that sits between the Livox driver and DLIO:

    /livox/lidar (raw, PointXYZRTLT) ─► ground_removal ─► /livox/lidar_filtered ─► DLIO

It removes the dominant ground plane from each scan via RANSAC and republishes
the surviving points. Ground segmentation runs only on x/y/z, but the cloud is
filtered at the BYTE level so every original field — crucially the per-point
`timestamp` DLIO needs for deskewing (continuous-time motion correction) — is
preserved exactly.

RANSAC (vs the local map's per-cell lowest-point) because this is a single,
sparse, sensor-frame scan: a global plane fit is robust to that, whereas per-cell
segmentation needs the dense temporally-accumulated cloud the local map builds.

NOTE: feeding a ground-removed cloud into a LiDAR-inertial odometry can degrade
it (the ground is a strong pitch/roll/Z constraint). Toggle with the launch arg
`ground_removal:=false` to send DLIO the raw cloud and compare.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


def ground_mask_ransac(xyz: np.ndarray, dist_thresh: float, max_iter: int,
                       low_quantile: float, min_normal_z: float,
                       min_inlier_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Return a boolean keep-mask (True = NOT ground) for the cloud.

    Fits the dominant near-horizontal plane to the lower part of the cloud via
    RANSAC and marks its inliers as ground. Fail-safe: if no good plane is found,
    keep everything (never blank the cloud feeding DLIO).
    """
    n = xyz.shape[0]
    keep = np.ones(n, dtype=bool)
    if n < 50:
        return keep

    finite = np.isfinite(xyz).all(axis=1)
    z = xyz[:, 2]
    # Candidate ground points: the lower band by z (cloud is ~z-up after the
    # driver's mount-roll). Bias RANSAC to the floor, not high horizontal slabs.
    z_cut = np.quantile(z[finite], low_quantile)
    cand_idx = np.where(finite & (z <= z_cut))[0]
    if cand_idx.shape[0] < 30:
        return keep

    best_inliers, best_count = None, 0
    for _ in range(max_iter):
        s = cand_idx[rng.integers(0, cand_idx.shape[0], size=3)]
        p0, p1, p2 = xyz[s[0]], xyz[s[1]], xyz[s[2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-6:
            continue
        nrm = nrm / norm
        if abs(nrm[2]) < min_normal_z:   # reject non-horizontal planes (walls)
            continue
        d = -nrm.dot(p0)
        dist = np.abs(xyz @ nrm + d)
        inliers = finite & (dist < dist_thresh)
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers

    if best_inliers is None or best_count < min_inlier_frac * n:
        return keep  # no convincing ground plane → keep all
    return ~best_inliers


class GroundRemovalNode(Node):
    def __init__(self):
        super().__init__("ground_removal")
        p = self.declare_parameter
        self.input_topic = p("input_topic", "/livox/lidar").value
        self.output_topic = p("output_topic", "/livox/lidar_filtered").value
        self.dist_thresh = float(p("dist_thresh", 0.08).value)       # plane inlier band (m)
        self.max_iter = int(p("ransac_iters", 60).value)
        self.low_quantile = float(p("low_quantile", 0.5).value)      # ground-candidate z band
        self.min_normal_z = float(p("min_normal_z", 0.85).value)     # |n_z| to count as horizontal
        self.min_inlier_frac = float(p("min_inlier_frac", 0.05).value)  # min plane support to remove
        self._rng = np.random.default_rng(0)

        # RELIABLE both ways: the Livox driver publishes /livox/lidar RELIABLE and
        # DLIO subscribes the (filtered) cloud RELIABLE — a best-effort pub would be
        # refused by DLIO ("incompatible QoS, no messages sent").
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(PointCloud2, self.output_topic, qos)
        self.create_subscription(PointCloud2, self.input_topic, self._on_cloud, qos)
        self.get_logger().info(
            f"ground_removal: {self.input_topic} -> {self.output_topic} "
            f"(RANSAC dist={self.dist_thresh:.2f}m, preserves all point fields)")
        self._tick = 0

    def _on_cloud(self, msg: PointCloud2) -> None:
        # read_points (NOT read_points_numpy): PointXYZRTLT mixes field datatypes
        # (f32 xyz, u8 tag/line, f64 timestamp) which read_points_numpy rejects.
        # skip_nans=False keeps points in raw-data order so the keep-mask aligns
        # with the byte-level filter below.
        rec = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
        xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
        keep = ground_mask_ransac(
            xyz, self.dist_thresh, self.max_iter, self.low_quantile,
            self.min_normal_z, self.min_inlier_frac, self._rng)

        # Byte-level filter: keep every field (incl. per-point timestamp for deskew).
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
        kept = raw[keep]

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = int(kept.shape[0])
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * out.width
        out.is_dense = msg.is_dense
        out.data = kept.tobytes()
        self.pub.publish(out)

        self._tick += 1
        if self._tick <= 5 or self._tick % 20 == 0:
            self.get_logger().info(
                f"in={xyz.shape[0]} kept={int(keep.sum())} removed_ground={int((~keep).sum())}")


def main(args=None):
    rclpy.init(args=args)
    node = GroundRemovalNode()
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
