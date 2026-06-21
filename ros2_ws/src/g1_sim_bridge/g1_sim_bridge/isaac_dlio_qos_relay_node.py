"""Isaac Sim -> DLIO QoS relay (simulation only).

DLIO (direct_lidar_inertial_odometry) consumes a plain ``sensor_msgs/PointCloud2``
and a ``sensor_msgs/Imu`` directly -- no Livox ``CustomMsg`` needed. The only
mismatch with Isaac Sim's ROS 2 bridge is QoS:

  * DLIO's cloud subscriber (``pointcloud``) is RELIABLE, but Isaac publishes
    ``/livox/lidar`` BEST_EFFORT -> a reliable subscriber will not match a
    best-effort publisher, so DLIO would see nothing.
  * DLIO's IMU subscriber uses ``SensorDataQoS`` (BEST_EFFORT), which already
    matches Isaac's BEST_EFFORT ``/livox/imu_raw`` -- but we still republish it
    on ``/livox/imu`` (RELIABLE) so sim and real use the *same* topic name and
    so any reliable consumer (RViz, debuggers) can also subscribe.

So this node is a thin QoS bridge, sim only:

  * Cloud:  /livox/lidar     (PointCloud2, BEST_EFFORT) -> /livox/lidar_reliable
            (PointCloud2, RELIABLE), contents unchanged (optionally decimated).
  * IMU:    /livox/imu_raw   (Imu, BEST_EFFORT)         -> /livox/imu (RELIABLE),
            contents unchanged.

On the real robot this node is NOT used -- the stock Livox SDK driver emits a
RELIABLE-friendly ``/livox/lidar`` PointCloud2 (xfer_format=0) + ``/livox/imu``
directly, which DLIO auto-detects as ``SensorType::LIVOX`` (per-point timestamp)
and deskews.

Replaces the FAST-LIO-era ``isaac_livox_custom_adapter_node`` (PointCloud2 ->
CustomMsg), which DLIO makes unnecessary.

Parameters
----------
input_topic     Isaac cloud (PointCloud2, BEST_EFFORT).  Default: /livox/lidar
output_topic    RELIABLE cloud for DLIO.                 Default: /livox/lidar_reliable
imu_in_topic    Isaac IMU (BEST_EFFORT).                 Default: /livox/imu_raw
imu_out_topic   RELIABLE IMU for DLIO.                   Default: /livox/imu
stride          Keep every Nth cloud point (1 = all).    Default: 1
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2


class IsaacDlioQosRelay(Node):

    def __init__(self):
        super().__init__('isaac_dlio_qos_relay')

        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('output_topic', '/livox/lidar_reliable')
        self.declare_parameter('imu_in_topic', '/livox/imu_raw')
        self.declare_parameter('imu_out_topic', '/livox/imu')
        self.declare_parameter('stride', 1)

        self._stride = max(1, int(self.get_parameter('stride').value))
        out_topic = str(self.get_parameter('output_topic').value)
        imu_out_topic = str(self.get_parameter('imu_out_topic').value)

        # Isaac publishes the cloud/IMU BEST_EFFORT; subscribe to match.
        be_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5,
        )
        # Republish RELIABLE so DLIO's reliable cloud subscriber matches (and a
        # reliable publisher still satisfies DLIO's BEST_EFFORT IMU subscriber).
        rel_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )

        self._cloud_pub = self.create_publisher(PointCloud2, out_topic, rel_qos)
        self._imu_pub = self.create_publisher(Imu, imu_out_topic, rel_qos)

        self.create_subscription(
            PointCloud2, str(self.get_parameter('input_topic').value),
            self._cloud_cb, be_qos)
        self.create_subscription(
            Imu, str(self.get_parameter('imu_in_topic').value),
            self._imu_cb, be_qos)

        self.get_logger().info(
            f'Isaac->DLIO QoS relay: PointCloud2 -> {out_topic!r} (RELIABLE), '
            f'IMU -> {imu_out_topic!r} (RELIABLE); stride={self._stride}'
        )

    def _imu_cb(self, msg: Imu) -> None:
        # Pure QoS bridge (BEST_EFFORT -> RELIABLE); contents unchanged.
        self._imu_pub.publish(msg)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        # Always rebuild a clean, finite XYZ cloud. Isaac's RTX lidar keeps
        # no-return rays as NaN/Inf points; forwarding them verbatim makes DLIO's
        # removeNaN + crop empty the scan (deskewed points: 0) and then heap-crash
        # on the empty cloud ('free(): invalid next size'). Stripping non-finite
        # points here gives DLIO the same valid-returns-only cloud the real Livox
        # driver emits (XYZ is all the sim path needs: no per-point time -> DLIO
        # auto-detects UNKNOWN and skips deskew).
        try:
            raw = point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f'cloud parse error: {exc}', throttle_duration_sec=5.0)
            return
        pts = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32)
        # skip_nans drops NaN; also drop Inf (no-return rays at +inf range).
        pts = pts[np.isfinite(pts).all(axis=1)]
        if self._stride > 1:
            pts = pts[::self._stride]
        if pts.shape[0] == 0:
            # Never forward an empty cloud -- DLIO crashes on it. Just wait.
            self.get_logger().warning(
                'cloud has no finite points -- not forwarding (lidar sees nothing?)',
                throttle_duration_sec=5.0)
            return
        self._cloud_pub.publish(point_cloud2.create_cloud_xyz32(msg.header, pts))


def main(args=None):
    rclpy.init(args=args)
    node = IsaacDlioQosRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
