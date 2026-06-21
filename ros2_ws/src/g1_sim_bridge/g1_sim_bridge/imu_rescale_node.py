#!/usr/bin/env python3
"""
Livox IMU unit rescale relay (g -> m/s^2) for the real robot.

The Livox MID-360's built-in IMU reports **linear acceleration in units of g**
(so ~1.0 at rest), but DLIO and the ROS sensor_msgs/Imu convention expect
**m/s^2** (~9.81 at rest). Feeding the raw g-unit values straight into DLIO makes
it subtract 9.81 m/s^2 of gravity from a ~1.0 reading: the ~8.8 m/s^2 residual
saturates the accel-bias estimate at abias_max and the vertical velocity/position
diverge until DLIO crashes.

This node subscribes to the raw IMU, multiplies linear_acceleration by g, and
republishes an otherwise-identical message. Angular velocity (already rad/s) and
orientation are passed through untouched, so DLIO calibrates and runs on correct
m/s^2 data with its IMU calibration left ON.

    /livox/imu  (g)  ──►  rescale ×9.80665  ──►  /livox/imu_ms2  (m/s^2)  ──► DLIO
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu


class ImuRescaleNode(Node):
    def __init__(self):
        super().__init__("imu_rescale")
        self.input_topic = self.declare_parameter("input_topic", "/livox/imu").value
        self.output_topic = self.declare_parameter("output_topic", "/livox/imu_ms2").value
        # Standard gravity; the factor that converts Livox accel from g to m/s^2.
        self.scale = float(self.declare_parameter("accel_scale", 9.80665).value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=200,
        )
        self.pub = self.create_publisher(Imu, self.output_topic, qos)
        self.create_subscription(Imu, self.input_topic, self._on_imu, qos)
        self.get_logger().info(
            f"imu_rescale: {self.input_topic} (g) -> {self.output_topic} "
            f"(m/s^2, accel x{self.scale:.5f})"
        )

    def _on_imu(self, msg: Imu) -> None:
        msg.linear_acceleration.x *= self.scale
        msg.linear_acceleration.y *= self.scale
        msg.linear_acceleration.z *= self.scale
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuRescaleNode()
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
