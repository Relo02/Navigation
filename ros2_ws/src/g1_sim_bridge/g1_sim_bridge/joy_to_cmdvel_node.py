#!/usr/bin/env python3
"""Map a Unitree G1 gamepad (sensor_msgs/Joy) -> /cmd_vel (geometry_msgs/Twist).

The G1 official controller presents to the PC as a standard Xbox-style gamepad
(the same layout RoboJuDo's joystick reader assumes): left stick = translate,
right stick X = turn. ROS 2's `joy_node` publishes its axes on /joy; this node
turns them into the (vx, vy, yaw) velocity command the AMO gait expects, on the
single canonical topic /cmd_vel that drives BOTH worlds:

    joy_node -> /joy -> THIS -> /cmd_vel ->  sim : g1_sim_scene --stabilize
                                             real: cmd_vel_to_amo_node -> ws:8766

Default axis layout (SDL/Xbox, matches RoboJuDo joystick.py):
    LeftY  (axis 1) -> vx  (forward, stick up = +)
    LeftX  (axis 0) -> vy  (lateral, stick left = +, REP-103)
    RightX (axis 3) -> yaw (turn,    stick left = +ccw)
SDL reports stick-up / stick-left as NEGATIVE, so those axes are inverted to get
intuitive signs. Flip any *_invert / scale param if your pad differs.

A deadman button can be required (enable_button >= 0): the command is only
non-zero while that button is held — recommended on the real robot.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy


class JoyToCmdVel(Node):
    def __init__(self):
        super().__init__("joy_to_cmdvel")
        # axis indices (SDL/Xbox layout)
        self.declare_parameter("axis_vx", 1)      # LeftY
        self.declare_parameter("axis_vy", 0)      # LeftX
        self.declare_parameter("axis_yaw", 3)     # RightX
        # velocity caps (m/s, m/s, rad/s) -- match amo max_forward_vel/max_yaw_rate
        self.declare_parameter("scale_vx", 0.8)
        self.declare_parameter("scale_vy", 0.4)
        self.declare_parameter("scale_yaw", 0.4)
        # sign: SDL up/left are negative, so invert to make up=+forward, left=+
        self.declare_parameter("invert_vx", True)
        self.declare_parameter("invert_vy", True)
        self.declare_parameter("invert_yaw", True)
        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("enable_button", -1)   # >=0 -> deadman; -1 -> always on
        self.declare_parameter("publish_rate", 30.0)  # republish latest at this Hz
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        g = self.get_parameter
        self._ax = (int(g("axis_vx").value), int(g("axis_vy").value), int(g("axis_yaw").value))
        self._sc = (float(g("scale_vx").value), float(g("scale_vy").value), float(g("scale_yaw").value))
        self._inv = (bool(g("invert_vx").value), bool(g("invert_vy").value), bool(g("invert_yaw").value))
        self._dz = float(g("deadzone").value)
        self._enable = int(g("enable_button").value)
        rate = float(g("publish_rate").value)

        self._twist = Twist()
        self._pub = self.create_publisher(Twist, g("cmd_vel_topic").value, 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.create_timer(1.0 / max(1.0, rate), lambda: self._pub.publish(self._twist))
        self.get_logger().info(
            f"joy->cmd_vel: axes(vx,vy,yaw)={self._ax} scale={self._sc} "
            f"invert={self._inv} deadman_btn={self._enable}")

    def _axis(self, joy: Joy, idx: int, scale: float, invert: bool) -> float:
        if idx < 0 or idx >= len(joy.axes):
            return 0.0
        v = float(joy.axes[idx])
        if abs(v) < self._dz:
            return 0.0
        return (-v if invert else v) * scale

    def _on_joy(self, joy: Joy):
        if self._enable >= 0 and not (
                self._enable < len(joy.buttons) and joy.buttons[self._enable]):
            self._twist = Twist()   # deadman not held -> stop
            return
        t = Twist()
        t.linear.x = self._axis(joy, self._ax[0], self._sc[0], self._inv[0])
        t.linear.y = self._axis(joy, self._ax[1], self._sc[1], self._inv[1])
        t.angular.z = self._axis(joy, self._ax[2], self._sc[2], self._inv[2])
        self._twist = t


def main(argv=None):
    rclpy.init(args=argv)
    node = JoyToCmdVel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
