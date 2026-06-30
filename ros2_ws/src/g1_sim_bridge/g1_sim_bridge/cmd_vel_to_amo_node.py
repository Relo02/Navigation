#!/usr/bin/env python3
"""Bridge /cmd_vel (geometry_msgs/Twist) -> the real-robot AMO velocity command.

The AMO deployment driver (amo/amo_inference.py) is NOT a ROS 2 process -- it
talks to the robot over CycloneDDS and accepts velocity commands on a WebSocket
server (port 8766, same interface the MPC planner uses). This node lets the
Unitree G1 gamepad on the ROS 2 side drive the real robot: it subscribes to
/cmd_vel and forwards (vx, vy, yaw) as JSON to that WebSocket.

So the SAME /cmd_vel topic drives both worlds (sim can use the keyboard since the
Unitree pad pairs with the real robot):
    sim  : keyboard/joy -> /cmd_vel -> g1_sim_scene.py --stabilize (Isaac rclpy)
    real : joy_node -> joy_to_cmdvel -> /cmd_vel -> THIS -> ws://amo:8766 -> amo_inference

Run amo_inference with command.source=websocket, then:

    ros2 run g1_sim_bridge cmd_vel_to_amo_node --ros-args \
        -p amo_host:=127.0.0.1 -p amo_port:=8766

A minimal stdlib WebSocket client is used (send-only) so no extra Python package
is needed in the ROS 2 container.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool


class _WSClient:
    """Minimal send-only RFC6455 WebSocket client (text frames, masked)."""

    def __init__(self, host: str, port: int, path: str = "/"):
        self.host, self.port, self.path = host, int(port), path
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=2.0)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        s.sendall(req.encode())
        resp = s.recv(1024).decode("latin-1", "ignore")
        if "101" not in resp.split("\r\n", 1)[0]:
            s.close()
            raise ConnectionError(f"WS handshake failed: {resp[:80]!r}")
        self._sock = s

    @staticmethod
    def _frame(payload: bytes) -> bytes:
        n = len(payload)
        header = bytes([0x81])                      # FIN + text opcode
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return header + mask + masked

    def send_json(self, obj: dict) -> bool:
        with self._lock:
            try:
                if self._sock is None:
                    self._connect()
                self._sock.sendall(self._frame(json.dumps(obj).encode()))
                return True
            except (OSError, ConnectionError):
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                self._sock = None
                return False

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None


class CmdVelToAmo(Node):
    def __init__(self):
        super().__init__("cmd_vel_to_amo")
        self.declare_parameter("amo_host", "127.0.0.1")
        self.declare_parameter("amo_port", 8766)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("max_forward_vel", 0.8)
        self.declare_parameter("max_lateral_vel", 0.4)
        self.declare_parameter("max_yaw_rate", 0.4)
        # Fail-safe watchdog: if no command arrives for this long, forward ZERO
        # velocity to the gait instead of re-sending the last command forever.
        # Without this the robot coasts on a stale command whenever the MPC
        # hiccups, dies, or is killed — unacceptable on real hardware.
        self.declare_parameter("cmd_timeout_sec", 0.5)
        # Manual e-stop latch: while /estop is true, ZERO velocity is forwarded
        # to the gait regardless of /mpc/cmd_vel, so the robot stops and holds.
        # Releasing /estop resumes normal forwarding (navigation re-enabled).
        self.declare_parameter("estop_topic", "/estop")

        host = self.get_parameter("amo_host").value
        port = self.get_parameter("amo_port").value
        topic = self.get_parameter("cmd_vel_topic").value
        rate = float(self.get_parameter("rate_hz").value)
        self._cmd_timeout = float(self.get_parameter("cmd_timeout_sec").value)
        estop_topic = str(self.get_parameter("estop_topic").value)
        self._vmax = (float(self.get_parameter("max_forward_vel").value),
                      float(self.get_parameter("max_lateral_vel").value),
                      float(self.get_parameter("max_yaw_rate").value))

        self._cmd = (0.0, 0.0, 0.0)
        self._last_cmd_t = time.monotonic()
        self._timed_out = False
        self._estop = False
        self._ws = _WSClient(host, port)
        self._connected = False

        self.create_subscription(Twist, topic, self._on_twist, 10)
        # E-stop latched (TRANSIENT_LOCAL) so the bridge picks up the current
        # stop state even if it (re)starts after the e-stop was engaged.
        estop_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Bool, estop_topic, self._on_estop, estop_qos)
        self.create_timer(1.0 / max(1.0, rate), self._tick)
        self.get_logger().info(
            f"bridging {topic} -> ws://{host}:{port} at {rate:.0f} Hz "
            f"(caps vx={self._vmax[0]} vy={self._vmax[1]} yaw={self._vmax[2]}; "
            f"watchdog={self._cmd_timeout:.2f}s; estop_topic={estop_topic})")

    @staticmethod
    def _clip(v, lim):
        return max(-lim, min(lim, float(v)))

    def _on_twist(self, msg: Twist):
        self._cmd = (
            self._clip(msg.linear.x, self._vmax[0]),
            self._clip(msg.linear.y, self._vmax[1]),
            self._clip(msg.angular.z, self._vmax[2]),
        )
        self._last_cmd_t = time.monotonic()
        if self._timed_out:
            self._timed_out = False
            self.get_logger().info("command stream resumed")

    def _on_estop(self, msg: Bool):
        engaged = bool(msg.data)
        if engaged != self._estop:
            self._estop = engaged
            if engaged:
                self.get_logger().warn("E-STOP ENGAGED — forwarding ZERO velocity")
                self._cmd = (0.0, 0.0, 0.0)
                self._last_cmd_t = time.monotonic()
                self._ws.send_json({"vx": 0.0, "vy": 0.0, "yaw": 0.0})
            else:
                self.get_logger().info("E-STOP RELEASED — navigation re-enabled")

    def _tick(self):
        # Manual e-stop has top priority: hold the robot at zero regardless of
        # what the MPC is publishing, until the latch is released.
        if self._estop:
            self._ws.send_json({"vx": 0.0, "vy": 0.0, "yaw": 0.0})
            return
        # Watchdog: stale command stream → force zero velocity (fail-safe stop).
        if time.monotonic() - self._last_cmd_t > self._cmd_timeout:
            self._cmd = (0.0, 0.0, 0.0)
            if not self._timed_out:
                self._timed_out = True
                self.get_logger().warn(
                    f"no cmd_vel for >{self._cmd_timeout:.2f}s — sending ZERO velocity")
        vx, vy, yaw = self._cmd
        ok = self._ws.send_json({"vx": vx, "vy": vy, "yaw": yaw})
        if ok and not self._connected:
            self._connected = True
            self.get_logger().info("connected to AMO WebSocket")
        elif not ok and self._connected:
            self._connected = False
            self.get_logger().warn("AMO WebSocket dropped; retrying")

    def destroy_node(self):
        self._ws.close()
        super().destroy_node()


def main(argv=None):
    rclpy.init(args=argv)
    node = CmdVelToAmo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
