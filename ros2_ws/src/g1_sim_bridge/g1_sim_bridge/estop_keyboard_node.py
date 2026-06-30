#!/usr/bin/env python3
"""Keyboard e-stop for autonomous navigation.

Reads single-key commands from stdin and publishes a latched Bool on /estop that
cmd_vel_to_amo_node honours: while /estop is true the bridge forwards ZERO
velocity to the AMO gait (robot stops and holds), independent of what the MPC is
publishing. Releasing /estop resumes normal forwarding so navigation continues.

Designed to run in the FOREGROUND of the autonomy.sh terminal (the two launches
go to logfiles, this owns the keyboard):

    s + Enter   →  STOP   (engage e-stop; robot holds at zero)
    g + Enter   →  GO     (release e-stop; navigation re-enabled)
    q + Enter   →  quit this helper (engages the e-stop first, fail-safe)

The publisher is persistent and latched (TRANSIENT_LOCAL), so the command takes
effect immediately and a bridge that starts/restarts later still inherits the
current stop state.
"""
from __future__ import annotations

import sys
import time
from select import select

import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool


_BANNER = (
    "\n"
    "==================== NAV E-STOP ====================\n"
    "  s + Enter  ->  STOP  (zero velocity to the gait, hold)\n"
    "  g + Enter  ->  GO    (release e-stop, resume navigation)\n"
    "  q + Enter  ->  quit  (engages e-stop on exit)\n"
    "====================================================\n"
)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = rclpy.create_node("estop_keyboard")
    topic = node.declare_parameter("estop_topic", "/estop").value

    # Latched so the bridge inherits the current state even if it joins late.
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    pub = node.create_publisher(Bool, str(topic), qos)

    def publish(engaged: bool) -> None:
        pub.publish(Bool(data=engaged))

    # Start released so a fresh run does not surprise-stop a running mission;
    # the user presses 's' to stop. (Change to True here if you prefer the
    # mission to start latched-stopped and require an explicit 'g' to begin.)
    publish(False)
    node.get_logger().info(f"e-stop keyboard ready on {topic}")
    sys.stdout.write(_BANNER)
    sys.stdout.flush()

    engaged = False
    last_pub_t = 0.0

    try:
        while rclpy.ok():
            # Keep the latched state fresh. This makes STOP robust across DDS
            # discovery races and bridge restarts, instead of relying on one
            # stdin-triggered publish.
            now = time.monotonic()
            if engaged and now - last_pub_t >= 0.1:
                publish(True)
                last_pub_t = now

            ready, _, _ = select([sys.stdin], [], [], 0.1)
            if not ready:
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            line = sys.stdin.readline()
            if line == "":
                publish(True)
                break

            key = line.strip().lower()
            if key in ("s", "stop"):
                engaged = True
                publish(True)
                last_pub_t = time.monotonic()
                sys.stdout.write(">> E-STOP ENGAGED — robot holding at zero. 'g' to resume.\n")
            elif key in ("g", "go", "r", "resume"):
                engaged = False
                publish(False)
                last_pub_t = time.monotonic()
                sys.stdout.write(">> RESUMED — navigation re-enabled.\n")
            elif key in ("q", "quit", "exit"):
                engaged = True
                publish(True)   # fail-safe: stop on the way out
                sys.stdout.write(">> quitting e-stop helper (e-stop engaged).\n")
                break
            elif key == "":
                continue
            else:
                sys.stdout.write("   (use: s=stop, g=go, q=quit)\n")
            sys.stdout.flush()
            rclpy.spin_once(node, timeout_sec=0.0)
    except (EOFError, KeyboardInterrupt):
        publish(True)   # fail-safe: stdin closed / Ctrl-C → engage stop
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
