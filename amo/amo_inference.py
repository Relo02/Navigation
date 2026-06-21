#!/usr/bin/env python3
"""
AMO policy inference for the Unitree G1 — with the joint-smoothing filter.

This is the single driver that puts the **AMO code** (RoboJuDo AMOPolicy +
UnitreeEnv, in amo_policy.py) and the **smoothing filter** (joint_filters.py)
together. It runs the 50 Hz control loop:

    measured_q ─► policy_target(cmd) ─► JointSmoother(A→D→C) ─► env.step

so the joints never snap to the policy reference at activation — they S-curve
blend from the captured posture while PD gains ramp soft→full, and an always-on
slew filter keeps commanded joints converging smoothly instead of tracking the
policy instantaneously. See docs/amo_inference_plan.md.

Run (inside the amo_policy container)::

    python /workspace/amo/amo_inference.py --config /workspace/config/amo_g1.yaml
    python /workspace/amo/amo_inference.py --observe_only          # dry run, no motor cmds
    python /workspace/amo/amo_inference.py --net_if eth0 --vx 0.3  # walk forward

Config file values are defaults; CLI flags and env vars override them.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Make sibling modules importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from activation_utils import command_ramp_factor  # noqa: E402
from amo_policy import AmoDeployment, gravity_tilt_angle  # noqa: E402
from joint_filters import JointSmoother  # noqa: E402

logger = logging.getLogger("amo.inference")


# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "control": {"freq_hz": 50, "net_if": "eth0", "device": "cpu", "observe_only": False},
    "model": {"robojudo_root": None, "amo_pd_gains": True},
    "activation": {
        "blend_s": 5.0, "gain_ramp_s": 2.0, "kp_scale_start": 0.15,
        "kd_scale_start": 0.50, "stabilize_s": 10.0,
        "startup_command_ramp_s": 3.0, "safety_tilt_rad": 0.6, "state_timeout_s": 0.5,
    },
    "filter": {
        "kind": "critdamp", "tau": 0.08, "wn": 12.0, "clamp_delta_rad": 0.05,
        "release_s": None, "release_ramp_s": 1.0,
    },
    "command": {
        "source": "zero", "constant": [0.0, 0.0, 0.0],
        "max_forward_vel": 0.8, "max_yaw_rate": 0.4, "websocket_port": 8766,
    },
}


def _deep_update(base: dict, extra: dict) -> dict:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None) -> dict:
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if path:
        p = Path(path)
        if not p.exists():
            logger.warning("config %s not found; using defaults", p)
        else:
            import yaml
            with open(p) as fh:
                _deep_update(cfg, yaml.safe_load(fh) or {})
            logger.info("loaded config from %s", p)
    return cfg


# ── velocity command sources ───────────────────────────────────────────────────
class CommandSource:
    """Thread-safe holder for the (vx, vy, yaw) velocity command."""

    def __init__(self, initial=(0.0, 0.0, 0.0)):
        self._lock = threading.Lock()
        self._cmd = np.asarray(initial, dtype=np.float32)

    def get(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()

    def set(self, vx, vy, yaw) -> None:
        with self._lock:
            self._cmd = np.asarray([vx, vy, yaw], dtype=np.float32)


def start_websocket_source(source: CommandSource, port: int) -> None:
    """Best-effort WebSocket server accepting {"vx","vy","yaw"} JSON messages.

    Mirrors the /velocity_target interface the MPC planner uses. Runs in a
    daemon thread; failures degrade to the last-held command rather than
    crashing the control loop.
    """
    import asyncio
    import json

    async def _handler(ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
                source.set(
                    float(msg.get("vx", 0.0)),
                    float(msg.get("vy", 0.0)),
                    float(msg.get("yaw", msg.get("yaw_rate", 0.0))),
                )
            except (ValueError, TypeError) as exc:
                logger.warning("bad WS command %r: %s", raw, exc)

    def _run():
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed; command source falls back to zero")
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve():
            async with websockets.serve(_handler, "0.0.0.0", port):
                logger.info("WS command server listening on :%d", port)
                await asyncio.Future()

        try:
            loop.run_until_complete(_serve())
        except Exception as exc:  # noqa: BLE001
            logger.error("WS command server stopped: %s", exc)

    threading.Thread(target=_run, daemon=True, name="ws-command").start()


def build_command_source(cmd_cfg: dict, cli_const) -> CommandSource:
    source = CommandSource()
    kind = (cmd_cfg.get("source") or "zero").lower()
    if cli_const is not None:
        source.set(*cli_const)
        logger.info("command source: constant (CLI) %s", cli_const)
    elif kind == "constant":
        source.set(*cmd_cfg.get("constant", [0.0, 0.0, 0.0]))
        logger.info("command source: constant %s", cmd_cfg.get("constant"))
    elif kind == "websocket":
        start_websocket_source(source, int(cmd_cfg.get("websocket_port", 8766)))
        logger.info("command source: websocket :%s", cmd_cfg.get("websocket_port"))
    else:
        logger.info("command source: zero (stand in place)")
    return source


# ── control loop ────────────────────────────────────────────────────────────────
class Pacer:
    """Fixed-rate loop pacer with drift correction."""

    def __init__(self, dt: float):
        self.dt = dt
        self._next = time.perf_counter()

    def wait(self) -> None:
        self._next += self.dt
        sleep_t = self._next - time.perf_counter()
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            self._next = time.perf_counter()


def run(cfg: dict) -> int:
    ctrl = cfg["control"]
    act = cfg["activation"]
    filt = cfg["filter"]
    cmd_cfg = cfg["command"]

    dt = 1.0 / float(ctrl["freq_hz"])
    observe_only = bool(ctrl["observe_only"])

    dep = AmoDeployment(
        robojudo_root=cfg["model"].get("robojudo_root"),
        net_if=ctrl["net_if"],
        device=ctrl["device"],
        observe_only=observe_only,
        max_forward_vel=cmd_cfg["max_forward_vel"],
        max_yaw_rate=cmd_cfg["max_yaw_rate"],
        amo_pd_gains=bool(cfg["model"].get("amo_pd_gains", True)),
    )
    dep.setup()
    dep.wait_until_ready()
    dt = dep.control_dt  # honour the policy's native control frequency

    # Filtering (clamp + running filter) holds only while the joints reach the
    # policy reference, then releases for full reactivity. Default the release
    # point to the end of the pose blend when not set explicitly.
    release_s = filt.get("release_s")
    if release_s is None:
        release_s = act["blend_s"]

    smoother = JointSmoother(
        dt,
        blend_s=act["blend_s"],
        gain_ramp_s=act["gain_ramp_s"],
        kp_scale_start=act["kp_scale_start"],
        kd_scale_start=act["kd_scale_start"],
        clamp_delta=filt["clamp_delta_rad"],
        filter_kind=filt["kind"],
        filter_tau=filt["tau"],
        filter_wn=filt["wn"],
        release_s=float(release_s),
        release_ramp_s=float(filt.get("release_ramp_s", 0.0)),
    )

    command_source = build_command_source(cmd_cfg, cfg.get("_cli_const"))

    # Seed every smoothing layer at the current measured posture so the first
    # commanded delta is ~0 — the core anti-snap guarantee.
    dep.update()
    smoother.reset(dep.dof_pos)

    # Command stays zero through the blend + gain ramp + stabilize window; the
    # live velocity command then eases in over startup_command_ramp_s.
    command_hold_s = act["blend_s"] + act["stabilize_s"]
    startup_ramp_s = act["startup_command_ramp_s"]
    safety_tilt = act["safety_tilt_rad"]
    state_timeout = act["state_timeout_s"]

    logger.info(
        "starting AMO control loop @ %.1f Hz | filter=%s | blend=%.1fs gain_ramp=%.1fs "
        "stabilize=%.1fs | release=%.1fs(+%.1fs ramp) | observe_only=%s",
        1.0 / dt, filt["kind"], act["blend_s"], act["gain_ramp_s"],
        act["stabilize_s"], smoother.release_s, smoother.release_ramp_s, observe_only,
    )

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        logger.info("signal %d received; damping motors and exiting", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    pacer = Pacer(dt)
    last_state_t = time.perf_counter()
    tick = 0
    try:
        while not stop["flag"]:
            dep.update()

            # ── safety guard ──────────────────────────────────────────────
            now = time.perf_counter()
            if dep.env.low_state is not None:
                last_state_t = now
            elif (now - last_state_t) > state_timeout:
                logger.error("LowState stale for %.2fs; aborting", now - last_state_t)
                break
            tilt = gravity_tilt_angle(dep.base_quat)
            if tilt > safety_tilt:
                logger.error("tilt %.2f rad > %.2f rad; aborting", tilt, safety_tilt)
                break

            # ── velocity command (held zero during startup, then ramped in) ─
            elapsed = smoother.elapsed
            if elapsed < command_hold_s:
                command3 = np.zeros(3, dtype=np.float32)
            else:
                live = command_source.get()
                ramp = command_ramp_factor(elapsed - command_hold_s, startup_ramp_s)
                command3 = (live * ramp).astype(np.float32)

            # ── AMO inference → smoothing → motor command ──────────────────
            raw_target = dep.policy_target(command3)
            cmd_q = smoother.step(raw_target)
            s_kp, s_kd = smoother.gain_scales()
            dep.command(cmd_q, s_kp, s_kd)

            if tick % int(round(1.0 / dt)) == 0:  # ~1 Hz status
                logger.info(
                    "t=%5.1fs phase=%-9s filt=%.2f gains=(%.2f,%.2f) cmd=(%.2f,%.2f,%.2f) "
                    "max|Δq|=%.3f tilt=%.2f",
                    elapsed,
                    "blend" if smoother.blending else
                    ("stabilize" if elapsed < command_hold_s else "walk"),
                    1.0 - smoother.release_factor,
                    s_kp, s_kd, command3[0], command3[1], command3[2],
                    float(np.max(np.abs(cmd_q - dep.dof_pos))), tilt,
                )
            tick += 1
            pacer.wait()
    finally:
        logger.info("shutting down (damping motors)")
        dep.shutdown()
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AMO policy inference with joint smoothing.")
    p.add_argument("--config", default=os.environ.get("AMO_CONFIG_PATH"),
                   help="YAML config path (env: AMO_CONFIG_PATH).")
    p.add_argument("--net_if", default=None, help="CycloneDDS NIC to the robot.")
    p.add_argument("--device", default=None, help="torch device (cpu/cuda).")
    p.add_argument("--observe_only", action="store_true", default=None,
                   help="Subscribe to DDS but never publish motor commands.")
    p.add_argument("--robojudo_root", default=None, help="Override RoboJuDo root.")
    p.add_argument("--filter", dest="filter_kind", default=None,
                   choices=["none", "ewma", "critdamp"], help="Layer-D filter type.")
    p.add_argument("--vx", type=float, default=None, help="Constant forward velocity.")
    p.add_argument("--vy", type=float, default=None, help="Constant lateral velocity.")
    p.add_argument("--yaw", type=float, default=None, help="Constant yaw rate.")
    p.add_argument("--command_source", default=None,
                   choices=["zero", "constant", "websocket"],
                   help="Override command.source (websocket = joystick via cmd_vel_to_amo).")
    return p.parse_args(argv)


def apply_overrides(cfg: dict, a: argparse.Namespace) -> dict:
    # Environment overrides (lower precedence than CLI).
    if os.environ.get("UNITREE_NET_IFACE"):
        cfg["control"]["net_if"] = os.environ["UNITREE_NET_IFACE"]
    if os.environ.get("ROBOJUDO_ROOT"):
        cfg["model"]["robojudo_root"] = os.environ["ROBOJUDO_ROOT"]

    # CLI overrides.
    if a.net_if is not None:
        cfg["control"]["net_if"] = a.net_if
    if a.device is not None:
        cfg["control"]["device"] = a.device
    if a.observe_only:
        cfg["control"]["observe_only"] = True
    if a.robojudo_root is not None:
        cfg["model"]["robojudo_root"] = a.robojudo_root
    if a.filter_kind is not None:
        cfg["filter"]["kind"] = a.filter_kind
    if a.command_source is not None:
        cfg["command"]["source"] = a.command_source

    if any(v is not None for v in (a.vx, a.vy, a.yaw)):
        cfg["_cli_const"] = (a.vx or 0.0, a.vy or 0.0, a.yaw or 0.0)
    return cfg


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    a = parse_args(argv)
    cfg = apply_overrides(load_config(a.config), a)
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
