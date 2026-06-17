"""
Joint-target smoothing for the AMO policy.

Implements layer D of docs/amo_inference_plan.md (the always-on running filter)
and the ``JointSmoother`` that composes the full smoothing stack:

    raw target_q ─► (A) startup pose blend ─► (D) slew/low-pass filter
                 ─► (C) per-tick rate clamp ─► commanded_q

plus the layer-B PD-gain ramp scales (exposed via ``gain_scales``). The PD
gains themselves are applied by the caller (``env.set_gains``).

Pure numpy — no robojudo / torch / DDS imports — so it unit-tests without a
robot (tests/test_joint_filters.py).
"""

from __future__ import annotations

import numpy as np

from activation_utils import blend_pose, clamp_step_delta, gain_ramp_scale


class EWMAFilter:
    """First-order exponential low-pass: ``y += a*(target - y)``.

    ``tau`` is the smoothing time constant in seconds; larger ``tau`` → smoother
    but laggier. With ``tau <= 0`` the filter is a pass-through (alpha = 1).
    """

    def __init__(self, tau: float, dt: float):
        self.tau = float(tau)
        self.dt = float(dt)
        self._y: np.ndarray | None = None

    @property
    def alpha(self) -> float:
        if self.tau <= 0.0:
            return 1.0
        return self.dt / (self.tau + self.dt)

    def reset(self, y0) -> None:
        self._y = np.asarray(y0, dtype=np.float32).copy()

    def step(self, target) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        if self._y is None:
            self._y = target.copy()
        a = self.alpha
        self._y = ((1.0 - a) * self._y + a * target).astype(np.float32)
        return self._y.copy()


class CriticallyDampedFilter:
    """Critically-damped (zeta=1) second-order tracking filter.

    Tracks position *and* velocity so there is no commanded-velocity
    discontinuity and no overshoot — the well-behaved way to follow a moving
    reference. ``wn`` is the natural frequency in rad/s (bandwidth); larger ``wn``
    tracks faster with less smoothing. Integrated semi-implicitly (update qd
    first, then q) for stability at control-loop rates.
    """

    def __init__(self, wn: float, dt: float):
        self.wn = float(wn)
        self.dt = float(dt)
        self._y: np.ndarray | None = None
        self._yd: np.ndarray | None = None

    def reset(self, y0, yd0=None) -> None:
        self._y = np.asarray(y0, dtype=np.float32).copy()
        if yd0 is None:
            self._yd = np.zeros_like(self._y)
        else:
            self._yd = np.asarray(yd0, dtype=np.float32).copy()

    def step(self, target) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        if self._y is None:
            self.reset(target)
        # critically-damped 2nd-order: q'' = wn^2 (target - q) - 2 wn q'
        accel = self.wn * self.wn * (target - self._y) - 2.0 * self.wn * self._yd
        self._yd = (self._yd + accel * self.dt).astype(np.float32)
        self._y = (self._y + self._yd * self.dt).astype(np.float32)
        return self._y.copy()


def make_filter(kind: str, dt: float, *, tau: float = 0.08, wn: float = 12.0):
    """Construct a layer-D filter. ``kind`` ∈ {``none``, ``ewma``, ``critdamp``}."""
    kind = (kind or "none").lower()
    if kind == "none":
        return None
    if kind == "ewma":
        return EWMAFilter(tau=tau, dt=dt)
    if kind == "critdamp":
        return CriticallyDampedFilter(wn=wn, dt=dt)
    raise ValueError(f"unknown joint filter kind: {kind!r}")


class JointSmoother:
    """Full A→D→C smoothing stack on the path from policy output to motor cmd.

    Per tick the caller does::

        smoother.reset(measured_q)          # once, at hand-off
        ...
        cmd_q       = smoother.step(raw_target_q)
        s_kp, s_kd  = smoother.gain_scales()
        env.set_gains(kps_full * s_kp, kds_full * s_kd)
        env.step(cmd_q)

    Layers:
      A  startup pose blend: measured_q ─smoothstep→ raw target over ``blend_s``.
      D  running filter (``filter_kind``): always-on, damps abrupt policy output.
      C  per-tick clamp: |cmd - prev| ≤ ``clamp_delta`` per joint (hard safety).
      B  gain ramp scales (``gain_scales``): soft → full over ``gain_ramp_s``.
    """

    def __init__(
        self,
        dt: float,
        *,
        blend_s: float = 5.0,
        gain_ramp_s: float = 2.0,
        kp_scale_start: float = 0.15,
        kd_scale_start: float = 0.50,
        clamp_delta: float = 0.05,
        filter_kind: str = "critdamp",
        filter_tau: float = 0.08,
        filter_wn: float = 12.0,
    ):
        self.dt = float(dt)
        self.blend_s = float(blend_s)
        self.gain_ramp_s = float(gain_ramp_s)
        self.kp_scale_start = float(kp_scale_start)
        self.kd_scale_start = float(kd_scale_start)
        self.clamp_delta = float(clamp_delta)
        self._filter = make_filter(filter_kind, self.dt, tau=filter_tau, wn=filter_wn)

        self._measured_q: np.ndarray | None = None
        self._prev_cmd: np.ndarray | None = None
        self._t = 0.0

    def reset(self, measured_q) -> None:
        """Seed every layer at the current measured posture (so tick 0 ≈ no-op)."""
        self._measured_q = np.asarray(measured_q, dtype=np.float32).copy()
        self._prev_cmd = self._measured_q.copy()
        self._t = 0.0
        if self._filter is not None:
            self._filter.reset(self._measured_q)

    @property
    def elapsed(self) -> float:
        return self._t

    @property
    def blending(self) -> bool:
        return self.blend_s > 0.0 and self._t < self.blend_s

    def gain_scales(self) -> tuple[float, float]:
        """Layer B: (kp_scale, kd_scale), smoothstep soft→full over gain_ramp_s."""
        s_kp = gain_ramp_scale(self._t, self.gain_ramp_s, self.kp_scale_start)
        s_kd = gain_ramp_scale(self._t, self.gain_ramp_s, self.kd_scale_start)
        return s_kp, s_kd

    def step(self, raw_target_q, dt: float | None = None) -> np.ndarray:
        """Return the smoothed, clamped joint command for this tick."""
        if self._measured_q is None or self._prev_cmd is None:
            self.reset(raw_target_q)
        step_dt = self.dt if dt is None else float(dt)
        raw = np.asarray(raw_target_q, dtype=np.float32)

        # Layer A — startup pose blend from the captured posture to the target.
        if self.blending:
            alpha = self._t / self.blend_s
            target = blend_pose(self._measured_q, raw, alpha)
        else:
            target = raw

        # Layer D — always-on running filter.
        if self._filter is not None:
            target = self._filter.step(target)

        # Layer C — per-tick rate clamp (final hard safety rail).
        cmd = clamp_step_delta(self._prev_cmd, target, self.clamp_delta)

        self._prev_cmd = cmd
        self._t += step_dt
        return cmd
