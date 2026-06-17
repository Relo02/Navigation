"""
Pure smooth-activation helpers (layers A/B/C of docs/amo_inference_plan.md).

Dependency-free (numpy only, no robojudo / torch / DDS / argparse) so they can
be unit-tested without a robot. Mirrors the proven helpers used by the
G1_navigation deployment (documentation/G1_SAFE_INITIAL_STATE.md).
"""

from __future__ import annotations

import numpy as np


def smoothstep(alpha: float) -> float:
    """S-curve easing ``alpha**2 * (3 - 2*alpha)``, clamped to [0, 1].

    Zero slope at both ends, so commanded joint *velocity* starts and finishes
    at zero — no step in velocity at the ramp edges.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def gain_ramp_scale(elapsed_s: float, ramp_s: float, scale_start: float) -> float:
    """Per-tick PD-gain scale: smoothstep from ``scale_start`` (t=0) to 1.0 (t>=ramp_s).

    ``scale_start`` is the soft initial fraction of the full gain (e.g. 0.15 for
    kp). With ``ramp_s <= 0`` the ramp is skipped and full gain (1.0) is used.
    """
    if ramp_s <= 0.0:
        return 1.0
    g = smoothstep(elapsed_s / ramp_s)
    return float(scale_start + (1.0 - scale_start) * g)


def command_ramp_factor(elapsed_s: float, ramp_s: float) -> float:
    """Command in-ramp: smoothstep from 0.0 (t=0) to 1.0 (t>=ramp_s).

    Multiplies the (vx, vy, yaw_rate) command so the robot eases into motion
    instead of stepping to full speed. ``ramp_s <= 0`` disables it (factor 1.0).
    """
    if ramp_s <= 0.0:
        return 1.0
    return smoothstep(elapsed_s / ramp_s)


def blend_pose(start_q, target_q, alpha: float):
    """Smoothstep interpolation between two joint-position vectors.

    ``alpha`` is the raw linear progress in [0, 1]; the S-curve is applied
    internally. Returns a float32 array shaped like the inputs.
    """
    s = smoothstep(alpha)
    start = np.asarray(start_q, dtype=np.float32)
    target = np.asarray(target_q, dtype=np.float32)
    return ((1.0 - s) * start + s * target).astype(np.float32)


def clamp_step_delta(prev_q, desired_q, max_delta: float):
    """Clamp the per-tick change from ``prev_q`` toward ``desired_q``.

    Caps |desired - prev| at ``max_delta`` per joint per control tick — a safety
    rail so a bad target can't snap a joint. ``max_delta <= 0`` returns the
    desired target unclamped.
    """
    prev = np.asarray(prev_q, dtype=np.float32)
    desired = np.asarray(desired_q, dtype=np.float32)
    if max_delta <= 0.0:
        return desired.copy()
    delta = np.clip(desired - prev, -max_delta, max_delta)
    return (prev + delta).astype(np.float32)
