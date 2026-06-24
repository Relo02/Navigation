"""
CasADi / IPOPT MPC trajectory tracker for Go2 quadruped.

Model  (fixes #3 holonomic mismatch + #4 actuator lag)
-------------------------------------------------------
  State    x = [px, py, yaw, vx, vy, wz]       NX = 6
  Control  u = [vx_cmd, vy_cmd, wz_cmd]         NU = 3

  Dynamics — discrete first-order lag (exact ZOH):
    lag_v = 1 - exp(-dt / tau_v)
    lag_w = 1 - exp(-dt / tau_w)

    vx_{k+1}  = (1-lag_v)*vx_k  + lag_v *vx_cmd_k
    vy_{k+1}  = (1-lag_w)*vy_k  + lag_w *vy_cmd_k
    wz_{k+1}  = (1-lag_w)*wz_k  + lag_w *wz_cmd_k

    px_{k+1}  = px_k + (vx_{k+1}*cos(yaw_k) - vy_{k+1}*sin(yaw_k))*dt
    py_{k+1}  = py_k + (vx_{k+1}*sin(yaw_k) + vy_{k+1}*cos(yaw_k))*dt
    yaw_{k+1} = yaw_k + wz_{k+1}*dt

  Position update uses the post-lag velocity so the MPC's predicted trajectory
  matches reality instead of assuming instantaneous response.

Obstacle avoidance — hybrid tanh + quadratic barrier (fix #7):
    J_obs = W*[0.5*(1-tanh(0.5*alpha*(d-r))) + 2*max(0, r-d)^2]

Warm-start health (fixes #1/#2):
    - zero-velocity fallback after _MAX_CONSEC_FAILURES consecutive IPOPT failures
    - cost-spike detection clears warm-start cache automatically

Adaptive velocity limits (fix #9):
    - NLP bounds are CasADi parameters, not compile-time constants
    - update_velocity_limits() adjusts them at runtime without NLP rebuild

author: Lorenzo Ortolani (adapted for Go2)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import casadi as ca

from a_star_mpc_planner.gaussian_grid_map import FixedGaussianGridMap


# ============================================================
# Configuration
# ============================================================

@dataclass
class MPCConfig:
    """All tunable MPC parameters."""

    # Horizon
    N: int   = 30
    dt: float = 0.1

    # Actuator lag time constants [s]  — fix #3/#4
    tau_v: float = 0.12   # forward/lateral velocity response time
    tau_w: float = 0.10   # angular velocity response time

    # Velocity limits (applied to commands u)
    vx_max:    float = 1.0
    vy_max:    float = 0.5
    omega_max: float = 1.5

    # Desired cruise speed
    v_ref: float = 0.5

    # Tracking cost weights  (applied to position/yaw states only)
    Q_x:        float = 20.0   # x-axis position tracking (forward)
    Q_y:        float = 20.0   # y-axis position tracking (lateral) — separate from Q_x
    Q_yaw:      float = 0.5
    Q_terminal: float = 50.0

    # Control-effort / smoothness weights
    R_vx:    float = 1.0   # forward velocity command effort
    R_vy:    float = 1.0   # lateral velocity command effort — separate from R_vx
    R_omega: float = 0.5
    R_jerk:  float = 0.2

    # Logistic sigmoid obstacle barrier
    W_obs_sigmoid:      float = 500.0
    obs_alpha:          float = 8.0
    obs_r:              float = 0.8

    # LiDAR point selection
    max_obs_constraints: int   = 15
    obs_check_radius:    float = 3.0

    # IPOPT
    max_iter:   int  = 100
    warm_start: bool = True
    print_level: int = 0


# ============================================================
# Result
# ============================================================

@dataclass
class MPCResult:
    success:       bool
    x_pred:        np.ndarray   # (N+1, 6)  [px, py, yaw, vx, vy, wz]
    u_opt:         np.ndarray   # (N,   3)  [vx_cmd, vy_cmd, wz_cmd]
    cost:          float
    solve_time_ms: float
    security_mode: bool = False

    @property
    def next_position(self) -> np.ndarray:
        return self.x_pred[1, :2]

    @property
    def next_yaw(self) -> float:
        return float(self.x_pred[1, 2])

    @property
    def predicted_xy(self) -> np.ndarray:
        return self.x_pred[:, :2]

    @property
    def predicted_yaw(self) -> np.ndarray:
        return self.x_pred[:, 2]


# ============================================================
# MPC Tracker
# ============================================================

class MPCTracker:
    """
    6-D path-tracking MPC with first-order actuator lag and sigmoid obstacle barrier.

    State:   [px, py, yaw, vx, vy, wz]
    Control: [vx_cmd, vy_cmd, wz_cmd]
    """

    NX = 6   # [px, py, yaw, vx, vy, wz]
    NU = 3   # [vx_cmd, vy_cmd, wz_cmd]
    _OBS_SENTINEL = 1e3

    _COST_SPIKE_FACTOR   = 5.0
    _COST_HISTORY_LEN    = 8
    _MAX_CONSEC_FAILURES = 3

    def __init__(self, config: Optional[MPCConfig] = None):
        self.cfg = config or MPCConfig()

        # Warm-start storage
        self._prev_u: Optional[np.ndarray] = None   # (N, NU)
        self._prev_x: Optional[np.ndarray] = None   # (N+1, NX)

        # Cached parametric NLP
        self._nlp_built: bool = False
        self._opti:   Optional[ca.Opti] = None
        self._X:      Optional[ca.MX]   = None
        self._U:      Optional[ca.MX]   = None
        self._p_x0:   Optional[ca.MX]   = None
        self._p_xref: Optional[ca.MX]   = None
        self._p_obs:  Optional[ca.MX]   = None

        # Parametric velocity-limit parameters (fix #9 — no NLP rebuild needed)
        self._p_vx_max:    Optional[ca.MX] = None
        self._p_vy_max:    Optional[ca.MX] = None
        self._p_omega_max: Optional[ca.MX] = None

        # Runtime-adaptive limits (initialised from config, reduced by mpc_node on failures)
        self._vx_max_eff    = self.cfg.vx_max
        self._vy_max_eff    = self.cfg.vy_max
        self._omega_max_eff = self.cfg.omega_max

        # Forward-only path progress
        self._path_progress_idx: int = 0
        self._last_valid_x0: Optional[np.ndarray] = None

        # Health tracking (#1 & #2)
        self._consecutive_failures: int = 0
        self._cost_history: list = []

    # ------------------------------------------------------------------
    # Grid map (API compat — not used in NLP)
    # ------------------------------------------------------------------

    def update_grid(self, grid_map: FixedGaussianGridMap) -> None:
        pass

    # ------------------------------------------------------------------
    # Adaptive velocity limits (fix #9)
    # ------------------------------------------------------------------

    def update_velocity_limits(
        self,
        vx_max:    Optional[float] = None,
        vy_max:    Optional[float] = None,
        omega_max: Optional[float] = None,
    ) -> None:
        """Adjust velocity command bounds at runtime (no NLP rebuild required)."""
        if vx_max    is not None:
            self._vx_max_eff    = float(vx_max)
        if vy_max    is not None:
            self._vy_max_eff    = float(vy_max)
        if omega_max is not None:
            self._omega_max_eff = float(omega_max)

    # ------------------------------------------------------------------
    # LiDAR point selection
    # ------------------------------------------------------------------

    def _select_obs_points(
        self,
        pts_2d:   np.ndarray,
        robot_xy: np.ndarray,
    ) -> np.ndarray:
        """Return up to max_obs_constraints nearest points, padded with sentinels."""
        n_target = self.cfg.max_obs_constraints

        if len(pts_2d) > 0:
            finite_mask = np.isfinite(pts_2d).all(axis=1)
            pts_2d = pts_2d[finite_mask]

        if len(pts_2d) > 0:
            dists = np.linalg.norm(pts_2d - robot_xy, axis=1)
            mask  = dists < self.cfg.obs_check_radius
            if np.any(mask):
                close   = pts_2d[mask]
                d_close = dists[mask]
                n_sel   = min(len(close), n_target)
                idx     = np.argsort(d_close)[:n_sel]
                selected = close[idx]
            else:
                selected = np.empty((0, 2))
        else:
            selected = np.empty((0, 2))

        n_found = len(selected)
        if n_found < n_target:
            sentinel = np.full((n_target - n_found, 2), self._OBS_SENTINEL)
            selected = np.vstack([selected, sentinel]) if n_found > 0 else sentinel

        return selected   # (max_obs_constraints, 2)

    # ------------------------------------------------------------------
    # Parametric NLP — built once
    # ------------------------------------------------------------------

    def _build_nlp(self) -> None:
        """
        Build the parametric NLP for 6-D kinematic model with actuator lag.

        State indices:  0=px  1=py  2=yaw  3=vx  4=vy  5=wz
        Cost penalises only position+yaw states (indices 0-2); velocity states
        are driven implicitly by the lag dynamics and position tracking.
        """
        cfg    = self.cfg
        N, dt  = cfg.N, cfg.dt
        NX, NU = self.NX, self.NU
        n_obs  = cfg.max_obs_constraints

        # Pre-compute lag coefficients (exact ZOH first-order response)
        lag_v = float(1.0 - np.exp(-dt / max(cfg.tau_v, 1e-6)))
        lag_w = float(1.0 - np.exp(-dt / max(cfg.tau_w, 1e-6)))

        opti   = ca.Opti()
        X      = opti.variable(NX, N + 1)
        U      = opti.variable(NU, N)
        p_x0   = opti.parameter(NX)
        p_xref = opti.parameter(NX, N + 1)
        p_obs  = opti.parameter(2, n_obs)

        # Parametric velocity limits (fix #9 — updated each solve, no rebuild)
        p_vx_max    = opti.parameter()
        p_vy_max    = opti.parameter()
        p_omega_max = opti.parameter()

        # Weight matrices — only position/yaw tracked, velocity states free
        q   = np.array([cfg.Q_x, cfg.Q_y, cfg.Q_yaw, 0.0, 0.0, 0.0])
        Q   = np.diag(q)
        Q_T = np.diag(q * cfg.Q_terminal)
        R   = np.diag([cfg.R_vx, cfg.R_vy, cfg.R_omega])

        # ── Objective ────────────────────────────────────────────────
        cost = 0.0

        for k in range(N):
            # Position + yaw tracking
            e    = X[:, k] - p_xref[:, k]
            cost += ca.mtimes([e.T, Q, e])

            # Control effort
            u_k   = U[:, k]
            cost += ca.mtimes([u_k.T, R, u_k])

            # Jerk smoothness
            if k > 0:
                du    = U[:, k] - U[:, k - 1]
                cost += cfg.R_jerk * ca.dot(du, du)

            # Hybrid obstacle barrier: tanh (soft zone) + quadratic (inside radius)
            for j in range(n_obs):
                dist_k = ca.sqrt(
                    (X[0, k] - p_obs[0, j]) ** 2 +
                    (X[1, k] - p_obs[1, j]) ** 2 + 1e-6
                )
                s_k         = cfg.obs_alpha * (dist_k - cfg.obs_r)
                cost       += cfg.W_obs_sigmoid * 0.5 * (1.0 - ca.tanh(0.5 * s_k))
                penetration  = ca.fmax(0.0, cfg.obs_r - dist_k)
                cost        += cfg.W_obs_sigmoid * 2.0 * penetration ** 2

        # Terminal cost
        e_T   = X[:, N] - p_xref[:, N]
        cost += ca.mtimes([e_T.T, Q_T, e_T])

        for j in range(n_obs):
            dist_T      = ca.sqrt(
                (X[0, N] - p_obs[0, j]) ** 2 +
                (X[1, N] - p_obs[1, j]) ** 2 + 1e-6
            )
            s_T          = cfg.obs_alpha * (dist_T - cfg.obs_r)
            cost        += cfg.W_obs_sigmoid * 0.5 * (1.0 - ca.tanh(0.5 * s_T))
            penetration_T = ca.fmax(0.0, cfg.obs_r - dist_T)
            cost         += cfg.W_obs_sigmoid * 2.0 * penetration_T ** 2

        opti.minimize(cost)

        # ── Dynamics — 6-D first-order lag (fixes #3/#4) ────────────
        for k in range(N):
            px_k  = X[0, k];  py_k  = X[1, k];  yaw_k = X[2, k]
            vx_k  = X[3, k];  vy_k  = X[4, k];  wz_k  = X[5, k]
            vx_cmd = U[0, k]; vy_cmd = U[1, k]; wz_cmd = U[2, k]

            # Actuator lag (exact ZOH discrete first-order response)
            vx_next  = (1.0 - lag_v) * vx_k  + lag_v  * vx_cmd
            vy_next  = (1.0 - lag_w) * vy_k  + lag_w  * vy_cmd
            wz_next  = (1.0 - lag_w) * wz_k  + lag_w  * wz_cmd

            # Position update with post-lag velocity (more accurate than commanded)
            cos_yaw  = ca.cos(yaw_k)
            sin_yaw  = ca.sin(yaw_k)
            px_next  = px_k  + (vx_next * cos_yaw - vy_next * sin_yaw) * dt
            py_next  = py_k  + (vx_next * sin_yaw + vy_next * cos_yaw) * dt
            yaw_next = yaw_k + wz_next * dt

            opti.subject_to(X[0, k + 1] == px_next)
            opti.subject_to(X[1, k + 1] == py_next)
            opti.subject_to(X[2, k + 1] == yaw_next)
            opti.subject_to(X[3, k + 1] == vx_next)
            opti.subject_to(X[4, k + 1] == vy_next)
            opti.subject_to(X[5, k + 1] == wz_next)

        opti.subject_to(X[:, 0] == p_x0)

        # ── Box constraints on commands (parametric — fix #9) ────────
        for k in range(N):
            opti.subject_to(U[0, k] >= 0.0)
            opti.subject_to(U[0, k] <= p_vx_max)
            opti.subject_to(opti.bounded(-p_vy_max,    U[1, k],  p_vy_max))
            opti.subject_to(opti.bounded(-p_omega_max, U[2, k],  p_omega_max))

        # ── Solver ────────────────────────────────────────────────────
        p_opts = {'expand': True, 'print_time': False}
        s_opts = {
            'max_iter':              cfg.max_iter,
            'print_level':           cfg.print_level,
            'sb':                    'yes',
            'warm_start_init_point': 'yes' if cfg.warm_start else 'no',
        }
        opti.solver('ipopt', p_opts, s_opts)

        self._opti      = opti
        self._X         = X
        self._U         = U
        self._p_x0      = p_x0
        self._p_xref    = p_xref
        self._p_obs     = p_obs
        self._p_vx_max    = p_vx_max
        self._p_vy_max    = p_vy_max
        self._p_omega_max = p_omega_max
        self._nlp_built = True

        self._prev_u = None
        self._prev_x = None

    # ------------------------------------------------------------------
    # Reference trajectory
    # ------------------------------------------------------------------

    def _build_reference(
        self,
        robot_state: np.ndarray,
        path_world:  list,
    ) -> np.ndarray:
        """
        Build an (N+1, 6) reference trajectory.

        Columns 0-2: [px, py, yaw] sampled along the A* path at v_ref m/s.
        Columns 3-5: [vx, vy, wz] desired velocity — used as warm-start seed
                     (not penalised in cost since Q[3:6] = 0).
        """
        N, dt, v_ref = self.cfg.N, self.cfg.dt, self.cfg.v_ref
        x_ref = np.zeros((N + 1, self.NX))

        if not path_world or len(path_world) < 2:
            x_ref[:, :3] = robot_state[:3]
            return x_ref

        path    = np.array(path_world, dtype=float)[:, :2]
        diffs   = np.diff(path, axis=0)
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        arc     = np.concatenate([[0.0], np.cumsum(seg_len)])
        total   = float(arc[-1])

        robot_xy  = robot_state[:2]
        distances = np.linalg.norm(path - robot_xy, axis=1)
        i_closest = int(np.argmin(distances))
        s0        = arc[i_closest]

        x_ref[0, 0] = robot_state[0]
        x_ref[0, 1] = robot_state[1]
        x_ref[0, 2] = robot_state[2]
        x_ref[0, 3] = robot_state[3] if len(robot_state) > 3 else v_ref
        x_ref[0, 4] = robot_state[4] if len(robot_state) > 4 else 0.0
        x_ref[0, 5] = robot_state[5] if len(robot_state) > 5 else 0.0

        for k in range(1, N + 1):
            s_k  = min(s0 + v_ref * k * dt, total)
            idx  = int(np.searchsorted(arc, s_k, side='right')) - 1
            idx  = np.clip(idx, 0, len(path) - 2)

            seg_l  = seg_len[idx]
            t      = np.clip((s_k - arc[idx]) / (seg_l + 1e-9), 0.0, 1.0)
            pos_xy = path[idx] + t * diffs[idx]
            seg_dir = diffs[idx] / (seg_l + 1e-9)
            yaw_k  = np.arctan2(seg_dir[1], seg_dir[0])

            x_ref[k, 0] = pos_xy[0]
            x_ref[k, 1] = pos_xy[1]
            x_ref[k, 2] = yaw_k
            x_ref[k, 3] = v_ref   # warm-start hint: cruise at v_ref
            x_ref[k, 4] = 0.0
            x_ref[k, 5] = 0.0

        return x_ref

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        robot_state:        np.ndarray,
        path_world:         list,
        obstacle_points_2d: Optional[np.ndarray] = None,
    ) -> MPCResult:
        """
        Solve the MPC optimisation.

        Parameters
        ----------
        robot_state        : (6,) [px, py, yaw, vx, vy, wz]  — accepts (3,) for
                             backward compat, padding velocity states with zeros.
        path_world         : list of (x, y[, z]) waypoints from A*
        obstacle_points_2d : (M, 2) LiDAR obstacle positions in world frame
                             (may be predicted future positions for dynamic obs)
        """
        t0       = time.perf_counter()
        cfg      = self.cfg
        N        = cfg.N
        NX, NU   = self.NX, self.NU

        # Accept 3-D state (backward compat) or 6-D state
        x0 = np.asarray(robot_state, dtype=float)
        if len(x0) == 3:
            x0 = np.concatenate([x0, [0.0, 0.0, 0.0]])
        elif len(x0) != NX:
            raise ValueError(f"Expected state length 3 or {NX}, got {len(x0)}")

        if not np.isfinite(x0).all():
            if self._last_valid_x0 is not None and np.isfinite(self._last_valid_x0).all():
                x0 = self._last_valid_x0.copy()
            else:
                x0 = np.zeros(NX, dtype=float)
        self._last_valid_x0 = x0.copy()

        path_len = len(path_world) if path_world else 0
        self._path_progress_idx = min(self._path_progress_idx, max(path_len - 1, 0))

        x_ref = self._build_reference(x0, path_world)
        if not np.isfinite(x_ref).all():
            x_ref = np.tile(x0, (N + 1, 1))

        # Obstacle array — always max_obs_constraints rows (sentinels when sparse)
        robot_xy = x0[:2]
        if obstacle_points_2d is not None and len(obstacle_points_2d) > 0:
            obs_pts = self._select_obs_points(obstacle_points_2d, robot_xy)
        else:
            obs_pts = np.full((cfg.max_obs_constraints, 2), self._OBS_SENTINEL)
        if not np.isfinite(obs_pts).all():
            obs_pts = np.full((cfg.max_obs_constraints, 2), self._OBS_SENTINEL)

        if not self._nlp_built:
            self._build_nlp()

        opti = self._opti

        # ── Parameter values ─────────────────────────────────────────
        opti.set_value(self._p_x0,      x0)
        opti.set_value(self._p_xref,    x_ref.T)    # (NX, N+1)
        opti.set_value(self._p_obs,     obs_pts.T)  # (2, n_obs)

        # Adaptive velocity limits (fix #9)
        opti.set_value(self._p_vx_max,    max(self._vx_max_eff,    0.05))
        opti.set_value(self._p_vy_max,    max(self._vy_max_eff,    0.05))
        opti.set_value(self._p_omega_max, max(self._omega_max_eff, 0.05))

        # ── Warm start ───────────────────────────────────────────────
        if cfg.warm_start and self._prev_u is not None and self._prev_x is not None:
            try:
                opti.set_initial(self._U, self._prev_u.T)
                opti.set_initial(self._X, self._prev_x.T)
            except Exception:
                opti.set_initial(self._X, x_ref.T)
                opti.set_initial(self._U, np.zeros((NU, N)))
        else:
            opti.set_initial(self._X, x_ref.T)
            opti.set_initial(self._U, np.zeros((NU, N)))

        # ── Cold-start recovery after too many consecutive failures (#1) ──
        # A stale/bad warm start (e.g. carried across an A* path reset or a
        # large yaw flip) is the usual cause of an IPOPT failure cascade.
        # Do NOT latch into an early return here: the previous version returned
        # cost=inf without ever re-attempting a solve, and _consecutive_failures
        # is only reset inside the success branch — so a single 3-failure burst
        # left the MPC permanently emitting cost=inf / x_ref (≈ blind cruise at
        # v_ref) until the node restarted. Instead drop to a cold start
        # (reference initial guess, no warm start) and fall through to RE-SOLVE,
        # which lets the optimiser recover on its own.
        if self._consecutive_failures >= self._MAX_CONSEC_FAILURES:
            self._prev_u = None
            self._prev_x = None
            opti.set_initial(self._X, x_ref.T)
            opti.set_initial(self._U, np.zeros((NU, N)))

        # ── Solve ────────────────────────────────────────────────────
        ipopt_ok = True
        try:
            sol      = opti.solve()
            success  = True
            cost_val = float(sol.value(opti.f))
        except RuntimeError:
            ipopt_ok = False
            sol      = opti.debug
            success  = False
            # Always clear warm start on IPOPT failure (#2)
            self._prev_u = None
            self._prev_x = None
            try:
                cost_val = float(sol.value(opti.f))
            except Exception:
                cost_val = float('inf')

        # ── Extract solution ─────────────────────────────────────────
        try:
            U_opt = np.array(sol.value(self._U), dtype=float)
            X_opt = np.array(sol.value(self._X), dtype=float)
            if np.any(np.isnan(U_opt)) or np.any(np.isnan(X_opt)):
                raise ValueError('NaN in solution')
            u_seq  = U_opt.T    # (N,  NU)
            x_pred = X_opt.T    # (N+1, NX)

            if ipopt_ok:
                self._consecutive_failures = 0
                self._cost_history.append(cost_val)
                if len(self._cost_history) > self._COST_HISTORY_LEN:
                    self._cost_history.pop(0)

                # Clear warm start if cost spikes — prevents cascading degradation (#2)
                if len(self._cost_history) >= 3:
                    avg = float(np.mean(self._cost_history[:-1]))
                    if avg > 0 and cost_val > avg * self._COST_SPIKE_FACTOR:
                        self._prev_u = None
                        self._prev_x = None
                    else:
                        self._prev_u = np.vstack([u_seq[1:],  u_seq[-1:]])
                        self._prev_x = np.vstack([x_pred[1:], x_pred[-1:]])
                else:
                    self._prev_u = np.vstack([u_seq[1:],  u_seq[-1:]])
                    self._prev_x = np.vstack([x_pred[1:], x_pred[-1:]])
            else:
                self._consecutive_failures += 1

        except Exception:
            success  = False
            self._consecutive_failures += 1
            self._prev_u = None
            self._prev_x = None
            u_seq  = np.zeros((N, NU))
            x_pred = x_ref.copy()

        return MPCResult(
            success=success,
            x_pred=x_pred,
            u_opt=u_seq,
            cost=cost_val,
            solve_time_ms=(time.perf_counter() - t0) * 1e3,
        )
