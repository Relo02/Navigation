"""
AMO policy + environment wrapper around the RoboJuDo framework.

This is the "AMO code": it loads RoboJuDo's ``AMOPolicy`` (the trained AMO RL
gait, torchscript) and a ``UnitreeEnv`` (the G1 over CycloneDDS via
unitree_sdk2py), maps the 23-DoF observation / 15-DoF action joint orders onto
the env's 29-DoF order, and builds the 29-DoF PD target consumed by
``env.step``. The smoothing of that target is *not* done here — it is layered on
top by ``amo_inference.py`` via ``joint_filters.JointSmoother``.

Mirrors the proven construction in
``G1_navigation/policy/real_g1_walking_policy.py`` (joint orders, default pose,
AMO PD-gain overrides, env config) but trimmed to the inference path.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import numpy as np

logger = logging.getLogger("amo.policy")


# ── G1 joint layout (must match RoboJuDo's G1AmoDoF ordering) ──────────────────
G1_AMO_OBS_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

G1_AMO_ACTION_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.5, "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.2, "left_elbow_joint": 0.3,
    "right_shoulder_pitch_joint": 0.5, "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": -0.2, "right_elbow_joint": 0.3,
}


def gravity_tilt_angle(quat_xyzw) -> float:
    """Angle (rad) between -gravity and the robot body-z axis, from base quat."""
    qz, qw = float(quat_xyzw[2]), float(quat_xyzw[3])
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return float(np.arccos(float(np.clip(-gz, -1.0, 1.0))))


def _resolve_robojudo_root(robojudo_root: str | None) -> Path:
    if robojudo_root:
        root = Path(robojudo_root).expanduser().resolve()
    else:
        # default deployment mount; see docker/docker-compose.yml
        root = Path("/workspace/RoboJuDo")
    if not root.exists():
        raise FileNotFoundError(
            f"RoboJuDo root not found: {root}. Mount the RoboJuDo checkout there "
            f"or set robojudo_root / ROBOJUDO_ROOT."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _inject_optional_stubs() -> None:
    """Shim optional RoboJuDo deps that inference does not need (only matters on
    a minimal install; the amo_policy image ships the real packages)."""
    if "mujoco" not in sys.modules:
        try:
            import mujoco  # noqa: F401
        except ImportError:
            mod = types.ModuleType("mujoco")
            mod.mjtGeom = type("mjtGeom", (), dict(
                mjGEOM_SPHERE=2, mjGEOM_CAPSULE=7, mjGEOM_BOX=6,
                mjGEOM_CYLINDER=3, mjGEOM_ARROW=5, mjGEOM_LINE=1))
            sys.modules["mujoco"] = mod


class AmoDeployment:
    """Holds a RoboJuDo AMOPolicy + UnitreeEnv and builds 29-DoF PD targets.

    Lifecycle::

        dep = AmoDeployment(robojudo_root=..., net_if="eth0")
        dep.setup()
        dep.wait_until_ready()
        ...
        dep.update()                      # refresh state from DDS
        q   = dep.dof_pos                 # measured 29-DoF posture
        tgt = dep.policy_target(cmd3)     # 29-DoF PD target from the policy
        dep.command(cmd_q, s_kp, s_kd)    # set gains + publish (no-op if observe_only)
    """

    def __init__(
        self,
        *,
        robojudo_root: str | None = None,
        net_if: str = "eth0",
        device: str = "cpu",
        observe_only: bool = False,
        max_forward_vel: float = 0.8,
        max_yaw_rate: float = 0.4,
        amo_pd_gains: bool = True,
    ):
        self.robojudo_root = robojudo_root
        self.net_if = net_if
        self.device = device
        self.observe_only = observe_only
        self.max_forward_vel = float(max_forward_vel)
        self.max_yaw_rate = float(max_yaw_rate)
        self.amo_pd_gains = amo_pd_gains

        self.env = None
        self.policy = None
        self.cfg = None
        self.control_dt = 0.02
        self._obs_idx: np.ndarray | None = None
        self._action_idx: np.ndarray | None = None
        self._env_default_q: np.ndarray | None = None
        self._position_limits = None
        self.kps_full: np.ndarray | None = None
        self.kds_full: np.ndarray | None = None

    # ── setup ────────────────────────────────────────────────────────────────
    def setup(self) -> None:
        _resolve_robojudo_root(self.robojudo_root)
        _inject_optional_stubs()

        self._load_policy()
        self._build_env()
        self._build_joint_maps()
        self._compute_gains()

    def _load_policy(self) -> None:
        from robojudo.config.g1.policy.g1_amo_policy_cfg import G1AmoPolicyCfg
        from robojudo.policy.amo_policy import AMOPolicy

        self.cfg = G1AmoPolicyCfg()
        self.policy = AMOPolicy(cfg_policy=self.cfg, device=self.device)
        self.policy.reset()
        self.control_dt = 1.0 / float(self.cfg.freq)
        logger.info("loaded RoboJuDo AMOPolicy: %s (control_dt=%.4fs)",
                    self.cfg.policy_file, self.control_dt)

    def _build_env(self) -> None:
        from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg
        from robojudo.environment.unitree_env import UnitreeEnv

        cfg_env = G1RealEnvCfg()
        cfg_env.env_type = "UnitreeEnv"
        cfg_env.unitree.net_if = self.net_if
        cfg_env.unitree.lowcmd_topic = "rt/lowcmd"
        cfg_env.unitree.lowstate_topic = "rt/lowstate"
        cfg_env.unitree.sport_state_topic = "rt/odommodestate"
        cfg_env.unitree.msg_type = "hg"
        cfg_env.unitree.robot = "g1"
        cfg_env.odometry_type = "UNITREE"
        cfg_env.act = not self.observe_only

        logger.info("creating UnitreeEnv (net_if=%s, act=%s, observe_only=%s)",
                    self.net_if, cfg_env.act, self.observe_only)
        self.env = UnitreeEnv(cfg_env=cfg_env, device=self.device)

    def _build_joint_maps(self) -> None:
        env_joint_names = list(self.env.joint_names)
        self._obs_idx = self._indices(env_joint_names, G1_AMO_OBS_JOINT_ORDER, "obs")
        self._action_idx = self._indices(env_joint_names, G1_AMO_ACTION_JOINT_ORDER, "action")

        q = np.zeros(len(env_joint_names), dtype=np.float32)
        for name, value in DEFAULT_JOINT_POS.items():
            if name in env_joint_names:
                q[env_joint_names.index(name)] = value
        self._env_default_q = q
        self._position_limits = getattr(self.env, "position_limits", None)

    @staticmethod
    def _indices(env_joint_names, order, label) -> np.ndarray:
        missing = [n for n in order if n not in env_joint_names]
        if missing:
            raise RuntimeError(f"{label} joint order missing from env: {missing}")
        return np.asarray([env_joint_names.index(n) for n in order], dtype=np.int64)

    def _compute_gains(self) -> None:
        """AMO-trained PD gains overlaid on the env defaults (layer-B target)."""
        kps = np.asarray(self.env.stiffness, dtype=np.float64).copy()
        kds = np.asarray(self.env.damping, dtype=np.float64).copy()
        if self.amo_pd_gains:
            from robojudo.config.g1.policy.g1_amo_policy_cfg import G1AmoDoF
            amo = G1AmoDoF()
            env_joint_names = list(self.env.joint_names)
            for name, kp, kd in zip(amo.joint_names, amo.stiffness, amo.damping):
                if name in env_joint_names:
                    idx = env_joint_names.index(name)
                    kps[idx] = float(kp)
                    kds[idx] = float(kd)
            logger.info("applied AMO PD-gain overrides for %d joints", len(amo.joint_names))
        self.kps_full = kps
        self.kds_full = kds

    # ── runtime ──────────────────────────────────────────────────────────────
    def wait_until_ready(self) -> None:
        if hasattr(self.env, "wait_for_low_state"):
            logger.info("waiting for first LowState over DDS ...")
            self.env.wait_for_low_state()
        self.update()
        logger.info("first state received; robot is ready for activation")

    def update(self) -> None:
        self.env.update()

    @property
    def dof_pos(self) -> np.ndarray:
        return np.asarray(self.env.dof_pos, dtype=np.float32).copy()

    @property
    def base_quat(self) -> np.ndarray:
        return np.asarray(self.env.base_quat, dtype=np.float32)

    @property
    def num_dofs(self) -> int:
        return int(self.env.num_dofs)

    def _build_ctrl(self, command3) -> dict:
        vx, vy, yaw = (float(command3[0]), float(command3[1]), float(command3[2]))
        return {"UnitreeCtrl": {
            "axes": {
                "LeftY": float(np.clip(vx / self.max_forward_vel, -1.0, 1.0)),
                "LeftX": float(np.clip(vy / self.max_forward_vel, -1.0, 1.0)),
                "RightX": float(np.clip(-yaw / self.max_yaw_rate, -1.0, 1.0)),
                "RightY": 0.0,
            },
            "button_event": [],
        }}

    def _env_data(self):
        dof_pos = np.asarray(self.env.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(self.env.dof_vel, dtype=np.float32)
        return types.SimpleNamespace(
            dof_pos=dof_pos[self._obs_idx].astype(np.float32),
            dof_vel=dof_vel[self._obs_idx].astype(np.float32),
            base_quat=np.asarray(self.env.base_quat, dtype=np.float32),
            base_ang_vel=np.asarray(self.env.base_ang_vel, dtype=np.float32),
        )

    def policy_target(self, command3) -> np.ndarray:
        """Run one policy inference for ``command3=(vx,vy,yaw)`` and return the
        29-DoF PD target. Does NOT publish — the caller smooths then commands."""
        ctrl_data = self._build_ctrl(command3)
        env_data = self._env_data()
        obs, _ = self.policy.get_observation(env_data, ctrl_data)
        action = self.policy.get_action(obs)
        self.policy.post_step_callback()
        return self._make_target(action)

    def standing_target(self) -> np.ndarray:
        """29-DoF PD target for the AMO standing pose (action = 0)."""
        zero = np.zeros(len(self._action_idx), dtype=np.float32)
        return self._make_target(zero)

    def _make_target(self, action) -> np.ndarray:
        target_q = self._env_default_q.copy()
        target_q[self._obs_idx] = np.asarray(self.policy.default_dof_pos, dtype=np.float32)
        target_q[self._action_idx] = np.asarray(self.policy.default_pos + action, dtype=np.float32)
        return self._clamp_to_limits(target_q)

    def _clamp_to_limits(self, target_q) -> np.ndarray:
        if self._position_limits is None:
            return np.asarray(target_q, dtype=np.float32)
        lo = self._position_limits[:, 0]
        hi = self._position_limits[:, 1]
        return np.clip(target_q, lo, hi).astype(np.float32)

    def command(self, cmd_q, s_kp: float = 1.0, s_kd: float = 1.0) -> None:
        """Apply (scaled) gains and publish the PD target. No-op if observe_only."""
        if self.observe_only:
            return
        self.env.set_gains(self.kps_full * float(s_kp), self.kds_full * float(s_kd))
        self.env.step(self._clamp_to_limits(cmd_q))

    def shutdown(self) -> None:
        if self.env is not None and hasattr(self.env, "shutdown"):
            self.env.shutdown()
