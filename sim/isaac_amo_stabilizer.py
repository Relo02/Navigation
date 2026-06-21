"""Drive the Navigation AMO policy on the Isaac-sim G1 so it stands/stays still.

This reuses the *exact* Navigation AMO code (amo/amo_policy.py::AmoDeployment +
the RoboJuDo AMOPolicy it loads). AmoDeployment talks to the robot only through
a small env interface -- it reads ``env.{dof_pos,dof_vel,base_quat,base_ang_vel,
joint_names,stiffness,damping,position_limits,num_dofs}`` and writes through
``env.set_gains()`` / ``env.step()``. We provide that interface backed by an
Isaac articulation instead of the real-robot UnitreeEnv (CycloneDDS), so the
proven policy + joint maps + default pose + AMO PD gains are used unchanged.

Standing also keeps the (otherwise unactuated) robot from collapsing -- the
default Isaac articulation has no drive stiffness, so the many small hand bodies
flop and self-collide into a PhysX broadphase blow-up. Holding the AMO default
pose with proper PD gains from the first step prevents that.

Usage (inside g1_sim_scene.py, after sim_ctx.play()):

    from isaac_amo_stabilizer import Stabilizer
    stab = Stabilizer(robot_prim="/World/g1", robojudo_root=..., command=(0,0,0))
    stab.setup()
    while running:
        stab.on_physics_step()   # called every physics step; decimates internally
        simulation_app.update()
"""
from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

logger = logging.getLogger("amo.isaac")


def _inject_robojudo_stubs():
    """Shim optional RoboJuDo deps not needed for inference, so `import robojudo`
    works under Isaac's bundled Python. Same proven set as walking_policy.py."""
    if "mujoco" not in sys.modules:
        mod = types.ModuleType("mujoco")
        mod.mjtGeom = type("mjtGeom", (), dict(
            mjGEOM_SPHERE=2, mjGEOM_CAPSULE=7, mjGEOM_BOX=6,
            mjGEOM_CYLINDER=3, mjGEOM_ARROW=5, mjGEOM_LINE=1))
        sys.modules["mujoco"] = mod
    if "mujoco_viewer" not in sys.modules:
        sys.modules["mujoco_viewer"] = types.ModuleType("mujoco_viewer")
    if "colorlog" not in sys.modules:
        class _ColoredFormatter(logging.Formatter):
            def __init__(self, fmt=None, datefmt=None, style="%", **_kw):
                super().__init__(fmt=fmt, datefmt=datefmt, style=style)

            def format(self, record):
                if not hasattr(record, "log_color"):
                    record.log_color = ""
                if not hasattr(record, "reset"):
                    record.reset = ""
                return super().format(record)
        mod = types.ModuleType("colorlog")
        mod.getLogger = logging.getLogger
        mod.ColoredFormatter = _ColoredFormatter
        sys.modules["colorlog"] = mod
    for name, attrs in [
        ("msgpack", {"packb": lambda *a, **kw: b"", "unpackb": lambda *a, **kw: {}}),
        ("msgpack_numpy", {"patch": lambda: None}),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(mod, key, value)
            sys.modules[name] = mod
    if "box" not in sys.modules:
        class _Box(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as exc:
                    raise AttributeError(key) from exc

            def __setattr__(self, key, value):
                self[key] = value
        mod = types.ModuleType("box")
        mod.Box = _Box
        sys.modules["box"] = mod

# Make the Navigation amo/ package importable (sibling of sim/).
_NAV_ROOT = Path(__file__).resolve().parents[1]
_AMO_DIR = _NAV_ROOT / "amo"
if str(_AMO_DIR) not in sys.path:
    sys.path.insert(0, str(_AMO_DIR))


def _quat_rotate_inverse(q_wxyz, v):
    """Rotate world-frame vector v into body frame given body quat (w,x,y,z)."""
    w, x, y, z = (float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]))
    # q_conj * v * q ; expand for a 3-vector
    qv = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return (v - 2.0 * (w * uv - uuv)).astype(np.float32)


class IsaacG1Env:
    """UnitreeEnv-compatible facade over an Isaac SingleArticulation.

    Only the members AmoDeployment touches are implemented. ``step`` sets the
    position targets (the Isaac main loop advances physics); ``set_gains``
    pushes PD gains onto the articulation drives.
    """

    def __init__(self, articulation, default_stiffness=60.0, default_damping=2.0):
        from isaacsim.core.utils.types import ArticulationAction
        self._ArticulationAction = ArticulationAction
        self._art = articulation
        self.joint_names = list(articulation.dof_names)
        self.num_dofs = len(self.joint_names)

        # State caches refreshed by update().
        self.dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
        self.dof_vel = np.zeros(self.num_dofs, dtype=np.float32)
        self.base_quat = np.array([0, 0, 0, 1], dtype=np.float32)   # xyzw
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.low_state = object()   # AmoDeployment only checks "is not None"

        # Default PD gains for ALL dofs; AMO overrides the 23 named joints (and
        # the hands keep these so they hold position instead of flopping).
        self.stiffness = np.full(self.num_dofs, float(default_stiffness), dtype=np.float64)
        self.damping = np.full(self.num_dofs, float(default_damping), dtype=np.float64)

        # Joint position limits (lower, upper) per dof, from the articulation.
        try:
            limits = articulation.get_dof_limits()  # shape (num_dofs, 2)
            self.position_limits = np.asarray(limits, dtype=np.float32).reshape(self.num_dofs, 2)
        except Exception:  # noqa: BLE001
            self.position_limits = None

    # ── interface AmoDeployment expects ───────────────────────────────────────
    def wait_for_low_state(self):
        return None

    def update(self):
        self.dof_pos = np.asarray(self._art.get_joint_positions(), dtype=np.float32).reshape(-1)
        self.dof_vel = np.asarray(self._art.get_joint_velocities(), dtype=np.float32).reshape(-1)
        _, quat_wxyz = self._art.get_world_pose()           # Isaac returns (pos, wxyz)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
        self.base_quat = quat_wxyz[[1, 2, 3, 0]]            # -> xyzw for RoboJuDo
        ang_w = np.asarray(self._art.get_angular_velocity(), dtype=np.float32).reshape(-1)
        self.base_ang_vel = _quat_rotate_inverse(quat_wxyz, ang_w.astype(np.float64))

    def set_gains(self, stiffness, damping):
        ctrl = self._art.get_articulation_controller()
        ctrl.set_gains(kps=np.asarray(stiffness, dtype=np.float32),
                       kds=np.asarray(damping, dtype=np.float32))

    def step(self, pd_target, hand_pose=None):
        # apply_action sets PD *targets* (drives toward them via the joint
        # stiffness/damping); set_joint_positions would teleport instead.
        self._art.apply_action(self._ArticulationAction(
            joint_positions=np.asarray(pd_target, dtype=np.float32)))

    def shutdown(self):
        pass


class Stabilizer:
    """Wraps an Isaac-backed AmoDeployment and runs the AMO standing loop."""

    def __init__(self, robot_prim="/World/g1", robojudo_root=None,
                 command=(0.0, 0.0, 0.0), physics_dt=1.0 / 200.0, device="cpu",
                 cmd_vel_topic="/cmd_vel"):
        self.robot_prim = robot_prim
        self.robojudo_root = robojudo_root or os.environ.get(
            "ROBOJUDO_ROOT",
            str(_NAV_ROOT.parent / "g1-isaac-sim" / "policies" / "RoboJuDo"))
        self.command = np.asarray(command, dtype=np.float32)
        self.physics_dt = float(physics_dt)
        self.device = device
        self.cmd_vel_topic = cmd_vel_topic
        self._dep = None
        self._decim = 1
        self._phys_count = 0
        self._ros_node = None        # rclpy node for /cmd_vel teleop (optional)

    def _setup_cmd_vel(self):
        """Subscribe to /cmd_vel (geometry_msgs/Twist) so teleop can drive the
        gait. In sim the publisher is usually the keyboard (teleop_twist_keyboard)
        or a gamepad (joy_to_cmdvel) -- both publish RELIABLE on /cmd_vel, so we
        match. Best-effort setup: if rclpy/msgs are unavailable the robot just
        stands in place (command stays zero)."""
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            if not rclpy.ok():
                rclpy.init(args=None)
            node = rclpy.create_node("amo_cmd_vel_listener")

            def _on_twist(msg):
                self.command = np.array(
                    [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

            node.create_subscription(Twist, self.cmd_vel_topic, _on_twist, 10)
            self._ros_node = node
            self._rclpy = rclpy
            logger.info("AMO teleop: subscribed to %s (geometry_msgs/Twist)", self.cmd_vel_topic)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AMO teleop disabled (no /cmd_vel): %s", exc)
            self._ros_node = None

    def setup(self):
        os.environ.setdefault("ROBOJUDO_ROOT", self.robojudo_root)
        _inject_robojudo_stubs()
        from isaacsim.core.prims import SingleArticulation

        art = SingleArticulation(prim_path=self.robot_prim, name="g1_amo")
        art.initialize()        # requires the timeline to be playing

        # Build an AmoDeployment whose env is our Isaac facade (override the
        # DDS env construction; everything else in AmoDeployment is reused).
        from amo_policy import AmoDeployment

        env = IsaacG1Env(art)

        class _IsaacAmoDeployment(AmoDeployment):
            def _build_env(self_inner):
                self_inner.env = env

            def wait_until_ready(self_inner):
                self_inner.update()

        self._dep = _IsaacAmoDeployment(
            robojudo_root=self.robojudo_root, device=self.device,
            observe_only=False, amo_pd_gains=True)
        self._dep.setup()
        self._dep.wait_until_ready()

        # AMO runs at cfg.freq (50 Hz); decimate the physics step accordingly.
        self._decim = max(1, int(round(self._dep.control_dt / self.physics_dt)))
        logger.info("AMO stabilizer ready: control_dt=%.4fs physics_dt=%.4fs decim=%d, cmd=%s",
                    self._dep.control_dt, self.physics_dt, self._decim, self.command.tolist())

        # Seed the drive gains + an initial standing target so the very first
        # physics steps hold the pose (anti-collapse) before the policy ticks.
        self._dep.update()
        self._dep.command(self._dep.standing_target(), 1.0, 1.0)

        # Teleop: pull velocity commands off /cmd_vel.
        self._setup_cmd_vel()

    def on_physics_step(self):
        """Call once per physics step; runs policy inference every `decim` steps."""
        if self._dep is None:
            return
        # Drain any pending /cmd_vel messages (non-blocking) to refresh command.
        if self._ros_node is not None:
            self._rclpy.spin_once(self._ros_node, timeout_sec=0.0)
        if self._phys_count % self._decim == 0:
            self._dep.update()
            target = self._dep.policy_target(self.command)
            self._dep.command(target, 1.0, 1.0)
        self._phys_count += 1

    def shutdown(self):
        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        if self._dep is not None:
            self._dep.shutdown()
