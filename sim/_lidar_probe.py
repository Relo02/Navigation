"""Throwaway probe: which RTX-lidar creation method actually renders points in
Isaac 5.1? Creates 3 lidars over a simple ground+boxes scene, publishes each on
its own ROS2 topic. Compare widths with `ros2 topic echo /probe_X --field width`.

  A /probe_cam   force_camera_prim=True, config=Livox_Mid360 (current g1_sim_scene)
  B /probe_omni  config=Livox_Mid360   (modern OmniLidar path, no camera prim)
  C /probe_rotary config=Example_Rotary (built-in USD-asset config, known good)
"""
import os
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_motion_bvh": True})

import carb  # noqa: E402
import omni  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.api.objects import GroundPlane, VisualCuboid  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Gf  # noqa: E402

for _ext in ("isaacsim.sensors.rtx", "isaacsim.ros2.bridge"):
    enable_extension(_ext)
app.update()

# Register our custom lidar config folder (same as g1_sim_scene.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_settings = carb.settings.get_settings()
_KEY = "/app/sensors/nv/lidar/profileBaseFolder"
_folders = list(_settings.get(_KEY) or [])
_cfgdir = os.path.join(_HERE, "lidar_configs") + "/"
if _cfgdir not in _folders:
    _folders.append(_cfgdir)
    _settings.set_string_array(_KEY, _folders)

# Minimal scene with geometry the lidar can hit.
ctx = omni.usd.get_context()
ctx.new_stage()
app.update()
GroundPlane(prim_path="/World/ground", size=50.0)
for i, (x, y) in enumerate([(3, 0), (-3, 0), (0, 3), (0, -3), (2, 2)]):
    VisualCuboid(prim_path=f"/World/box{i}", position=(x, y, 0.5),
                 scale=(1.0, 1.0, 1.0))
app.update()


def make_lidar(path, topic, config, force_camera_prim):
    kwargs = dict(path=path, parent=None, config=config)
    if force_camera_prim:
        kwargs["force_camera_prim"] = True
    ok, prim = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", **kwargs)
    if prim is None or not prim.IsValid():
        carb.log_warn(f"[probe] {topic}: creation FAILED")
        return
    # lidar sits at z=1.0 so it sees ground + boxes
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
    rp = rep.create.render_product(prim.GetPath(), [1, 1], name=topic.strip("/") + "_rp")
    w = rep.writers.get("RtxLidarROS2PublishPointCloud")
    w.initialize(topicName=topic, frameId="probe")
    w.attach([rp])
    carb.log_warn(f"[probe] {topic}: created via config={config} "
                  f"force_camera_prim={force_camera_prim}")


make_lidar("/World/probe_cam", "/probe_cam", "Livox_Mid360", True)
make_lidar("/World/probe_omni", "/probe_omni", "Livox_Mid360", False)
make_lidar("/World/probe_rotary", "/probe_rotary", "Example_Rotary", False)

sim = SimulationContext(physics_dt=1.0 / 200.0, rendering_dt=1.0 / 60.0,
                        stage_units_in_meters=1.0)
app.update()
sim.play()
carb.log_warn("[probe] playing; publishing /probe_cam /probe_omni /probe_rotary")
while app.is_running():
    app.update()
sim.stop()
app.close()
