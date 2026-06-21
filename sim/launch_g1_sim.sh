#!/usr/bin/env bash
# Launch Isaac Sim with the Unitree G1 + a Livox MID-360 LiDAR and IMU, all
# published to ROS 2 for the DLIO (direct_lidar_inertial_odometry) stack.
#
#   /livox/lidar  (sensor_msgs/PointCloud2, frame=livox_frame)
#   /livox/imu    (sensor_msgs/Imu,         frame=livox_frame)
#   /clock        (rosgraph_msgs/Clock)
#
# A companion ROS 2 node (g1_sim_bridge, in Navigation/ros2_ws) QoS-relays
# /livox/lidar -> /livox/lidar_reliable + /livox/imu_raw -> /livox/imu so DLIO's
# RELIABLE cloud subscriber matches. See Navigation/sim/README.md.
#
# Usage:
#   ./launch_g1_sim.sh                # GUI
#   ./launch_g1_sim.sh --headless     # no window
#   ROS_DOMAIN_ID=7 ./launch_g1_sim.sh
#   ISAAC_SIM_PATH=/opt/isaac-sim ./launch_g1_sim.sh
#   ./launch_g1_sim.sh --usd /path/to/other_stage.usd
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE="$SCRIPT_DIR/g1_sim_scene.py"

# --teleop[=keyboard|joystick|none]: how to send /cmd_vel references to the AMO
# gait (only meaningful with --stabilize). Parsed out here so it is NOT forwarded
# to the Isaac scene. keyboard (default) just prints the command to run in the
# localization container; joystick backgrounds the gamepad nodes there.
TELEOP=""
SCENE_ARGS=()
for _a in "$@"; do
    case "$_a" in
        --teleop)    TELEOP="keyboard" ;;
        --teleop=*)  TELEOP="${_a#*=}" ;;
        *)           SCENE_ARGS+=("$_a") ;;
    esac
done
set -- "${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}"

# Isaac Sim lives in the g1-isaac-sim repo (not in Navigation). Override with
# ISAAC_SIM_PATH if your install is elsewhere.
ISAAC_DIR="${ISAAC_SIM_PATH:-/home/lorenzo/TalosRoboticsAI/g1/g1-isaac-sim/isaac-sim}"

if [[ ! -x "$ISAAC_DIR/python.sh" ]]; then
    echo "ERROR: Isaac Sim python.sh not found at $ISAAC_DIR/python.sh" >&2
    echo "       Set ISAAC_SIM_PATH=/path/to/isaac-sim and retry." >&2
    exit 1
fi

# --- ROS 2 transport: match the navigation stack (Humble + CycloneDDS) ------
# FORCE humble: the deploy stack (livox_ros_driver2, DLIO, dlio_sim.yaml) is
# built for ROS 2 Humble. If a /opt/ros/jazzy is sourced in your shell, Isaac
# otherwise picks up ROS_DISTRO=jazzy and its bundled jazzy bridge, which does
# NOT interoperate with the Humble nodes (different typesupport).
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Scrub any system ROS (e.g. /opt/ros/jazzy) sourced in the shell, so Isaac uses
# ONLY its bundled Humble bridge. Isaac runs Python 3.11; system Humble/Jazzy on
# PYTHONPATH would break its embedded rclpy import.
_scrub() {  # remove path entries containing /opt/ros from a :-list
    local out="" e
    IFS=':' read -ra _parts <<< "${1:-}"
    for e in "${_parts[@]}"; do [[ -z "$e" || "$e" == */opt/ros/* ]] || out="${out:+$out:}$e"; done
    echo "$out"
}
export PYTHONPATH="$(_scrub "${PYTHONPATH:-}")"
export LD_LIBRARY_PATH="$(_scrub "${LD_LIBRARY_PATH:-}")"
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH 2>/dev/null || true

# Use Isaac's *bundled* Humble bridge libraries.
BRIDGE_LIB="$ISAAC_DIR/exts/isaacsim.ros2.bridge/$ROS_DISTRO/lib"
if [[ -d "$BRIDGE_LIB" ]]; then
    export LD_LIBRARY_PATH="$BRIDGE_LIB:${LD_LIBRARY_PATH:-}"
else
    echo "WARNING: bundled ROS 2 bridge libs not found at $BRIDGE_LIB" >&2
fi

# Regenerate the MID-360 config if it is missing.
if [[ ! -f "$SCRIPT_DIR/lidar_configs/Livox_Mid360.json" ]]; then
    echo "[launch] generating MID-360 lidar config..."
    python3 "$SCRIPT_DIR/gen_mid360_config.py"
fi

# Isaac 5.1's IsaacSensorCreateRtxLidar resolves custom JSON profiles by name
# from the lidar config search folders. Runtime registration races the lidar
# core plugin, so deterministically expose our profile by symlinking it into a
# folder that is ALREADY on the default search path.
ISAAC_LIDAR_CFG_DIR="$ISAAC_DIR/exts/isaacsim.sensors.rtx/data/lidar_configs"
if [[ -d "$ISAAC_LIDAR_CFG_DIR" ]]; then
    ln -sf "$SCRIPT_DIR/lidar_configs/Livox_Mid360.json" \
           "$ISAAC_LIDAR_CFG_DIR/Livox_Mid360.json" 2>/dev/null \
        && echo "[launch] MID-360 profile linked into Isaac config path" \
        || echo "[launch] WARNING: could not link MID-360 profile into $ISAAC_LIDAR_CFG_DIR" >&2
fi

# Reference-command teleop for --stabilize. Isaac runs in the foreground below,
# so the gamepad nodes (non-interactive) are backgrounded in the localization
# container; the keyboard (needs its own TTY) is left for the user to run.
case "${TELEOP}" in
    joystick)
        LOC_SVC="${LOCALIZATION_CONTAINER:-localization}"
        echo "[launch] --teleop=joystick: starting gamepad nodes in '${LOC_SVC}' ..."
        docker exec -d "${LOC_SVC}" bash -lc \
            'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && \
             exec ros2 launch g1_sim_bridge joystick.launch.py' \
            && echo "[launch] joystick teleop running (stop: docker exec ${LOC_SVC} pkill -f joy)" \
            || echo "[launch] WARNING: could not start joystick in '${LOC_SVC}'" >&2
        ;;
    keyboard)
        echo "[launch] --teleop=keyboard: in the localization container run ->"
        echo "[launch]   ros2 run teleop_twist_keyboard teleop_twist_keyboard"
        ;;
    ""|none) : ;;
    *) echo "[launch] WARNING: unknown --teleop='${TELEOP}' (use keyboard|joystick|none)" >&2 ;;
esac

echo "[launch] ROS_DISTRO=$ROS_DISTRO  RMW=$RMW_IMPLEMENTATION  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[launch] starting Isaac Sim (G1 + MID-360 + IMU -> ROS 2)..."
exec "$ISAAC_DIR/python.sh" "$SCENE" "${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}"
