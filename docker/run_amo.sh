#!/usr/bin/env bash
# Run the AMO policy inference (amo/amo_inference.py) inside the amo_policy
# container via docker compose.
#
# The AMO gait is a velocity tracker; it never sees the goal. Where the velocity
# command (vx, vy, yaw_rate) comes from is the command source (see MODES).
#
# All extra args are forwarded to amo_inference.py, e.g.:
#   ./run_amo.sh                         # uses amo_g1.yaml as-is (source: zero -> stands)
#   ./run_amo.sh --observe_only          # dry run: subscribe to DDS, never publish
#   ./run_amo.sh --vx 0.3                # constant forward velocity (overrides config)
#   ./run_amo.sh --filter critdamp       # enable the running joint filter
#
# MODES (mutually exclusive; pick one, or neither to use the config's source):
#   AUTONOMOUS=1 ./run_amo.sh            # FULL AUTONOMY: track velocity_target from the
#                                        #   A*+MPC planner over WS :8766 (source=websocket).
#                                        #   The planner (a_star_mpc_planner) must be running
#                                        #   in the ROS 2 / localization container — this
#                                        #   script only starts the gait side.
#   JOYSTICK=1   ./run_amo.sh            # MANUAL: drive with the Unitree G1 pad. Used for
#                                        #   teleop / SLAM-mapping a space WITHOUT autonomy.
#
# Autonomous bring-up (NET_IF set on each, robot held still ~3 s for DLIO init):
#   1) localization:  ros2 launch g1_bringup real_localization.launch.py
#   2) localization:  ros2 launch a_star_mpc_planner planner.launch.py
#   3) amo_policy:     AUTONOMOUS=1 NET_IF=enp12s0 ./run_amo.sh
#   then send a goal in RViz ("2D Goal Pose" -> /global_goal). See
#   docs/A_STAR_MPC_PLANNER.md and docs/system_architecture.md.
#
# Env overrides:
#   NET_IF=eth0      ./run_amo.sh ...    # CycloneDDS NIC to the robot
#   CONFIG=/workspace/config/amo_g1.yaml ./run_amo.sh ...
#   SERVICE=amo_policy ./run_amo.sh ...
#   BUILD=1          ./run_amo.sh ...    # (re)build the image first
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"     # Navigation/docker
cd "${HERE}"

SERVICE="${SERVICE:-amo_policy}"
CONFIG="${CONFIG:-/workspace/config/amo_g1.yaml}"

# NIC the robot is on. Exported so docker-compose.yml's
# UNITREE_NET_IFACE:-lo default is overridden and the entrypoint binds
# CycloneDDS (and the policy's DDS init) to the same interface.
export UNITREE_NET_IFACE="${NET_IF:-${UNITREE_NET_IFACE:-eth0}}"

# Pick `docker compose` (v2) or fall back to `docker-compose` (v1).
if docker compose version >/dev/null 2>&1; then
    DC=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    DC=(docker-compose)
else
    echo "error: neither 'docker compose' nor 'docker-compose' is available" >&2
    exit 1
fi

if [[ "${BUILD:-0}" == "1" ]]; then
    echo ">> building ${SERVICE} image ..."
    "${DC[@]}" build "${SERVICE}"
fi

# Command-source selection. AUTONOMOUS and JOYSTICK are mutually exclusive;
# with neither set, amo_inference uses command.source from amo_g1.yaml.
if [[ "${AUTONOMOUS:-0}" == "1" && "${JOYSTICK:-0}" == "1" ]]; then
    echo "error: set only one of AUTONOMOUS=1 or JOYSTICK=1, not both" >&2
    exit 1
fi

AMO_EXTRA=()
if [[ "${AUTONOMOUS:-0}" == "1" ]]; then
    # AUTONOMOUS=1: track the A*+MPC planner's velocity_target over the WS
    # command server (:8766, published by --service-ports below). The planner's
    # mpc_node emits a Twist on /mpc/cmd_vel; g1_sim_bridge/cmd_vel_to_amo_node
    # forwards it as {vx,vy,yaw} JSON to this server. Start the planner stack
    # FIRST (real_localization + planner.launch.py) so a goal can be set.
    echo ">> AUTONOMOUS: tracking A*+MPC velocity_target on WS :8766 ..."
    echo ">>   (planner must be running in the ROS 2 container; set goal in RViz)"
    AMO_EXTRA+=(--command_source websocket)
elif [[ "${JOYSTICK:-0}" == "1" ]]; then
    # JOYSTICK=1: drive the robot with the Unitree G1 gamepad. The pad is
    # delivered by the robot inside LowState.wireless_remote (the same DDS
    # LowState the policy already subscribes to), so amo_inference reads it
    # in-process -- no ROS /joy node, no rt/wireless_controller topic, no
    # websocket bridge. Hold the deadman button (default R1) to move; release to
    # stop. Used for teleop / SLAM-mapping a space, NOT autonomous navigation.
    echo ">> JOYSTICK: G1 pad via LowState.wireless_remote (hold R1 to move) ..."
    AMO_EXTRA+=(--command_source joystick)
fi

echo ">> running AMO inference on ${SERVICE} (NIC=${UNITREE_NET_IFACE}, config=${CONFIG})"
echo ">> forwarded args: ${AMO_EXTRA[*]} $*"

# --rm: ephemeral container; --service-ports: publish the WS command port (8766)
# so an MPC planner / the joystick bridge can reach it when source=websocket.
exec "${DC[@]}" run --rm --service-ports "${SERVICE}" \
    python /workspace/amo/amo_inference.py --config "${CONFIG}" "${AMO_EXTRA[@]}" "$@"
