#!/usr/bin/env bash
# Autonomous navigation launcher: bring up BOTH the perception/localization stack
# and the A*+MPC planner with one command, as two SEPARATE processes.
#
# Each `ros2 launch` is its own process tree and runs all of its nodes
# concurrently (ROS 2 launch already manages that — no manual threading needed),
# so this just starts two launch files and ties their lifetimes together: Ctrl-C
# tears BOTH down cleanly (the plain `cmd & cmd` form would orphan the first).
#
# This is the all-in-one alternative to starting the two launch files by hand
# (see README "Autonomous navigation"). It runs in the ROS 2 / localization
# container. The AMO gait still runs separately in the amo_policy container:
#
#     AUTONOMOUS=1 NET_IF=<nic> ./docker/run_amo.sh
#
# Env overrides:
#   ROS_DOMAIN_ID=42    DDS domain (default 42; matches both launch files)
#   PLANNER_DELAY=3     seconds to wait after localization before the planner,
#                       so DLIO finishes its IMU/gravity init (hold the robot
#                       STILL during this window). Set 0 to start them together.
set -uo pipefail

# Resolve the workspace root: the nearest ancestor (or the script's own dir) that
# holds install/setup.bash. Works whether this script sits in ros2_ws/ or
# ros2_ws/src/, and whether the workspace is mounted at /ws or anywhere else.
HERE="$(cd "$(dirname "$0")" && pwd)"
WS=""
d="${HERE}"
while [[ "${d}" != "/" ]]; do
    if [[ -f "${d}/install/setup.bash" ]]; then WS="${d}"; break; fi
    d="$(dirname "${d}")"
done
if [[ -z "${WS}" ]]; then
    echo "error: no install/setup.bash found at or above ${HERE} — build the workspace first" >&2
    echo "       (inside the localization container: build_ws)" >&2
    exit 1
fi
cd "${WS}"
# Disable nounset around the ROS/colcon sourcing: install/setup.bash references
# unset vars (COLCON_TRACE, AMENT_TRACE, …) and is not `set -u`-safe.
set +u
# shellcheck disable=SC1091
source install/setup.bash
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
PLANNER_DELAY="${PLANNER_DELAY:-3}"

pids=()
cleanup() {
    trap - INT TERM EXIT
    echo ""
    echo ">> stopping localization + planner ..."
    # SIGINT lets each `ros2 launch` shut its own nodes down gracefully.
    kill -INT "${pids[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

echo ">> [1/2] localization (DLIO + g1_local_map) on ROS_DOMAIN_ID=${ROS_DOMAIN_ID} ..."
ros2 launch g1_bringup real_localization.launch.py &
pids+=($!)

if (( PLANNER_DELAY > 0 )); then
    echo ">> waiting ${PLANNER_DELAY}s for DLIO IMU/gravity init — keep the robot STILL ..."
    sleep "${PLANNER_DELAY}"
fi

echo ">> [2/2] A*+MPC planner (+ cmd_vel -> AMO WS bridge) ..."
ros2 launch a_star_mpc_planner planner.launch.py &
pids+=($!)

echo ">> both launches running. Start the gait:  AUTONOMOUS=1 NET_IF=<nic> ./docker/run_amo.sh"
echo ">> then set a goal in RViz (2D Goal Pose -> /global_goal). Ctrl-C stops both."

# Exit (and clean up) as soon as EITHER launch dies, so a crash never leaves the
# other half running silently.
wait -n 2>/dev/null || wait
cleanup
