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

# ── Separate logs for localization vs planner ────────────────────────────────
# Both launches used to share this terminal, so DLIO/g1_local_map and the
# A*+MPC planner logs interleaved. Now each launch's stdout+stderr goes to its
# OWN file so you can read them independently:
#     tail -f logs/localization_latest.log     # DLIO + g1_local_map
#     tail -f logs/planner_latest.log          # A* node + MPC node (+ bridge)
# Override the directory with LOG_DIR=/path ./autonomy.sh. Set LOG_TO_CONSOLE=1
# to ALSO mirror both streams to this terminal (they will interleave again).
LOG_DIR="${LOG_DIR:-${WS}/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCALIZATION_LOG="${LOG_DIR}/localization_${TS}.log"
PLANNER_LOG="${LOG_DIR}/planner_${TS}.log"
# Stable "latest" symlinks so you can tail without knowing the timestamp.
ln -sfn "$(basename "${LOCALIZATION_LOG}")" "${LOG_DIR}/localization_latest.log"
ln -sfn "$(basename "${PLANNER_LOG}")"      "${LOG_DIR}/planner_latest.log"
LOG_TO_CONSOLE="${LOG_TO_CONSOLE:-0}"

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

# Launch a `ros2 launch`, sending its output to a dedicated logfile (and, when
# LOG_TO_CONSOLE=1, also to this terminal via tee). Records the launch PID — NOT
# tee's — so cleanup signals the launch directly.
run_launch() {
    local logfile="$1"; shift
    if [[ "${LOG_TO_CONSOLE}" == "1" ]]; then
        "$@" > >(tee -a "${logfile}") 2>&1 &
    else
        "$@" > "${logfile}" 2>&1 &
    fi
    pids+=($!)
}

echo ">> [1/2] localization (DLIO + g1_local_map) on ROS_DOMAIN_ID=${ROS_DOMAIN_ID} ..."
echo ">>       logs -> ${LOCALIZATION_LOG}"
run_launch "${LOCALIZATION_LOG}" ros2 launch g1_bringup real_localization.launch.py

if (( PLANNER_DELAY > 0 )); then
    echo ">> waiting ${PLANNER_DELAY}s for DLIO IMU/gravity init — keep the robot STILL ..."
    sleep "${PLANNER_DELAY}"
fi

echo ">> [2/2] A*+MPC planner (+ cmd_vel -> AMO WS bridge) ..."
echo ">>       logs -> ${PLANNER_LOG}"
run_launch "${PLANNER_LOG}" ros2 launch a_star_mpc_planner planner.launch.py

echo ""
echo ">> both launches running. Read their logs SEPARATELY (each in its own terminal):"
echo ">>     tail -f ${LOG_DIR}/localization_latest.log"
echo ">>     tail -f ${LOG_DIR}/planner_latest.log"
echo ">> Start the gait:  AUTONOMOUS=1 NET_IF=<nic> ./docker/run_amo.sh"
echo ">> then set a goal in RViz (2D Goal Pose -> /global_goal)."

# ── Foreground keyboard e-stop ───────────────────────────────────────────────
# This owns the terminal's stdin (the two launches stream to logfiles), so you
# can SAFELY stop and later re-enable navigation without killing anything:
#     s + Enter  ->  STOP  (zero velocity to the AMO gait; robot holds)
#     g + Enter  ->  GO    (resume navigation)
#     q + Enter  ->  quit autonomy.sh (stops everything)
# DISABLE_ESTOP_KEYS=1 falls back to the old "Ctrl-C only / wait" behaviour.
#
# If a launch itself crashes, the velocity command to the gait still goes to
# ZERO automatically — the MPC fail-safe (stale pose/path) and the bridge
# cmd_vel watchdog both zero it — so a crash stops the robot even before you
# press q / Ctrl-C here.
echo ""
if [[ "${DISABLE_ESTOP_KEYS:-0}" == "1" ]]; then
    echo ">> e-stop keys disabled — Ctrl-C stops everything."
    wait -n 2>/dev/null || wait
else
    echo ">> SAFETY E-STOP active in THIS terminal:  s=stop  g=go  q=quit"
    ros2 run g1_sim_bridge estop_keyboard_node
fi
cleanup
