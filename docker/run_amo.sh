#!/usr/bin/env bash
# Run the AMO policy inference (amo/amo_inference.py) inside the amo_policy
# container via docker compose.
#
# All extra args are forwarded to amo_inference.py, e.g.:
#   ./run_amo.sh                         # uses amo_g1.yaml as-is (source: zero -> stands)
#   ./run_amo.sh --observe_only          # dry run: subscribe to DDS, never publish
#   ./run_amo.sh --vx 0.3                # constant forward velocity (overrides config)
#   ./run_amo.sh --filter critdamp       # enable the running joint filter
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

echo ">> running AMO inference on ${SERVICE} (NIC=${UNITREE_NET_IFACE}, config=${CONFIG})"
echo ">> forwarded args: $*"

# --rm: ephemeral container; --service-ports: publish the WS command port (8766)
# so an MPC planner can reach it when command.source=websocket.
exec "${DC[@]}" run --rm --service-ports "${SERVICE}" \
    python /workspace/amo/amo_inference.py --config "${CONFIG}" "$@"
