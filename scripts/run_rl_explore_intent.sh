#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rl_explore_env.sh"

echo "[rl_explore] Starting intent direction node"
echo "[rl_explore] topic=/rl_explore/intent_direction"
echo
echo "[rl_explore] Input directions:"
echo "[rl_explore]   q/front_left   w/front        e/front_right"
echo "[rl_explore]   a/left         s/clear        d/right"
echo "[rl_explore]   z/back_left    x/back         c/back_right"
echo
exec ros2 run rl_explore_policy intent_node
