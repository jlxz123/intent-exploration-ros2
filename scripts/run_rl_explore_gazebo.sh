#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rl_explore_env.sh"

MAP_NAME="${RL_EXPLORE_MAP_NAME:-train_map_l1}"
SPAWN_BASE_SEED="${RL_EXPLORE_SPAWN_BASE_SEED:-1}"
SPAWN_EVAL_INDEX="${RL_EXPLORE_SPAWN_EVAL_INDEX:-0}"

cd "${RL_EXPLORE_WORKSPACE}"
echo "[rl_explore] Starting Gazebo + Cartographer"
echo "[rl_explore] map_name=${MAP_NAME}"
echo "[rl_explore] spawn_base_seed=${SPAWN_BASE_SEED}"
echo "[rl_explore] spawn_eval_index=${SPAWN_EVAL_INDEX}"
exec ros2 launch rl_explore_gazebo rl_explore_gazebo.launch.py map_name:="${MAP_NAME}" spawn_base_seed:="${SPAWN_BASE_SEED}" spawn_eval_index:="${SPAWN_EVAL_INDEX}"
