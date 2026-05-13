#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rl_explore_env.sh"

CAMERA_TOPIC="${RL_EXPLORE_GUI_CAMERA_TOPIC:-/rl_explore/gui/camera_image}"
MAP_TOPIC="${RL_EXPLORE_GUI_MAP_TOPIC:-/map}"

echo "[rl_explore] Starting intent exploration GUI"
echo "[rl_explore] camera_topic=${CAMERA_TOPIC}"
echo "[rl_explore] map_topic=${MAP_TOPIC}"

exec ros2 run rl_explore_policy intent_gui_node \
  --ros-args \
  -p camera_topic:="${CAMERA_TOPIC}" \
  -p map_topic:="${MAP_TOPIC}"
