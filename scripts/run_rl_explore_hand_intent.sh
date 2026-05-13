#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rl_explore_env.sh"

CAMERA_INDEX="${RL_EXPLORE_HAND_CAMERA_INDEX:-0}"
SHOW_WINDOW="${RL_EXPLORE_HAND_SHOW_WINDOW:-true}"
IMAGE_TOPIC="${RL_EXPLORE_HAND_GUI_IMAGE_TOPIC:-}"
MODEL_PATH="${RL_EXPLORE_HAND_MODEL_PATH:-/home/zbf/achievement/graduate/hand_dir_detect/hand_landmarker.task}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rl_explore_mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

echo "[rl_explore] Starting hand intent node"
echo "[rl_explore] camera_index=${CAMERA_INDEX}"
echo "[rl_explore] show_window=${SHOW_WINDOW}"
echo "[rl_explore] image_topic=${IMAGE_TOPIC}"
echo "[rl_explore] model_path=${MODEL_PATH}"
echo "[rl_explore] topic=/rl_explore/intent_direction"
echo
echo "[rl_explore] Finger direction mapping:"
echo "[rl_explore]   up        -> front        -> 0"
echo "[rl_explore]   up-right   -> front_right  -> 1"
echo "[rl_explore]   right      -> right        -> 2"
echo "[rl_explore]   down-right -> back_right   -> 3"
echo "[rl_explore]   down       -> back         -> 4"
echo "[rl_explore]   down-left  -> back_left    -> 5"
echo "[rl_explore]   left       -> left         -> 6"
echo "[rl_explore]   up-left    -> front_left   -> 7"
echo
exec ros2 launch rl_explore_policy rl_explore_hand_intent.launch.py \
  camera_index:="${CAMERA_INDEX}" \
  show_window:="${SHOW_WINDOW}" \
  image_topic:="${IMAGE_TOPIC}" \
  model_path:="${MODEL_PATH}"
