#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/rl_explore_env.sh"

DEVICE="${RL_EXPLORE_DEVICE:-cuda}"
MAP_NAME="${RL_EXPLORE_MAP_NAME:-train_map_l1}"
POLICY_DELAY="${RL_EXPLORE_POLICY_DELAY:-12}"
SPAWN_BASE_SEED="${RL_EXPLORE_SPAWN_BASE_SEED:-1}"
SPAWN_EVAL_INDEX="${RL_EXPLORE_SPAWN_EVAL_INDEX:-0}"

cd "${RL_EXPLORE_WORKSPACE}"

echo "[rl_explore] Checking PyTorch in venv"
python - <<'PY'
import os
import torch

device = os.environ.get("RL_EXPLORE_DEVICE", "cuda")
print(f"[rl_explore] torch={torch.__version__}")
print(f"[rl_explore] torch_cuda_runtime={torch.version.cuda}")
print(f"[rl_explore] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[rl_explore] gpu={torch.cuda.get_device_name(0)}")
elif device.startswith("cuda"):
    print("[rl_explore] WARNING: CUDA requested, but torch reports CUDA unavailable. The policy node will fall back to CPU.")
PY

if [[ "${POLICY_DELAY}" != "0" ]]; then
    echo "[rl_explore] Waiting ${POLICY_DELAY}s before starting policy node"
    sleep "${POLICY_DELAY}"
fi

echo "[rl_explore] Starting policy node"
echo "[rl_explore] device=${DEVICE}"
echo "[rl_explore] world_name=${MAP_NAME}"
echo "[rl_explore] spawn_base_seed=${SPAWN_BASE_SEED}"
echo "[rl_explore] spawn_eval_index=${SPAWN_EVAL_INDEX}"
exec ros2 launch rl_explore_policy rl_explore_policy.launch.py device:="${DEVICE}" world_name:="${MAP_NAME}" spawn_base_seed:="${SPAWN_BASE_SEED}" spawn_eval_index:="${SPAWN_EVAL_INDEX}"
