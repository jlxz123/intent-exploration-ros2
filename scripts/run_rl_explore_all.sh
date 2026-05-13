#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${RL_EXPLORE_WORKSPACE:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MAP_NAME="${RL_EXPLORE_MAP_NAME:-}"
DEVICE="${RL_EXPLORE_DEVICE:-cuda}"
POLICY_DELAY="${RL_EXPLORE_POLICY_DELAY:-12}"
SPAWN_BASE_SEED="${RL_EXPLORE_SPAWN_BASE_SEED:-1}"
SPAWN_EVAL_INDEX="${RL_EXPLORE_SPAWN_EVAL_INDEX:-0}"
DRY_RUN=0
WITH_HAND_INTENT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --with-hand-intent)
            WITH_HAND_INTENT=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -ge 1 ]]; then
    MAP_NAME="$1"
fi

if [[ $# -ge 2 ]]; then
    DEVICE="$2"
fi

choose_map() {
    local maps_dir="${WORKSPACE}/src/rl_explore_gazebo/maps"
    local selected
    local available_maps

    available_maps="$(find "${maps_dir}" -maxdepth 1 -name '*.npy' -printf '%f\n' 2>/dev/null | sed 's/\.npy$//' | sort)"
    if [[ -z "${available_maps}" ]]; then
        echo "[rl_explore] No maps found under: ${maps_dir}" >&2
        exit 1
    fi

    if [[ "${DRY_RUN}" == "0" ]] && command -v zenity >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
        selected="$(printf '%s\n' "${available_maps}" | zenity \
            --list \
            --title="RL Explore Map" \
            --text="Choose a Gazebo map" \
            --column="map_name" \
            --height=360 \
            --width=320)" || exit 0
        [[ -n "${selected}" ]] || exit 0
        MAP_NAME="${selected}"
        return
    fi

    MAP_NAME="$(printf '%s\n' "${available_maps}" | head -n 1)"
}

if [[ -z "${MAP_NAME}" ]]; then
    choose_map
fi

GAZEBO_SCRIPT="${WORKSPACE}/scripts/run_rl_explore_gazebo.sh"
POLICY_SCRIPT="${WORKSPACE}/scripts/run_rl_explore_policy.sh"
HAND_INTENT_SCRIPT="${WORKSPACE}/scripts/run_rl_explore_hand_intent.sh"
INTENT_GUI_SCRIPT="${WORKSPACE}/scripts/run_rl_explore_intent_gui.sh"

[[ -x "${GAZEBO_SCRIPT}" ]] || { echo "[rl_explore] Missing executable: ${GAZEBO_SCRIPT}" >&2; exit 1; }
[[ -x "${POLICY_SCRIPT}" ]] || { echo "[rl_explore] Missing executable: ${POLICY_SCRIPT}" >&2; exit 1; }
if [[ "${WITH_HAND_INTENT}" == "1" ]]; then
    [[ -x "${HAND_INTENT_SCRIPT}" ]] || { echo "[rl_explore] Missing executable: ${HAND_INTENT_SCRIPT}" >&2; exit 1; }
    [[ -x "${INTENT_GUI_SCRIPT}" ]] || { echo "[rl_explore] Missing executable: ${INTENT_GUI_SCRIPT}" >&2; exit 1; }
fi

open_terminal() {
    local title="$1"
    local script_path="$2"
    local extra_env="${3:-}"
    local quoted_map
    local quoted_device
    local quoted_delay
    local quoted_spawn_base_seed
    local quoted_spawn_eval_index
    local quoted_script
    local command_env
    local command_text

    printf -v quoted_map "%q" "${MAP_NAME}"
    printf -v quoted_device "%q" "${DEVICE}"
    printf -v quoted_delay "%q" "${POLICY_DELAY}"
    printf -v quoted_spawn_base_seed "%q" "${SPAWN_BASE_SEED}"
    printf -v quoted_spawn_eval_index "%q" "${SPAWN_EVAL_INDEX}"
    printf -v quoted_script "%q" "${script_path}"

    command_env="RL_EXPLORE_MAP_NAME=${quoted_map} RL_EXPLORE_DEVICE=${quoted_device} RL_EXPLORE_POLICY_DELAY=${quoted_delay} RL_EXPLORE_SPAWN_BASE_SEED=${quoted_spawn_base_seed} RL_EXPLORE_SPAWN_EVAL_INDEX=${quoted_spawn_eval_index}"
    if [[ -n "${extra_env}" ]]; then
        command_env="${command_env} ${extra_env}"
    fi
    command_text="env ${command_env} ${quoted_script}; status=\$?; echo; echo \"[rl_explore] ${title} exited with status \$status\"; exec bash --noprofile --norc"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "---- ${title} ----"
        echo "${command_text}"
        return 0
    fi

    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="${title}" -- bash --noprofile --norc -c "${command_text}" &
    elif command -v konsole >/dev/null 2>&1; then
        konsole --new-tab -p tabtitle="${title}" -e bash --noprofile --norc -c "${command_text}" &
    elif command -v x-terminal-emulator >/dev/null 2>&1; then
        x-terminal-emulator -T "${title}" -e bash --noprofile --norc -c "${command_text}" &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -T "${title}" -e bash --noprofile --norc -c "${command_text}" &
    else
        echo "[rl_explore] No supported terminal emulator found." >&2
        echo "[rl_explore] Run manually: ${script_path}" >&2
        exit 1
    fi
}

echo "[rl_explore] Launching map=${MAP_NAME}, device=${DEVICE}, policy_delay=${POLICY_DELAY}s, spawn_base_seed=${SPAWN_BASE_SEED}, spawn_eval_index=${SPAWN_EVAL_INDEX}, hand_intent=${WITH_HAND_INTENT}"
open_terminal "RL Explore Gazebo" "${GAZEBO_SCRIPT}"
if [[ "${WITH_HAND_INTENT}" == "1" ]]; then
    open_terminal "RL Explore Policy" "${POLICY_SCRIPT}" "RL_EXPLORE_SHOW_CCRL_MAP_WINDOW=false RL_EXPLORE_MAP_IMAGE_TOPIC="
    open_terminal "RL Explore Hand Intent" "${HAND_INTENT_SCRIPT}" "RL_EXPLORE_HAND_SHOW_WINDOW=false RL_EXPLORE_HAND_GUI_IMAGE_TOPIC=/rl_explore/gui/camera_image"
    open_terminal "RL Explore Intent GUI" "${INTENT_GUI_SCRIPT}"
else
    open_terminal "RL Explore Policy" "${POLICY_SCRIPT}"
fi
