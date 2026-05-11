#!/usr/bin/env bash

_rl_explore_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_rl_explore_default_workspace="$(cd "${_rl_explore_script_dir}/.." && pwd)"
_rl_explore_default_venv="$(cd "${_rl_explore_default_workspace}/.." && pwd)/ros_torch_cuda_venv"

export RL_EXPLORE_WORKSPACE="${RL_EXPLORE_WORKSPACE:-${_rl_explore_default_workspace}}"
export RL_EXPLORE_VENV="${RL_EXPLORE_VENV:-${_rl_explore_default_venv}}"

_rl_explore_fail() {
    echo "[rl_explore] $*" >&2
    return 1 2>/dev/null || exit 1
}

[[ -f /opt/ros/humble/setup.bash ]] || _rl_explore_fail "Missing /opt/ros/humble/setup.bash"
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH

case $- in
    *u*) _rl_explore_had_nounset=1 ;;
    *) _rl_explore_had_nounset=0 ;;
esac

set +u
source /opt/ros/humble/setup.bash

[[ -f "${RL_EXPLORE_VENV}/bin/activate" ]] || _rl_explore_fail "Missing venv: ${RL_EXPLORE_VENV}"
source "${RL_EXPLORE_VENV}/bin/activate"

[[ -f "${RL_EXPLORE_WORKSPACE}/install/local_setup.bash" ]] || _rl_explore_fail "Workspace is not built: ${RL_EXPLORE_WORKSPACE}"
source "${RL_EXPLORE_WORKSPACE}/install/local_setup.bash"

if [[ "${_rl_explore_had_nounset}" == "1" ]]; then
    set -u
fi
unset _rl_explore_had_nounset

export PYTHONUNBUFFERED=1

unset _rl_explore_script_dir
unset _rl_explore_default_workspace
unset _rl_explore_default_venv
