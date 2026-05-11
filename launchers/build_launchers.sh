#!/usr/bin/env bash
set -euo pipefail

LAUNCHERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${LAUNCHERS_DIR}/src/rl_explore_launcher.c"
CC_BIN="${CC:-gcc}"

[[ -f "${SRC}" ]] || { echo "[rl_explore] Missing launcher source: ${SRC}" >&2; exit 1; }

"${CC_BIN}" -O2 -Wall -Wextra -std=c11 \
  -DRL_EXPLORE_LAUNCH_SCRIPT=\"scripts/run_rl_explore_all.sh\" \
  -DRL_EXPLORE_LAUNCHER_NAME=\"RL\ Explore\ Auto\" \
  "${SRC}" -o "${LAUNCHERS_DIR}/RL_Explore_Auto"

"${CC_BIN}" -O2 -Wall -Wextra -std=c11 \
  -DRL_EXPLORE_LAUNCH_SCRIPT=\"scripts/run_rl_explore_with_hand_intent.sh\" \
  -DRL_EXPLORE_LAUNCHER_NAME=\"RL\ Explore\ With\ Hand\ Intent\" \
  "${SRC}" -o "${LAUNCHERS_DIR}/RL_Explore_With_Hand_Intent"

chmod 755 "${LAUNCHERS_DIR}/RL_Explore_Auto" "${LAUNCHERS_DIR}/RL_Explore_With_Hand_Intent"

echo "[rl_explore] Built:"
echo "[rl_explore]   ${LAUNCHERS_DIR}/RL_Explore_Auto"
echo "[rl_explore]   ${LAUNCHERS_DIR}/RL_Explore_With_Hand_Intent"
