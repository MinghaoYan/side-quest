#!/bin/bash
# Thin ThetaEvolve/OpenEvolve launcher for MULTI-evolve on Qwen3-4B-Thinking-2507.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK="multievolve_extrapolate"
export SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-qwen3_4b_thinking_2507}"

bash "${ROOT_DIR}/run_thetaevolve_task.sh"
