#!/bin/bash
# Thin ThetaEvolve/OpenEvolve launcher for KuaRec on DeepSeek-R1-0528-Qwen3-8B.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK="kuairec"
export SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-dpsk_distill_qwen3_8b}"

bash "${ROOT_DIR}/run_thetaevolve_task.sh"
