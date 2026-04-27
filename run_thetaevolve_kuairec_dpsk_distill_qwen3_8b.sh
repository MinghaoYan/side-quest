#!/bin/bash
# Thin ThetaEvolve/OpenEvolve launcher for KuaRec on DeepSeek-R1-0528-Qwen3-8B.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TASK="kuairec"
export SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-dpsk_distill_qwen3_8b}"

export THETAEVOLVE_TRAIN_SEQ_LENGTH=2048
export THETAEVOLVE_MAX_POSITION_EMBEDDINGS=2048
export THETAEVOLVE_MAX_TOKENS_PER_GPU=1024
export THETAEVOLVE_LOG_PROBS_MAX_TOKENS_PER_GPU=1024

bash "${ROOT_DIR}/run_thetaevolve_task.sh"
