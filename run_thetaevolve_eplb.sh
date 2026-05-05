#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK="eplb"
export SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-dpsk_distill_qwen3_8b}"

if [ "${SMALL_MODEL_NAME}" = "dpsk_distill_qwen3_8b" ]; then
  bash "${ROOT_DIR}/run_thetaevolve_eplb_dpsk_distill_qwen3_8b.sh"
else
  bash "${ROOT_DIR}/run_thetaevolve_task.sh"
fi
