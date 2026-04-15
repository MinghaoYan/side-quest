#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PACEVOLVE_MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"
export PACEVOLVE_MODEL_ARGS_SCRIPT="scripts/models/deepseek-r1-distill-qwen-1.5B.sh"
export PACEVOLVE_SGLANG_MEM_FRACTION_STATIC="${PACEVOLVE_SGLANG_MEM_FRACTION_STATIC:-0.65}"

bash "${ROOT_DIR}/common_pacevolve.sh" "$@"
