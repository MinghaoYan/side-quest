#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PACEVOLVE_MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
export PACEVOLVE_MODEL_ARGS_SCRIPT="scripts/models/qwen3-8B.sh"
export PACEVOLVE_SGLANG_MEM_FRACTION_STATIC="${PACEVOLVE_SGLANG_MEM_FRACTION_STATIC:-0.4}"

bash "${ROOT_DIR}/common_pacevolve.sh" "$@"
