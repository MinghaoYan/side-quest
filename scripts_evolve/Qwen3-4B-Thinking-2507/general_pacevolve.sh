#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PACEVOLVE_MODEL_NAME="Qwen3-4B-Thinking-2507"
export PACEVOLVE_MODEL_ARGS_SCRIPT="scripts/models/qwen3-4B-Thinking-2507.sh"
export PACEVOLVE_SGLANG_MEM_FRACTION_STATIC="${PACEVOLVE_SGLANG_MEM_FRACTION_STATIC:-0.55}"

bash "${ROOT_DIR}/common_pacevolve.sh" "$@"
