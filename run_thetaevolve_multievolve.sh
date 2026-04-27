#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK="multievolve_extrapolate"

bash "${ROOT_DIR}/run_thetaevolve_task.sh"
