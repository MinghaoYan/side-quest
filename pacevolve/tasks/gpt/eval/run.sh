#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# --- Script to run GPT training ---
#
# Usage:
# ./run_training.sh <conda_env> '<train_files>' '<val_files>' <use_compile_flag>
#
# Example (with compile flag):
# ./run_training.sh nano 'data/*.bin' 'data/*.bin' true
#
# Example (without compile flag):
# ./run_training.sh nano 'data/*.bin' 'data/*.bin' false

# 1. Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <conda_env_name> '<train_files_pattern>' '<val_files_pattern>' [true|false]"
    exit 1
fi

# 2. Assign arguments to variables
CONDA_ENV_NAME=$1
TRAIN_FILES=$2
VAL_FILES=$3
USE_COMPILE=$4

# 3. Set up environment variables
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 4. Initialize Conda for shell scripts
source /opt/conda/etc/profile.d/conda.sh

# 5. Activate the specified Conda environment
conda activate "$CONDA_ENV_NAME"

# 6. Conditionally set the compile flag argument
COMPILE_FLAG_ARG="" # Default to empty
if [ "$USE_COMPILE" = "true" ]; then
    COMPILE_FLAG_ARG="-c"
fi

nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# 7. Run the training command
torchrun --standalone --nproc_per_node=16 "$SCRIPT_DIR/../src/train_gpt.py" \
    $COMPILE_FLAG_ARG \
    --train_files "$TRAIN_FILES" \
    --val_files "$VAL_FILES"