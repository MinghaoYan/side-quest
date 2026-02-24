#!/bin/bash
# Top-level script for running kernel_bench with SLIME + PACEvolve
#
# Usage: bash run_kernel_bench.sh
# Edit the CONFIGURATION SECTION below before running.

########################### CONFIGURATION SECTION #############################

#### Important: replace SAVE_PATH with your path with enough space ####
export SAVE_PATH="/workspace/logs"

#### Model selection ####
SMALL_MODEL_NAME="dpsk_distill_qwen3_8b"

#### Kernel selection ####
# Choose a kernel config ID (1-16). Maps to pacevolve/tasks/kernel_bench/config/config_{KERNEL_CONFIG_ID}.yaml
KERNEL_CONFIG_ID=1

#### Training mode: True for training, False for inference-only ####
IS_TRAINING=True

#### Training parameters ####
REWARD_PROCESS_TYPE="original_reward"

#### Random seed ####
SEED=3407

#### PACEvolve workflow parameters (exported as env vars for general_pacevolve.sh) ####
export PACEVOLVE_BACKTRACK_FREQ=-1
export PACEVOLVE_BACKTRACK_LEN=5
export PACEVOLVE_POWER_ALPHA=1.5
export PACEVOLVE_IDEA_CAP=5
export PACEVOLVE_MERGE_FREQ=1
export PACEVOLVE_SUMMARIZE_FREQ=20

#### Additional note for file names ####
NOTE=""

#### Replace with your own wandb settings ####
WANDB_API_KEY=aaa
WANDB_ENTITY=bbb
WANDB_PROJECT=ccc

########################## END CONFIGURATION SECTION ##########################

POSTFIX_STR="_seed${SEED}${NOTE}"

if [ "$SMALL_MODEL_NAME" = "dpsk_distill_qwen3_8b" ]; then
    MODEL_FAMILY="deepseek-ai"
    MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
    models_file_name="qwen3-8B.sh"
else
    echo "Unknown SMALL_MODEL_NAME: $SMALL_MODEL_NAME"
    exit 1
fi
echo "Using model: $MODEL_NAME"

# Path configuration
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tmp
mkdir -p $SAVE_PATH/hf
mkdir -p $SAVE_PATH/wandb
mkdir -p $SAVE_PATH/shm
mkdir -p $SAVE_PATH/triton

# Setup paths
export TMPDIR=/tmp
export HF_HOME=$SAVE_PATH/hf
export HUGGINGFACE_HUB_CACHE=$SAVE_PATH/hf/hub
export TRANSFORMERS_CACHE=$SAVE_PATH/hf/hub
export HF_DATASETS_CACHE=$SAVE_PATH/hf/datasets
export SAVE_SHM_DIR=$SAVE_PATH/shm
export TRITON_CACHE_DIR=$SAVE_PATH/triton

# wandb
export WANDB_CACHE_DIR=$SAVE_PATH/wandb
export WANDB_DIR=$SAVE_PATH/wandb
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_ENTITY=$WANDB_ENTITY
export WANDB_PROJECT=$WANDB_PROJECT

########################## AUTO-GENERATED PATHS #############################

# PACEvolve kernel_bench config
CONFIG_YAML="pacevolve/tasks/kernel_bench/config/config_${KERNEL_CONFIG_ID}.yaml"

# Extract kernel name from config for the run name
KERNEL_NAME=$(python3 -c "
import yaml
with open('${CONFIG_YAML}') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['sota_algo_name'])
" 2>/dev/null || echo "kernel${KERNEL_CONFIG_ID}")

RUN_NAME="${SMALL_MODEL_NAME}_pacevolve_kernel_bench_${KERNEL_NAME}${POSTFIX_STR}"

########################## MODEL SETUP #############################

FORCE_DOWNLOAD=0
if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ] && [ -f "$SAVE_SHM_DIR/$MODEL_NAME/config.json" ] && [ $FORCE_DOWNLOAD -eq 0 ]; then
    echo "Model $MODEL_NAME already exists at $SAVE_SHM_DIR/$MODEL_NAME, skipping download"
else
    if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ]; then
        echo "Incomplete model directory found at $SAVE_SHM_DIR/$MODEL_NAME, deleting and re-downloading"
        rm -rf "$SAVE_SHM_DIR/$MODEL_NAME"
    fi
    echo "Downloading model $MODEL_NAME to current directory first..."
    hf download $MODEL_FAMILY/$MODEL_NAME --local-dir ./$MODEL_NAME

    mkdir -p $SAVE_SHM_DIR
    echo "copy model from ./$MODEL_NAME to $SAVE_SHM_DIR/$MODEL_NAME"
    cp -r $MODEL_NAME $SAVE_SHM_DIR/
    echo "Model download and move completed"
fi

source scripts/models/${models_file_name}
if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ] || [ $FORCE_DOWNLOAD -eq 1 ]; then
    echo "Converting HF model to torch dist format..."
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} --hf-checkpoint $SAVE_SHM_DIR/$MODEL_NAME --save $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist
    echo "Conversion completed, torch dist model saved at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist"
else
    echo "Torch dist model already exists at $SAVE_SHM_DIR/${MODEL_NAME}_torch_dist, skipping conversion"
fi

########################## MAIN EXECUTION #############################

echo "=== Kernel Bench + PACEvolve Configuration ==="
echo "RUN_NAME:     ${RUN_NAME}"
echo "KERNEL_NAME:  ${KERNEL_NAME}"
echo "CONFIG_YAML:  ${CONFIG_YAML}"
echo "SAVE_PATH:    ${SAVE_PATH}"
echo "SEED:         ${SEED}"
echo "==============================================="

mkdir -p "${SAVE_PATH}/${RUN_NAME}"

bash scripts_evolve/${MODEL_NAME}/general_pacevolve.sh \
    "${WANDB_PROJECT}" \
    "${RUN_NAME}" \
    "${CONFIG_YAML}" \
    "${SAVE_PATH}" \
    "${IS_TRAINING}" \
    "${REWARD_PROCESS_TYPE}" \
    "${SEED}" \
    2>&1 | tee -a "${SAVE_PATH}/${RUN_NAME}/train_log.txt"
