#!/bin/bash
# Top-level script for running EPLB with SLIME + PACEvolve.
#
# Usage: bash run_eplb.sh
# Edit the CONFIGURATION SECTION below before running.

########################### CONFIGURATION SECTION #############################

#### Important: replace SAVE_PATH with your path with enough space ####
export SAVE_PATH="/workspace/logs"

#### Model selection ####
SMALL_MODEL_NAME="dpsk_distill_qwen3_8b"

#### EPLB task config selection ####
# Maps to pacevolve/tasks/eplb/config/config_{EPLB_CONFIG_ID}.yaml
EPLB_CONFIG_ID=1

#### Fast iteration defaults ####
# Set True for full training, False for fast rollout/debug iteration.
IS_TRAINING=True
EPLB_MAX_ITERS=80
EPLB_EVAL_TIMEOUT=600
EPLB_RECOMPILE_TIMEOUT=120

#### Training parameters ####
REWARD_PROCESS_TYPE="rl_normalized_reward"

#### PKPO-specific parameters (passed to general_pacevolve.sh via env vars) ####
export PKPO_K=4
export PKPO_ESTIMATOR_TYPE="sloo_minus_one"

#### PACEvolve workflow parameters ####
export PACEVOLVE_BACKTRACK_FREQ=-1
export PACEVOLVE_BACKTRACK_LEN=5
export PACEVOLVE_POWER_ALPHA=1.5
export PACEVOLVE_IDEA_CAP=5
export PACEVOLVE_MERGE_FREQ=1
export PACEVOLVE_SUMMARIZE_FREQ=20

#### Rollout overrides for quicker iterations ####
export PACEVOLVE_NUM_ROLLOUT=25600
export PACEVOLVE_ROLLOUT_BATCH_SIZE=2
export PACEVOLVE_N_SAMPLES_PER_PROMPT=8
export PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN=8192
export PACEVOLVE_MAX_CONCURRENT_EVALS=16

#### Random seed ####
SEED=3407

#### Additional note for file names ####
NOTE="_eplb"

#### Dataset location ####
# Must contain expert-load.json (download from Hugging Face - see pacevolve/tasks/eplb/README.md)
EPLB_DATA_PATH="${EPLB_DATA_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pacevolve/tasks/eplb/data}"

#### API key for PACEvolve gym's API model (Gemini/OpenAI/Anthropic) ####
# export GOOGLE_API_KEY="your_gemini_key"

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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="${ROOT_DIR}/pacevolve/tasks/eplb"
BASE_CONFIG_YAML="${TASK_ROOT}/config/config_${EPLB_CONFIG_ID}.yaml"

if [ ! -f "${BASE_CONFIG_YAML}" ]; then
    echo "Missing EPLB base config: ${BASE_CONFIG_YAML}"
    exit 1
fi

if [ ! -f "${EPLB_DATA_PATH}/expert-load.json" ]; then
    echo "EPLB workload file not found under: ${EPLB_DATA_PATH}"
    echo "Expected file: expert-load.json"
    echo "Download from: wget https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json"
    echo "Then place it in ${EPLB_DATA_PATH}/"
    echo "Or set EPLB_DATA_PATH=/path/to/eplb/data and rerun."
    exit 1
fi

RUN_NAME="${SMALL_MODEL_NAME}_pacevolve_eplb_cfg${EPLB_CONFIG_ID}${POSTFIX_STR}"

# Path configuration
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tmp
mkdir -p $SAVE_PATH/hf
mkdir -p $SAVE_PATH/wandb
mkdir -p $SAVE_PATH/shm
mkdir -p $SAVE_PATH/triton
mkdir -p "${SAVE_PATH}/${RUN_NAME}"

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

# Generate runtime config with task-local paths and fast-iteration overrides.
RUNTIME_CONFIG_YAML="${SAVE_PATH}/${RUN_NAME}/config_eplb_runtime.yaml"
python3 - "${BASE_CONFIG_YAML}" "${RUNTIME_CONFIG_YAML}" "${TASK_ROOT}" "${EPLB_DATA_PATH}" "${EPLB_MAX_ITERS}" "${EPLB_EVAL_TIMEOUT}" "${EPLB_RECOMPILE_TIMEOUT}" "${PACEVOLVE_N_SAMPLES_PER_PROMPT}" <<'PY'
import os
import sys
import yaml

(
    base_cfg,
    out_cfg,
    task_root,
    data_path,
    max_iters,
    eval_timeout,
    recompile_timeout,
    n_samples_per_prompt,
) = sys.argv[1:]
with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

cfg["paths"]["data_path"] = os.path.abspath(data_path)
cfg["paths"]["src_path"] = os.path.join(task_root, "src")
cfg["paths"]["eval_path"] = os.path.join(task_root, "eval")
cfg["paths"]["results_path"] = os.path.join(task_root, "results")
cfg["paths"]["log_dir"] = os.path.join(task_root, "verbose")
cfg["paths"]["transcript_dir"] = os.path.join(task_root, "transcripts")

cfg["experiment"]["max_iters"] = int(max_iters)
cfg["evaluation"]["eval_timeout"] = int(eval_timeout)
cfg["compilation"]["recompile_timeout"] = int(recompile_timeout)
cfg.setdefault("database", {})
cfg["database"]["num_islands"] = int(n_samples_per_prompt)

os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
with open(out_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

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

echo "=== EPLB + PACEvolve Configuration ==="
echo "RUN_NAME:     ${RUN_NAME}"
echo "CONFIG_YAML:  ${RUNTIME_CONFIG_YAML}"
echo "DATA_PATH:    ${EPLB_DATA_PATH}"
echo "SAVE_PATH:    ${SAVE_PATH}"
echo "SEED:         ${SEED}"
echo "IS_TRAINING:  ${IS_TRAINING}"
echo "MAX_ITERS:    ${EPLB_MAX_ITERS}"
echo "N_SAMPLES:    ${PACEVOLVE_N_SAMPLES_PER_PROMPT}"
echo "PKPO_K:       ${PKPO_K}"
echo "PKPO_ESTIMATOR_TYPE: ${PKPO_ESTIMATOR_TYPE}"
echo "========================================"

bash scripts_evolve/${MODEL_NAME}/general_pacevolve.sh \
    "${WANDB_PROJECT}" \
    "${RUN_NAME}" \
    "${RUNTIME_CONFIG_YAML}" \
    "${SAVE_PATH}" \
    "${IS_TRAINING}" \
    "${REWARD_PROCESS_TYPE}" \
    "${SEED}" \
    2>&1 | tee -a "${SAVE_PATH}/${RUN_NAME}/train_log.txt"
