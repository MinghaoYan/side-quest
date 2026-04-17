#!/bin/bash
# Top-level script for running EPLB with SLIME + PACEvolve.
#
# Usage: bash run_eplb.sh
# Edit the CONFIGURATION SECTION below before running.

########################### CONFIGURATION SECTION #############################

#### Important: replace SAVE_PATH with your path with enough space ####
export SAVE_PATH="${SAVE_PATH:-/workspace/logs}"

#### Model selection ####
SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-dpsk_distill_qwen3_8b}"

#### EPLB task config selection ####
# Maps to pacevolve/tasks/eplb/config/config_{EPLB_CONFIG_ID}.yaml
EPLB_CONFIG_ID="${EPLB_CONFIG_ID:-1}"

#### Fast iteration defaults ####
# Set True for full training, False for fast rollout/debug iteration.
IS_TRAINING="${IS_TRAINING:-True}"
EPLB_MAX_ITERS="${EPLB_MAX_ITERS:-80}"
EPLB_EVAL_TIMEOUT="${EPLB_EVAL_TIMEOUT:-600}"
EPLB_RECOMPILE_TIMEOUT="${EPLB_RECOMPILE_TIMEOUT:-120}"

#### Algorithm selection (ENTROPIC, HYBRID_PKPO_GRPO, PKPO, DR_GRPO, or GRPO) ####
# TTT-Discover uses the entropic objective with adaptive beta.
ADVANTAGE_ESTIMATOR_ALGORITHM="${ADVANTAGE_ESTIMATOR_ALGORITHM:-ENTROPIC}"
export ADVANTAGE_ESTIMATOR_ALGORITHM

#### Algorithm-specific reward processing ####
# Keep evolution workflow unchanged, but allow RL-facing reward transforms to be
# selected per algorithm. By default, entropic uses raw task reward while the
# PPO-style estimators keep the normalized reward shaping.
export ENTROPIC_REWARD_PROCESS_TYPE="${ENTROPIC_REWARD_PROCESS_TYPE:-original_reward}"
export GRPO_REWARD_PROCESS_TYPE="${GRPO_REWARD_PROCESS_TYPE:-rl_normalized_reward}"
export DR_GRPO_REWARD_PROCESS_TYPE="${DR_GRPO_REWARD_PROCESS_TYPE:-rl_normalized_reward}"
export PKPO_REWARD_PROCESS_TYPE="${PKPO_REWARD_PROCESS_TYPE:-rl_normalized_reward}"
export HYBRID_PKPO_GRPO_REWARD_PROCESS_TYPE="${HYBRID_PKPO_GRPO_REWARD_PROCESS_TYPE:-rl_normalized_reward}"

ALG_NORMALIZED=$(printf '%s' "${ADVANTAGE_ESTIMATOR_ALGORITHM}" | tr '[:upper:]' '[:lower:]')
case "${ALG_NORMALIZED}" in
    entropic)
        REWARD_PROCESS_TYPE="${ENTROPIC_REWARD_PROCESS_TYPE}"
        ;;
    grpo)
        REWARD_PROCESS_TYPE="${GRPO_REWARD_PROCESS_TYPE}"
        ;;
    dr_grpo|dr.grpo)
        REWARD_PROCESS_TYPE="${DR_GRPO_REWARD_PROCESS_TYPE}"
        ;;
    pkpo)
        REWARD_PROCESS_TYPE="${PKPO_REWARD_PROCESS_TYPE}"
        ;;
    hybrid_pkpo_grpo)
        REWARD_PROCESS_TYPE="${HYBRID_PKPO_GRPO_REWARD_PROCESS_TYPE}"
        ;;
    *)
        echo "Unsupported ADVANTAGE_ESTIMATOR_ALGORITHM: ${ADVANTAGE_ESTIMATOR_ALGORITHM}"
        exit 1
        ;;
esac
export REWARD_PROCESS_TYPE

#### PKPO-specific parameters (passed to general_pacevolve.sh via env vars) ####
export PKPO_K="${PKPO_K:-4}"
export PKPO_ESTIMATOR_TYPE="${PKPO_ESTIMATOR_TYPE:-sloo_minus_one}"

#### Hybrid PKPO/GRPO parameters ####
# The hybrid mixes normalized scalar advantages:
#   (1 - alpha) * GRPO-side + alpha * PKPO-side
export HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"
export HYBRID_ALPHA_ANNEAL_STEP="${HYBRID_ALPHA_ANNEAL_STEP:-200}"
export HYBRID_ALPHA_ANNEAL_TARGET="${HYBRID_ALPHA_ANNEAL_TARGET:-0.8}"
export HYBRID_GRPO_VARIANT="${HYBRID_GRPO_VARIANT:-grpo}"

#### Entropic-specific parameters (passed to general_pacevolve.sh via env vars) ####
# TTT-main / TTT-Discover uses gamma = ln(2) for the adaptive beta KL budget.
export ENTROPIC_KL_CONSTRAINT="${ENTROPIC_KL_CONSTRAINT:-0.6931471805599453}"

#### GRPO-specific parameters (passed to general_pacevolve.sh via env vars) ####
# Use ADVANTAGE_ESTIMATOR_ALGORITHM=GRPO for vanilla GRPO
# or ADVANTAGE_ESTIMATOR_ALGORITHM=DR_GRPO for the unbiased Dr. GRPO variant.
# Use ADVANTAGE_ESTIMATOR_ALGORITHM=HYBRID_PKPO_GRPO for the blended estimator.

#### PACEvolve workflow parameters ####
export PACEVOLVE_BACKTRACK_FREQ="${PACEVOLVE_BACKTRACK_FREQ:--1}"
export PACEVOLVE_BACKTRACK_LEN="${PACEVOLVE_BACKTRACK_LEN:-5}"
export PACEVOLVE_POWER_ALPHA="${PACEVOLVE_POWER_ALPHA:-1.5}"
export PACEVOLVE_IDEA_CAP="${PACEVOLVE_IDEA_CAP:-5}"
export PACEVOLVE_MERGE_FREQ="${PACEVOLVE_MERGE_FREQ:-1}"
export PACEVOLVE_SUMMARIZE_FREQ="${PACEVOLVE_SUMMARIZE_FREQ:-20}"

#### Rollout overrides for quicker iterations ####
export PACEVOLVE_NUM_ROLLOUT="${PACEVOLVE_NUM_ROLLOUT:-25600}"
export PACEVOLVE_ROLLOUT_BATCH_SIZE="${PACEVOLVE_ROLLOUT_BATCH_SIZE:-1}"
export PACEVOLVE_N_SAMPLES_PER_PROMPT="${PACEVOLVE_N_SAMPLES_PER_PROMPT:-8}"
export PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN="${PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN:-16384}"
export PACEVOLVE_MAX_CONCURRENT_EVALS="${PACEVOLVE_MAX_CONCURRENT_EVALS:-16}"

#### Random seed ####
SEED="${SEED:-3407}"

#### Additional note for file names ####
NOTE="${NOTE:-_eplb}"

#### Dataset location ####
# Must contain expert-load.json (download from Hugging Face - see pacevolve/tasks/eplb/README.md)
EPLB_DATA_PATH="${EPLB_DATA_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pacevolve/tasks/eplb/data}"
MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

#### API key for PACEvolve gym's API model (Gemini/OpenAI/Anthropic) ####
# export GOOGLE_API_KEY="your_gemini_key"

#### Replace with your own wandb settings ####
WANDB_API_KEY="${WANDB_API_KEY:-aaa}"
WANDB_ENTITY="${WANDB_ENTITY:-bbb}"
WANDB_PROJECT="${WANDB_PROJECT:-ccc}"

########################## END CONFIGURATION SECTION ##########################

POSTFIX_STR="_seed${SEED}${NOTE}"

case "${SMALL_MODEL_NAME}" in
    dpsk_prorl_v2_1.5b)
        MODEL_FAMILY="nvidia"
        MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"
        models_file_name="deepseek-r1-distill-qwen-1.5B.sh"
        ;;
    dpsk_distill_qwen3_8b)
        MODEL_FAMILY="deepseek-ai"
        MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
        models_file_name="qwen3-8B.sh"
        ;;
    qwen3_4b_thinking_2507|qwen3-4B-Thinking-2507|Qwen3-4B-Thinking-2507)
        MODEL_FAMILY="Qwen"
        MODEL_NAME="Qwen3-4B-Thinking-2507"
        models_file_name="qwen3-4B-Thinking-2507.sh"
        ;;
    *)
        echo "Unknown SMALL_MODEL_NAME: ${SMALL_MODEL_NAME}"
        echo "Supported values: dpsk_prorl_v2_1.5b, dpsk_distill_qwen3_8b, qwen3_4b_thinking_2507"
        exit 1
        ;;
esac
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

ALGO_STR=$(echo "${ADVANTAGE_ESTIMATOR_ALGORITHM}" | tr '[:upper:]' '[:lower:]')
RUN_NAME="${SMALL_MODEL_NAME}_pacevolve_eplb_${ALGO_STR}_cfg${EPLB_CONFIG_ID}${POSTFIX_STR}"

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

if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ] && [ -f "$SAVE_SHM_DIR/$MODEL_NAME/config.json" ] && [ "${FORCE_DOWNLOAD}" -eq 0 ]; then
    echo "Model $MODEL_NAME already exists at $SAVE_SHM_DIR/$MODEL_NAME, skipping download"
else
    if [ -d "$SAVE_SHM_DIR/$MODEL_NAME" ]; then
        echo "Incomplete model directory found at $SAVE_SHM_DIR/$MODEL_NAME, deleting and re-downloading"
        rm -rf "$SAVE_SHM_DIR/$MODEL_NAME"
    fi
    mkdir -p $SAVE_SHM_DIR
    if [ -n "${MODEL_LOCAL_PATH}" ]; then
        if [ ! -d "${MODEL_LOCAL_PATH}" ]; then
            echo "MODEL_LOCAL_PATH does not exist: ${MODEL_LOCAL_PATH}"
            exit 1
        fi
        echo "Copying model ${MODEL_NAME} from local path ${MODEL_LOCAL_PATH}"
        cp -R "${MODEL_LOCAL_PATH}" "$SAVE_SHM_DIR/$MODEL_NAME"
    else
        echo "Downloading model $MODEL_NAME to current directory first..."
        hf download $MODEL_FAMILY/$MODEL_NAME --local-dir ./$MODEL_NAME
        echo "copy model from ./$MODEL_NAME to $SAVE_SHM_DIR/$MODEL_NAME"
        cp -r $MODEL_NAME $SAVE_SHM_DIR/
        echo "Model download and move completed"
    fi
fi

source scripts/models/${models_file_name}
if [ ! -d "$SAVE_SHM_DIR/${MODEL_NAME}_torch_dist" ] || [ "${FORCE_DOWNLOAD}" -eq 1 ]; then
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
echo "ALGORITHM:    ${ADVANTAGE_ESTIMATOR_ALGORITHM}"
echo "REWARD_PROCESS_TYPE: ${REWARD_PROCESS_TYPE}"
echo "ENTROPIC_GAMMA: ${ENTROPIC_KL_CONSTRAINT}"
echo "PKPO_K:       ${PKPO_K}"
echo "PKPO_ESTIMATOR_TYPE: ${PKPO_ESTIMATOR_TYPE}"
echo "HYBRID_ALPHA: ${HYBRID_ALPHA}"
echo "HYBRID_ALPHA_STEP: ${HYBRID_ALPHA_ANNEAL_STEP}"
echo "HYBRID_ALPHA_TARGET: ${HYBRID_ALPHA_ANNEAL_TARGET}"
echo "HYBRID_GRPO_VARIANT: ${HYBRID_GRPO_VARIANT}"
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
