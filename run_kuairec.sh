#!/bin/bash
# Top-level launcher for the KuaRec PACEvolve task.

########################### CONFIGURATION SECTION #############################

export SAVE_PATH="${SAVE_PATH:-/workspace/logs}"

SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-dpsk_distill_qwen3_8b}"
KUAIREC_TASK_ID="${KUAIREC_TASK_ID:-kuairec}"
KUAIREC_CONFIG_ID="${KUAIREC_CONFIG_ID:-1}"

IS_TRAINING="${IS_TRAINING:-True}"
KUAIREC_MAX_ITERS="${KUAIREC_MAX_ITERS:-2000}"
KUAIREC_EVAL_TIMEOUT="${KUAIREC_EVAL_TIMEOUT:-1200}"
KUAIREC_RECOMPILE_TIMEOUT="${KUAIREC_RECOMPILE_TIMEOUT:-60}"

export ADVANTAGE_ESTIMATOR_ALGORITHM="${ADVANTAGE_ESTIMATOR_ALGORITHM:-HYBRID_PKPO_GRPO}"
REWARD_PROCESS_TYPE="${REWARD_PROCESS_TYPE:-rl_normalized_reward}"

export PKPO_K="${PKPO_K:-4}"
export PKPO_ESTIMATOR_TYPE="${PKPO_ESTIMATOR_TYPE:-sloo_minus_one}"

# Requested anneal: PKPO/Dr.GRPO mix alpha from 0.0 -> 0.8 over 200 steps.
export HYBRID_ALPHA="${HYBRID_ALPHA:-0.0}"
export HYBRID_ALPHA_ANNEAL_STEP="${HYBRID_ALPHA_ANNEAL_STEP:-200}"
export HYBRID_ALPHA_ANNEAL_TARGET="${HYBRID_ALPHA_ANNEAL_TARGET:-0.8}"
export HYBRID_GRPO_VARIANT="${HYBRID_GRPO_VARIANT:-dr_grpo}"

export PACEVOLVE_BACKTRACK_FREQ="${PACEVOLVE_BACKTRACK_FREQ:--1}"
export PACEVOLVE_BACKTRACK_LEN="${PACEVOLVE_BACKTRACK_LEN:-5}"
export PACEVOLVE_POWER_ALPHA="${PACEVOLVE_POWER_ALPHA:-1.5}"
export PACEVOLVE_IDEA_CAP="${PACEVOLVE_IDEA_CAP:-5}"
export PACEVOLVE_MERGE_FREQ="${PACEVOLVE_MERGE_FREQ:-1}"
export PACEVOLVE_SUMMARIZE_FREQ="${PACEVOLVE_SUMMARIZE_FREQ:-20}"

export PACEVOLVE_NUM_ROLLOUT="${PACEVOLVE_NUM_ROLLOUT:-25600}"
export PACEVOLVE_ROLLOUT_BATCH_SIZE="${PACEVOLVE_ROLLOUT_BATCH_SIZE:-1}"
export PACEVOLVE_N_SAMPLES_PER_PROMPT="${PACEVOLVE_N_SAMPLES_PER_PROMPT:-8}"
export PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN="${PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN:-16384}"

# Default 16-GPU split: 4 train + 4 rollout + 8 concurrent eval.
export PACEVOLVE_NUM_GPUS_PER_NODE="${PACEVOLVE_NUM_GPUS_PER_NODE:-16}"
export PACEVOLVE_TRAIN_GPUS_PER_NODE="${PACEVOLVE_TRAIN_GPUS_PER_NODE:-4}"
export PACEVOLVE_ROLLOUT_NUM_GPUS="${PACEVOLVE_ROLLOUT_NUM_GPUS:-4}"
export PACEVOLVE_ROLLOUT_NUM_GPUS_PER_ENGINE="${PACEVOLVE_ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
export PACEVOLVE_EVAL_GPU_IDS="${PACEVOLVE_EVAL_GPU_IDS:-8,9,10,11,12,13,14,15}"
export PACEVOLVE_MAX_CONCURRENT_EVALS="${PACEVOLVE_MAX_CONCURRENT_EVALS:-8}"

export PACEVOLVE_TENSOR_MODEL_PARALLEL_SIZE="${PACEVOLVE_TENSOR_MODEL_PARALLEL_SIZE:-4}"
export PACEVOLVE_PIPELINE_MODEL_PARALLEL_SIZE="${PACEVOLVE_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export PACEVOLVE_CONTEXT_PARALLEL_SIZE="${PACEVOLVE_CONTEXT_PARALLEL_SIZE:-1}"
export PACEVOLVE_EXPERT_MODEL_PARALLEL_SIZE="${PACEVOLVE_EXPERT_MODEL_PARALLEL_SIZE:-1}"
export PACEVOLVE_EXPERT_TENSOR_PARALLEL_SIZE="${PACEVOLVE_EXPERT_TENSOR_PARALLEL_SIZE:-1}"

SEED="${SEED:-3407}"
NOTE="${NOTE:-_kuairec}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

WANDB_API_KEY="${WANDB_API_KEY:-aaa}"
WANDB_ENTITY="${WANDB_ENTITY:-bbb}"
WANDB_PROJECT="${WANDB_PROJECT:-ccc}"

########################## END CONFIGURATION SECTION ##########################

POSTFIX_STR="_seed${SEED}${NOTE}"

if [ "${SMALL_MODEL_NAME}" = "dpsk_prorl_v2_1.5b" ]; then
    MODEL_FAMILY="nvidia"
    MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"
    models_file_name="deepseek-r1-distill-qwen-1.5B.sh"
elif [ "${SMALL_MODEL_NAME}" = "dpsk_distill_qwen3_8b" ]; then
    MODEL_FAMILY="deepseek-ai"
    MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
    models_file_name="qwen3-8B.sh"
else
    echo "Unknown SMALL_MODEL_NAME: ${SMALL_MODEL_NAME}"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="${ROOT_DIR}/pacevolve/tasks/${KUAIREC_TASK_ID}"
BASE_CONFIG_YAML="${TASK_ROOT}/config/config_${KUAIREC_CONFIG_ID}.yaml"

if [ ! -f "${BASE_CONFIG_YAML}" ]; then
    echo "Missing task config: ${BASE_CONFIG_YAML}"
    exit 1
fi

TASK_NAME="KUAIREC"
ALGO_STR="$(printf '%s' "${ADVANTAGE_ESTIMATOR_ALGORITHM}" | tr '[:upper:]' '[:lower:]')"
RUN_NAME="${SMALL_MODEL_NAME}_pacevolve_${KUAIREC_TASK_ID}_${ALGO_STR}_cfg${KUAIREC_CONFIG_ID}${POSTFIX_STR}"

mkdir -p "${SAVE_PATH}" "${SAVE_PATH}/tmp" "${SAVE_PATH}/hf" "${SAVE_PATH}/wandb" "${SAVE_PATH}/shm" "${SAVE_PATH}/triton" "${SAVE_PATH}/${RUN_NAME}"

export TMPDIR=/tmp
export HF_HOME="${SAVE_PATH}/hf"
export HUGGINGFACE_HUB_CACHE="${SAVE_PATH}/hf/hub"
export TRANSFORMERS_CACHE="${SAVE_PATH}/hf/hub"
export HF_DATASETS_CACHE="${SAVE_PATH}/hf/datasets"
export SAVE_SHM_DIR="${SAVE_PATH}/shm"
export TRITON_CACHE_DIR="${SAVE_PATH}/triton"
export WANDB_CACHE_DIR="${SAVE_PATH}/wandb"
export WANDB_DIR="${SAVE_PATH}/wandb"
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT

RUNTIME_CONFIG_YAML="${SAVE_PATH}/${RUN_NAME}/config_${KUAIREC_TASK_ID}_runtime.yaml"
"${PYTHON_BIN}" - "${BASE_CONFIG_YAML}" "${RUNTIME_CONFIG_YAML}" "${TASK_ROOT}" "${KUAIREC_MAX_ITERS}" "${KUAIREC_EVAL_TIMEOUT}" "${KUAIREC_RECOMPILE_TIMEOUT}" "${PACEVOLVE_N_SAMPLES_PER_PROMPT}" "${PACEVOLVE_MAX_CONCURRENT_EVALS}" <<'PY'
import os
import sys
import yaml

(
    base_cfg,
    out_cfg,
    task_root,
    max_iters,
    eval_timeout,
    recompile_timeout,
    n_samples_per_prompt,
    max_parallel_evals,
) = sys.argv[1:]

with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

configured_data_path = os.path.expanduser(str(cfg["paths"]["data_path"]))
if not os.path.isabs(configured_data_path):
    configured_data_path = os.path.abspath(os.path.join(task_root, configured_data_path))
cfg["paths"]["data_path"] = configured_data_path
cfg["paths"]["src_path"] = os.path.join(task_root, "src")
cfg["paths"]["eval_path"] = os.path.join(task_root, "eval")
cfg["paths"]["results_path"] = os.path.join(task_root, "results")
cfg["paths"]["log_dir"] = os.path.join(task_root, "verbose")
cfg["paths"]["transcript_dir"] = os.path.join(task_root, "transcripts")

cfg["experiment"]["max_iters"] = int(max_iters)
cfg["evaluation"]["eval_timeout"] = int(eval_timeout)
cfg["evaluation"]["max_parallel_evals"] = int(max_parallel_evals)
cfg["compilation"]["recompile_timeout"] = int(recompile_timeout)

cfg.setdefault("database", {})
cfg["database"]["num_islands"] = int(n_samples_per_prompt)

os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
with open(out_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

KUAIREC_DATA_PATH="$("${PYTHON_BIN}" - "${RUNTIME_CONFIG_YAML}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)

print(cfg["paths"]["data_path"])
PY
)"

if [ ! -f "${KUAIREC_DATA_PATH}" ]; then
    echo "Missing KuaRec processed CSV from config: ${KUAIREC_DATA_PATH}"
    echo "Update paths.data_path in ${BASE_CONFIG_YAML} or place the processed CSV there."
    exit 1
fi

if [ -d "${SAVE_SHM_DIR}/${MODEL_NAME}" ] && [ -f "${SAVE_SHM_DIR}/${MODEL_NAME}/config.json" ] && [ "${FORCE_DOWNLOAD}" -eq 0 ]; then
    echo "Model ${MODEL_NAME} already exists at ${SAVE_SHM_DIR}/${MODEL_NAME}, skipping download"
else
    if [ -d "${SAVE_SHM_DIR}/${MODEL_NAME}" ]; then
        echo "Incomplete model directory found at ${SAVE_SHM_DIR}/${MODEL_NAME}, deleting and re-downloading"
        rm -rf "${SAVE_SHM_DIR:?}/${MODEL_NAME}"
    fi
    mkdir -p "${SAVE_SHM_DIR}"
    if [ -n "${MODEL_LOCAL_PATH}" ]; then
        if [ ! -d "${MODEL_LOCAL_PATH}" ]; then
            echo "MODEL_LOCAL_PATH does not exist: ${MODEL_LOCAL_PATH}"
            exit 1
        fi
        echo "Copying model ${MODEL_NAME} from local path ${MODEL_LOCAL_PATH}"
        cp -R "${MODEL_LOCAL_PATH}" "${SAVE_SHM_DIR}/${MODEL_NAME}"
    else
        echo "Downloading model ${MODEL_NAME}..."
        hf download "${MODEL_FAMILY}/${MODEL_NAME}" --local-dir "./${MODEL_NAME}"
        cp -r "${MODEL_NAME}" "${SAVE_SHM_DIR}/"
    fi
fi

source "scripts/models/${models_file_name}"
if [ ! -d "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist" ] || [ "${FORCE_DOWNLOAD}" -eq 1 ]; then
    echo "Converting HF model to torch dist format..."
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}" --save "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
fi

echo "=== KuaRec + PACEvolve Configuration ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "TASK_ID: ${KUAIREC_TASK_ID}"
echo "TASK_NAME: ${TASK_NAME}"
echo "CONFIG_YAML: ${RUNTIME_CONFIG_YAML}"
echo "DATA_PATH: ${KUAIREC_DATA_PATH}"
echo "ALGORITHM: ${ADVANTAGE_ESTIMATOR_ALGORITHM}"
echo "REWARD_PROCESS_TYPE: ${REWARD_PROCESS_TYPE}"
echo "HYBRID_ALPHA_START: ${HYBRID_ALPHA}"
echo "HYBRID_ALPHA_STEP: ${HYBRID_ALPHA_ANNEAL_STEP}"
echo "HYBRID_ALPHA_TARGET: ${HYBRID_ALPHA_ANNEAL_TARGET}"
echo "HYBRID_GRPO_VARIANT: ${HYBRID_GRPO_VARIANT}"
echo "rollout_batch_size=${PACEVOLVE_ROLLOUT_BATCH_SIZE}"
echo "n_samples_per_prompt=${PACEVOLVE_N_SAMPLES_PER_PROMPT}"
echo "max_concurrent_evals=${PACEVOLVE_MAX_CONCURRENT_EVALS}"
echo "GPU split: train=${PACEVOLVE_TRAIN_GPUS_PER_NODE}, rollout=${PACEVOLVE_ROLLOUT_NUM_GPUS}, eval=${PACEVOLVE_EVAL_GPU_IDS}"
echo "========================================"

bash "scripts_evolve/${MODEL_NAME}/general_pacevolve.sh" \
    "${WANDB_PROJECT}" \
    "${RUN_NAME}" \
    "${RUNTIME_CONFIG_YAML}" \
    "${SAVE_PATH}" \
    "${IS_TRAINING}" \
    "${REWARD_PROCESS_TYPE}" \
    "${SEED}" \
    2>&1 | tee -a "${SAVE_PATH}/${RUN_NAME}/train_log.txt"
