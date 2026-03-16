#!/bin/bash
# Top-level script for running GPU-kernel PACEvolve tasks on an 8x H100 node.
#
# Default GPU split:
#   - training: GPUs 0,1
#   - rollout:  GPUs 2,3,4,5
#   - eval:     GPUs 6,7
#
# The evaluator side is serialized through a dedicated GPU lease pool, so even
# if rollout_batch_size=8, the 8 kernel evals will wait and run 2-at-a-time.

########################### CONFIGURATION SECTION #############################

export SAVE_PATH="${SAVE_PATH:-/workspace/logs}"

SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-dpsk_distill_qwen3_8b}"
KERNEL_TASK_ID="${KERNEL_TASK_ID:-kernel_bench}"   # kernel_bench | trimul
KERNEL_CONFIG_ID="${KERNEL_CONFIG_ID:-1}"
IS_TRAINING="${IS_TRAINING:-True}"
REWARD_PROCESS_TYPE="${REWARD_PROCESS_TYPE:-rl_normalized_reward}"

SEED="${SEED:-3407}"
NOTE="${NOTE:-_kernel}"

export PKPO_K="${PKPO_K:-4}"
export PKPO_ESTIMATOR_TYPE="${PKPO_ESTIMATOR_TYPE:-sloo_minus_one}"

export PACEVOLVE_BACKTRACK_FREQ="${PACEVOLVE_BACKTRACK_FREQ:--1}"
export PACEVOLVE_BACKTRACK_LEN="${PACEVOLVE_BACKTRACK_LEN:-5}"
export PACEVOLVE_POWER_ALPHA="${PACEVOLVE_POWER_ALPHA:-1.5}"
export PACEVOLVE_IDEA_CAP="${PACEVOLVE_IDEA_CAP:-5}"
export PACEVOLVE_MERGE_FREQ="${PACEVOLVE_MERGE_FREQ:-1}"
export PACEVOLVE_SUMMARIZE_FREQ="${PACEVOLVE_SUMMARIZE_FREQ:-20}"

# Batch 8 prompts per rollout by default.
export PACEVOLVE_NUM_ROLLOUT="${PACEVOLVE_NUM_ROLLOUT:-25600}"
export PACEVOLVE_ROLLOUT_BATCH_SIZE="${PACEVOLVE_ROLLOUT_BATCH_SIZE:-8}"
export PACEVOLVE_N_SAMPLES_PER_PROMPT="${PACEVOLVE_N_SAMPLES_PER_PROMPT:-1}"
export PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN="${PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN:-16384}"

# GPU partition.
export PACEVOLVE_NUM_GPUS_PER_NODE="${PACEVOLVE_NUM_GPUS_PER_NODE:-8}"
export PACEVOLVE_TRAIN_GPUS_PER_NODE="${PACEVOLVE_TRAIN_GPUS_PER_NODE:-2}"
export PACEVOLVE_ROLLOUT_NUM_GPUS="${PACEVOLVE_ROLLOUT_NUM_GPUS:-4}"
export PACEVOLVE_ROLLOUT_NUM_GPUS_PER_ENGINE="${PACEVOLVE_ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
export PACEVOLVE_EVAL_GPU_IDS="${PACEVOLVE_EVAL_GPU_IDS:-6,7}"
export PACEVOLVE_MAX_CONCURRENT_EVALS="${PACEVOLVE_MAX_CONCURRENT_EVALS:-2}"

# Match Megatron parallelism to the 2 training GPUs by default.
export PACEVOLVE_TENSOR_MODEL_PARALLEL_SIZE="${PACEVOLVE_TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PACEVOLVE_PIPELINE_MODEL_PARALLEL_SIZE="${PACEVOLVE_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export PACEVOLVE_CONTEXT_PARALLEL_SIZE="${PACEVOLVE_CONTEXT_PARALLEL_SIZE:-1}"
export PACEVOLVE_EXPERT_MODEL_PARALLEL_SIZE="${PACEVOLVE_EXPERT_MODEL_PARALLEL_SIZE:-1}"
export PACEVOLVE_EXPERT_TENSOR_PARALLEL_SIZE="${PACEVOLVE_EXPERT_TENSOR_PARALLEL_SIZE:-1}"

WANDB_API_KEY="${WANDB_API_KEY:-aaa}"
WANDB_ENTITY="${WANDB_ENTITY:-bbb}"
WANDB_PROJECT="${WANDB_PROJECT:-ccc}"

########################## END CONFIGURATION SECTION ##########################

POSTFIX_STR="_seed${SEED}${NOTE}"

if [ "${SMALL_MODEL_NAME}" = "dpsk_distill_qwen3_8b" ]; then
    MODEL_FAMILY="deepseek-ai"
    MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
    models_file_name="qwen3-8B.sh"
else
    echo "Unknown SMALL_MODEL_NAME: ${SMALL_MODEL_NAME}"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${KERNEL_TASK_ID}" in
    kernel_bench)
        TASK_ROOT="${ROOT_DIR}/pacevolve/tasks/kernel_bench"
        BASE_CONFIG_YAML="${TASK_ROOT}/config/config_${KERNEL_CONFIG_ID}.yaml"
        ;;
    trimul)
        TASK_ROOT="${ROOT_DIR}/pacevolve/tasks/trimul"
        BASE_CONFIG_YAML="${TASK_ROOT}/config/config_${KERNEL_CONFIG_ID}.yaml"
        ;;
    *)
        echo "Unsupported KERNEL_TASK_ID: ${KERNEL_TASK_ID}"
        echo "Expected one of: kernel_bench, trimul"
        exit 1
        ;;
esac

if [ ! -f "${BASE_CONFIG_YAML}" ]; then
    echo "Missing task config: ${BASE_CONFIG_YAML}"
    exit 1
fi

TASK_NAME=$(python3 -c "
import yaml
with open('${BASE_CONFIG_YAML}') as f:
    cfg = yaml.safe_load(f)
print(cfg['experiment']['sota_algo_name'])
" 2>/dev/null || echo "${KERNEL_TASK_ID}")

RUN_NAME="${SMALL_MODEL_NAME}_pacevolve_${KERNEL_TASK_ID}_${TASK_NAME}${POSTFIX_STR}"

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

RUNTIME_CONFIG_YAML="${SAVE_PATH}/${RUN_NAME}/config_${KERNEL_TASK_ID}_runtime.yaml"
python3 - "${BASE_CONFIG_YAML}" "${RUNTIME_CONFIG_YAML}" "${TASK_ROOT}" "${PACEVOLVE_N_SAMPLES_PER_PROMPT}" <<'PY'
import os
import sys
import yaml

base_cfg, out_cfg, task_root, n_samples_per_prompt = sys.argv[1:]
with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

cfg["paths"]["src_path"] = os.path.join(task_root, "src")
cfg["paths"]["eval_path"] = os.path.join(task_root, "eval")
cfg["paths"]["results_path"] = os.path.join(task_root, "results")
cfg["paths"]["log_dir"] = os.path.join(task_root, "verbose")
cfg["paths"]["transcript_dir"] = os.path.join(task_root, "transcripts")

cfg.setdefault("database", {})
cfg["database"]["num_islands"] = int(n_samples_per_prompt)

cfg.setdefault("evaluation", {})
cfg["evaluation"]["max_parallel_evals"] = 1

os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
with open(out_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

FORCE_DOWNLOAD=0
if [ -d "${SAVE_SHM_DIR}/${MODEL_NAME}" ] && [ -f "${SAVE_SHM_DIR}/${MODEL_NAME}/config.json" ] && [ "${FORCE_DOWNLOAD}" -eq 0 ]; then
    echo "Model ${MODEL_NAME} already exists at ${SAVE_SHM_DIR}/${MODEL_NAME}, skipping download"
else
    if [ -d "${SAVE_SHM_DIR}/${MODEL_NAME}" ]; then
        echo "Incomplete model directory found at ${SAVE_SHM_DIR}/${MODEL_NAME}, deleting and re-downloading"
        rm -rf "${SAVE_SHM_DIR:?}/${MODEL_NAME}"
    fi
    echo "Downloading model ${MODEL_NAME}..."
    hf download "${MODEL_FAMILY}/${MODEL_NAME}" --local-dir "./${MODEL_NAME}"
    cp -r "${MODEL_NAME}" "${SAVE_SHM_DIR}/"
fi

source "scripts/models/${models_file_name}"
if [ ! -d "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist" ] || [ "${FORCE_DOWNLOAD}" -eq 1 ]; then
    echo "Converting HF model to torch dist format..."
    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}" --save "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
fi

echo "=== Kernel + PACEvolve Configuration ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "TASK_ID: ${KERNEL_TASK_ID}"
echo "TASK_NAME: ${TASK_NAME}"
echo "CONFIG_YAML: ${RUNTIME_CONFIG_YAML}"
echo "GPU split: train=${PACEVOLVE_TRAIN_GPUS_PER_NODE}, rollout=${PACEVOLVE_ROLLOUT_NUM_GPUS}, eval=${PACEVOLVE_EVAL_GPU_IDS}"
echo "rollout_batch_size=${PACEVOLVE_ROLLOUT_BATCH_SIZE}"
echo "n_samples_per_prompt=${PACEVOLVE_N_SAMPLES_PER_PROMPT}"
echo "max_concurrent_evals=${PACEVOLVE_MAX_CONCURRENT_EVALS}"
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
