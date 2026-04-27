#!/bin/bash
# Shared ThetaEvolve/OpenEvolve launcher.
#
# Model wrappers set:
#   - THETAEVOLVE_MODEL_NAME
#   - THETAEVOLVE_MODEL_ARGS_SCRIPT
#
# This path intentionally uses SLIME's generic --evolving-gym backed by
# openevolve_adapted. It does not enable --pacevolve-gym or any PACEvolve
# workflow/context-management controls.

set -euo pipefail

if [ $# -ne 10 ]; then
  echo "Usage: $0 WANDB_PROJECT RUN_NAME INITIAL_PROGRAM EVALUATOR_FILE CONFIG_YAML SAVE_PATH IS_TRAINING LAZY_OUTPUT_PENALTY REWARD_PROCESS_TYPE SEED"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${THETAEVOLVE_MODEL_NAME:?THETAEVOLVE_MODEL_NAME must be set by the model wrapper}"
MODEL_ARGS_SCRIPT="${THETAEVOLVE_MODEL_ARGS_SCRIPT:?THETAEVOLVE_MODEL_ARGS_SCRIPT must be set by the model wrapper}"

WANDB_PROJECT=$1
RUN_NAME=$2
INITIAL_PROGRAM=$3
EVALUATOR_FILE=$4
CONFIG_YAML=$5
SAVE_PATH=$6
IS_TRAINING=$7
LAZY_OUTPUT_PENALTY=$8
REWARD_PROCESS_TYPE=$9
SEED=${10}

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

if [ "${LAZY_OUTPUT_PENALTY}" -ge 2 ]; then
  echo "Lazy-output penalty level ${LAZY_OUTPUT_PENALTY} requires database duplicate checks."
fi

SAVE_SHM_DIR="${SAVE_PATH}/shm"
CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
RECORD_PATH="${SAVE_PATH}/${RUN_NAME}/records"

if [ "${IS_TRAINING}" = "False" ] || [ "${IS_TRAINING}" = "false" ]; then
  DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
  echo "Inference-only mode enabled (IS_TRAINING=${IS_TRAINING})"
else
  DEBUG_ROLLOUT_ONLY=""
  echo "Normal training mode (IS_TRAINING=${IS_TRAINING})"
fi

mkdir -p "${CKPT_DIR}"

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true

set -x

export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
echo "NVLINK_COUNT: ${NVLINK_COUNT}"
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source "${ROOT_DIR}/${MODEL_ARGS_SCRIPT}"
declare -p DISTRIBUTED_ARGS >/dev/null 2>&1 || DISTRIBUTED_ARGS=()

CKPT_ARGS=(
  --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}"
  --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
  --load "${CKPT_DIR}/"
  --save "${CKPT_DIR}/"
  --save-interval "${THETAEVOLVE_SAVE_INTERVAL:-100}"
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --evolving-gym
  --evolving-gym-initial-program "${INITIAL_PROGRAM}"
  --evolving-gym-evaluator-file "${EVALUATOR_FILE}"
  --evolving-gym-config-path "${CONFIG_YAML}"
  --evolving-gym-max-concurrent-evals "${THETAEVOLVE_MAX_CONCURRENT_EVALS:-8}"
  --evolving-gym-log-prompts
  --evolving-gym-record
  --evolving-gym-record-dir "${RECORD_PATH}"
  --evolving-gym-lazy-output-penalty-level "${LAZY_OUTPUT_PENALTY}"
  --evolving-gym-seed "${SEED}"
  --evolving-gym-database-reinit-ratio "${THETAEVOLVE_DATABASE_REINIT_RATIO:-0.0}"
  --evolving-gym-smallest-restart-step "${THETAEVOLVE_SMALLEST_RESTART_STEP:-10000000}"
  --evolving-gym-largest-restart-step "${THETAEVOLVE_LARGEST_RESTART_STEP:-10000000}"
  --evolving-gym-add-historical-programs "${THETAEVOLVE_ADD_HISTORICAL_PROGRAMS:-0}"
  --evolving-gym-reward-process-type "${REWARD_PROCESS_TYPE}"

  --apply-chat-template

  --rm-type evolving-gym
  --reward-key reward

  --num-rollout "${THETAEVOLVE_NUM_ROLLOUT:-10000000}"
  --rollout-batch-size "${THETAEVOLVE_ROLLOUT_BATCH_SIZE:-1}"
  --n-samples-per-prompt "${THETAEVOLVE_N_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${THETAEVOLVE_ROLLOUT_MAX_RESPONSE_LEN:-16384}"
  --rollout-temperature "${THETAEVOLVE_ROLLOUT_TEMPERATURE:-1.0}"

  --over-sampling-batch-size "${THETAEVOLVE_OVER_SAMPLING_BATCH_SIZE:-${THETAEVOLVE_ROLLOUT_BATCH_SIZE:-1}}"
  --partial-rollout

  --num-steps-per-rollout "${THETAEVOLVE_NUM_STEPS_PER_ROLLOUT:-1}"
  --wandb-always-use-train-step
  --balance-data
)

if [ -n "${THETAEVOLVE_ROLLOUT_MAX_CONTEXT_LEN:-}" ]; then
  ROLLOUT_ARGS+=(--rollout-max-context-len "${THETAEVOLVE_ROLLOUT_MAX_CONTEXT_LEN}")
fi

if [ -n "${THETAEVOLVE_ROLLOUT_MAX_PROMPT_LEN:-}" ]; then
  ROLLOUT_ARGS+=(--rollout-max-prompt-len "${THETAEVOLVE_ROLLOUT_MAX_PROMPT_LEN}")
fi

NUM_GPUS_PER_NODE=${THETAEVOLVE_NUM_GPUS_PER_NODE:-16}
TRAIN_GPUS_PER_NODE=${THETAEVOLVE_TRAIN_GPUS_PER_NODE:-4}
ROLLOUT_NUM_GPUS=${THETAEVOLVE_ROLLOUT_NUM_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${THETAEVOLVE_ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
EVAL_GPU_IDS=${THETAEVOLVE_EVAL_GPU_IDS:-8,9,10,11,12,13,14,15}
IFS=',' read -r -a EVAL_GPU_ID_ARRAY <<< "${EVAL_GPU_IDS}"
EVAL_GPU_COUNT=0
for gpu_id in "${EVAL_GPU_ID_ARRAY[@]}"; do
  if [ -n "${gpu_id}" ]; then
    EVAL_GPU_COUNT=$((EVAL_GPU_COUNT + 1))
  fi
done

if [ $((TRAIN_GPUS_PER_NODE + ROLLOUT_NUM_GPUS + EVAL_GPU_COUNT)) -gt "${NUM_GPUS_PER_NODE}" ]; then
  echo "Invalid GPU split: train=${TRAIN_GPUS_PER_NODE}, rollout=${ROLLOUT_NUM_GPUS}, eval=${EVAL_GPU_COUNT}, total=${NUM_GPUS_PER_NODE}"
  exit 1
fi

if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
  echo "ROLLOUT_NUM_GPUS (${ROLLOUT_NUM_GPUS}) must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE (${ROLLOUT_NUM_GPUS_PER_ENGINE})"
  exit 1
fi

TRAIN_SEQ_LENGTH=${THETAEVOLVE_TRAIN_SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${THETAEVOLVE_MAX_POSITION_EMBEDDINGS:-${TRAIN_SEQ_LENGTH}}
MAX_TOKENS_PER_GPU=${THETAEVOLVE_MAX_TOKENS_PER_GPU:-2048}

PERF_ARGS=(
  --seq-length "${TRAIN_SEQ_LENGTH}"
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --tensor-model-parallel-size "${THETAEVOLVE_TENSOR_MODEL_PARALLEL_SIZE:-2}"
  --sequence-parallel
  --pipeline-model-parallel-size "${THETAEVOLVE_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
  --context-parallel-size "${THETAEVOLVE_CONTEXT_PARALLEL_SIZE:-1}"
  --expert-model-parallel-size "${THETAEVOLVE_EXPERT_MODEL_PARALLEL_SIZE:-1}"
  --expert-tensor-parallel-size "${THETAEVOLVE_EXPERT_TENSOR_PARALLEL_SIZE:-1}"

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

if [ -n "${THETAEVOLVE_LOG_PROBS_MAX_TOKENS_PER_GPU:-}" ]; then
  PERF_ARGS+=(--log-probs-max-tokens-per-gpu "${THETAEVOLVE_LOG_PROBS_MAX_TOKENS_PER_GPU}")
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --kl-coef -0.0
  --use-tis
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${THETAEVOLVE_LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay "${THETAEVOLVE_WEIGHT_DECAY:-0.1}"
  --adam-beta1 "${THETAEVOLVE_ADAM_BETA1:-0.9}"
  --adam-beta2 "${THETAEVOLVE_ADAM_BETA2:-0.98}"
)

if [ "${THETAEVOLVE_OPTIMIZER_CPU_OFFLOAD:-1}" = "1" ]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

WANDB_ARGS=()
if [ "${THETAEVOLVE_USE_WANDB:-1}" = "1" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-team "${WANDB_ENTITY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${RUN_NAME}"
    --wandb-key "${WANDB_API_KEY}"
  )
fi

SGLANG_ARGS=(
  --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${THETAEVOLVE_SGLANG_MEM_FRACTION_STATIC:-0.6}"
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

if [ -n "${THETAEVOLVE_SGLANG_SERVER_CONCURRENCY:-}" ]; then
  SGLANG_ARGS+=(--sglang-server-concurrency "${THETAEVOLVE_SGLANG_SERVER_CONCURRENCY}")
fi

MISC_ARGS=(
  ${DEBUG_ROLLOUT_ONLY}
  --seed "${SEED}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS_PER_NODE}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

export TRITON_DISABLE=1

export FAST_MOUNT="${SAVE_PATH}/fast_mount"
export HF_DATASETS_CACHE="${FAST_MOUNT}/hf/datasets"
export DATASETS_CACHE="${HF_DATASETS_CACHE}"
export DATASETS_TMPDIR="${FAST_MOUNT}/tmp"
export PYARROW_TMP_DIR="${FAST_MOUNT}/tmp"

mkdir -p "${HF_DATASETS_CACHE}" "${DATASETS_TMPDIR}"
echo "[disk] HF_DATASETS_CACHE=${HF_DATASETS_CACHE} TMPDIR=${TMPDIR}"

RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "${ROOT_DIR}:${ROOT_DIR}/openevolve_adapted:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "HF_HOME": "${HF_HOME}",
    "HUGGINGFACE_HUB_CACHE": "${HUGGINGFACE_HUB_CACHE}",
    "TRANSFORMERS_CACHE": "${TRANSFORMERS_CACHE}",
    "HF_DATASETS_CACHE": "${HF_DATASETS_CACHE}",
    "DATASETS_CACHE": "${DATASETS_CACHE}",
    "DATASETS_TMPDIR": "${DATASETS_TMPDIR}",
    "PYARROW_TMP_DIR": "${PYARROW_TMP_DIR}",
    "TMPDIR": "${TMPDIR}",
    "WANDB_CACHE_DIR": "${WANDB_CACHE_DIR}",
    "WANDB_DIR": "${WANDB_DIR}",
    "WANDB_GROUP": "${RUN_NAME}",
    "TRITON_DISABLE": "1",
    "THETAEVOLVE_EVAL_GPU_IDS": "${EVAL_GPU_IDS}"
  }
}
JSON
)"

echo "RUNTIME_ENV_JSON = ${RUNTIME_ENV_JSON}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${TRAIN_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${DISTRIBUTED_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"
