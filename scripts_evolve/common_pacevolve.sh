#!/bin/bash
# Shared PACEvolve launcher. Model-specific wrappers should set:
#   - PACEVOLVE_MODEL_NAME
#   - PACEVOLVE_MODEL_ARGS_SCRIPT
# and may optionally set model-specific defaults via env vars.

if [ $# -lt 7 ]; then
  echo "Usage: $0 WANDB_PROJECT RUN_NAME CONFIG_YAML SAVE_PATH IS_TRAINING REWARD_PROCESS_TYPE SEED [INITIAL_PROGRAM]"
  echo ""
  echo "Required parameters:"
  echo "  WANDB_PROJECT        - Weights & Biases project name"
  echo "  RUN_NAME             - Experiment run name"
  echo "  CONFIG_YAML          - Path to PACEvolve config YAML file"
  echo "  SAVE_PATH            - Save directory path"
  echo "  IS_TRAINING          - True for training, False for inference-only"
  echo "  REWARD_PROCESS_TYPE  - Reward processing type (original_reward, rl_normalized_reward, improve_reward)"
  echo "  SEED                 - Random seed for reproducibility"
  echo ""
  echo "Optional parameters:"
  echo "  INITIAL_PROGRAM      - Path to initial program file (defaults to config's sota_algo)"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_NAME="${PACEVOLVE_MODEL_NAME:?PACEVOLVE_MODEL_NAME must be set by the model wrapper}"
MODEL_ARGS_SCRIPT="${PACEVOLVE_MODEL_ARGS_SCRIPT:?PACEVOLVE_MODEL_ARGS_SCRIPT must be set by the model wrapper}"

WANDB_PROJECT=$1
RUN_NAME=$2
CONFIG_YAML=$3
SAVE_PATH=$4
IS_TRAINING=$5
REWARD_PROCESS_TYPE=$6
SEED=$7
INITIAL_PROGRAM=${8:-""}

SAVE_SHM_DIR="${SAVE_PATH}/shm"
CKPT_DIR="${SAVE_PATH}/${RUN_NAME}"
RECORD_PATH="${SAVE_PATH}/${RUN_NAME}/records"
TRAIN_SEQ_LENGTH="${PACEVOLVE_TRAIN_SEQ_LENGTH:-4096}"
MAX_POSITION_EMBEDDINGS="${PACEVOLVE_MAX_POSITION_EMBEDDINGS:-${TRAIN_SEQ_LENGTH}}"
MAX_TOKENS_PER_GPU="${PACEVOLVE_MAX_TOKENS_PER_GPU:-2048}"
LOG_PROBS_MAX_TOKENS_PER_GPU="${PACEVOLVE_LOG_PROBS_MAX_TOKENS_PER_GPU:-}"
ROLLOUT_MAX_CONTEXT_LEN="${PACEVOLVE_ROLLOUT_MAX_CONTEXT_LEN:-}"
ROLLOUT_MAX_PROMPT_LEN="${PACEVOLVE_ROLLOUT_MAX_PROMPT_LEN:-}"
SGLANG_MEM_FRACTION_STATIC="${PACEVOLVE_SGLANG_MEM_FRACTION_STATIC:-0.4}"
SGLANG_SERVER_CONCURRENCY="${PACEVOLVE_SGLANG_SERVER_CONCURRENCY:-}"

if [ "$IS_TRAINING" = "False" ] || [ "$IS_TRAINING" = "false" ]; then
    DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
    echo "Inference-only mode enabled (IS_TRAINING=$IS_TRAINING)"
else
    DEBUG_ROLLOUT_ONLY=""
    echo "Normal training mode (IS_TRAINING=$IS_TRAINING)"
fi

mkdir -p "${CKPT_DIR}"

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

set -ex

export PYTHONBUFFERED=16
export TOKENIZERS_PARALLELISM=false

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
echo "NVLINK_COUNT: $NVLINK_COUNT"
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${ROOT_DIR}/${MODEL_ARGS_SCRIPT}"

CKPT_ARGS=(
   --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}"
   --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
   --load "${CKPT_DIR}/"
   --save "${CKPT_DIR}/"
   --save-interval ${PACEVOLVE_SAVE_INTERVAL:-100}
)

PACEVOLVE_ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --pacevolve-gym
  --pacevolve-gym-config-path "${CONFIG_YAML}"
  --pacevolve-gym-max-concurrent-evals ${PACEVOLVE_MAX_CONCURRENT_EVALS:-1}
  --pacevolve-gym-log-prompts
  --pacevolve-gym-record
  --pacevolve-gym-record-dir "${RECORD_PATH}"
  --pacevolve-gym-seed ${SEED}
  --pacevolve-gym-reward-process-type "${REWARD_PROCESS_TYPE}"
  --pacevolve-gym-backtrack-freq ${PACEVOLVE_BACKTRACK_FREQ:--1}
  --pacevolve-gym-backtrack-len ${PACEVOLVE_BACKTRACK_LEN:-5}
  --pacevolve-gym-power-alpha ${PACEVOLVE_POWER_ALPHA:-1.5}
  --pacevolve-gym-idea-cap ${PACEVOLVE_IDEA_CAP:-5}
  --pacevolve-gym-merge-freq ${PACEVOLVE_MERGE_FREQ:-1}
  --pacevolve-gym-summarize-freq ${PACEVOLVE_SUMMARIZE_FREQ:-20}
)

if [ -n "${INITIAL_PROGRAM}" ]; then
  PACEVOLVE_ROLLOUT_ARGS+=(--pacevolve-gym-initial-program "${INITIAL_PROGRAM}")
fi

ROLLOUT_ARGS=(
  ${PACEVOLVE_ROLLOUT_ARGS[@]}

  --apply-chat-template

  --rm-type pacevolve-gym
  --reward-key reward

  --num-rollout ${PACEVOLVE_NUM_ROLLOUT:-10000000}
  --rollout-batch-size ${PACEVOLVE_ROLLOUT_BATCH_SIZE:-1}
  --n-samples-per-prompt ${PACEVOLVE_N_SAMPLES_PER_PROMPT:-8}
  --rollout-max-response-len ${PACEVOLVE_ROLLOUT_MAX_RESPONSE_LEN:-16384}
  --rollout-temperature ${PACEVOLVE_ROLLOUT_TEMPERATURE:-1.0}

  --over-sampling-batch-size ${PACEVOLVE_OVER_SAMPLING_BATCH_SIZE:-${PACEVOLVE_ROLLOUT_BATCH_SIZE:-1}}
  --partial-rollout

  --num-steps-per-rollout ${PACEVOLVE_NUM_STEPS_PER_ROLLOUT:-1}
  --wandb-always-use-train-step
  --balance-data
)

if [ -n "${ROLLOUT_MAX_CONTEXT_LEN}" ]; then
  ROLLOUT_ARGS+=(--rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}")
fi

if [ -n "${ROLLOUT_MAX_PROMPT_LEN}" ]; then
  ROLLOUT_ARGS+=(--rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}")
fi

NUM_GPUS_PER_NODE=${PACEVOLVE_NUM_GPUS_PER_NODE:-16}
TRAIN_GPUS_PER_NODE=${PACEVOLVE_TRAIN_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${PACEVOLVE_ROLLOUT_NUM_GPUS:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${PACEVOLVE_ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
TRAIN_TP_SIZE=${PACEVOLVE_TENSOR_MODEL_PARALLEL_SIZE:-4}
TRAIN_PP_SIZE=${PACEVOLVE_PIPELINE_MODEL_PARALLEL_SIZE:-1}
TRAIN_CP_SIZE=${PACEVOLVE_CONTEXT_PARALLEL_SIZE:-2}
TRAIN_EP_SIZE=${PACEVOLVE_EXPERT_MODEL_PARALLEL_SIZE:-1}
TRAIN_ETP_SIZE=${PACEVOLVE_EXPERT_TENSOR_PARALLEL_SIZE:-1}
EVAL_GPU_IDS=${PACEVOLVE_EVAL_GPU_IDS:-}
IFS=',' read -r -a EVAL_GPU_ID_ARRAY <<< "${EVAL_GPU_IDS}"
EVAL_GPU_COUNT=0
for gpu_id in "${EVAL_GPU_ID_ARRAY[@]}"; do
  if [ -n "${gpu_id}" ]; then
    EVAL_GPU_COUNT=$((EVAL_GPU_COUNT + 1))
  fi
done

if [ $((TRAIN_GPUS_PER_NODE + ROLLOUT_NUM_GPUS + EVAL_GPU_COUNT)) -gt ${NUM_GPUS_PER_NODE} ]; then
  echo "Invalid GPU split: train=${TRAIN_GPUS_PER_NODE}, rollout=${ROLLOUT_NUM_GPUS}, eval=${EVAL_GPU_COUNT}, total=${NUM_GPUS_PER_NODE}"
  exit 1
fi

if [ $((ROLLOUT_NUM_GPUS % ROLLOUT_NUM_GPUS_PER_ENGINE)) -ne 0 ]; then
  echo "ROLLOUT_NUM_GPUS (${ROLLOUT_NUM_GPUS}) must be divisible by ROLLOUT_NUM_GPUS_PER_ENGINE (${ROLLOUT_NUM_GPUS_PER_ENGINE})"
  exit 1
fi

PERF_ARGS=(
  --seq-length ${TRAIN_SEQ_LENGTH}
  --max-position-embeddings ${MAX_POSITION_EMBEDDINGS}
  --tensor-model-parallel-size ${TRAIN_TP_SIZE}
  --sequence-parallel
  --pipeline-model-parallel-size ${TRAIN_PP_SIZE}
  --context-parallel-size ${TRAIN_CP_SIZE}
  --expert-model-parallel-size ${TRAIN_EP_SIZE}
  --expert-tensor-parallel-size ${TRAIN_ETP_SIZE}

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

if [ -n "${LOG_PROBS_MAX_TOKENS_PER_GPU}" ]; then
  PERF_ARGS+=(--log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}")
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --kl-coef -0.0
  --use-tis
)

DR_GRPO_ARGS=(
  --advantage-estimator dr_grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --kl-coef -0.0
  --use-tis
)

HYBRID_PKPO_GRPO_ARGS=(
  --advantage-estimator hybrid_pkpo_grpo
  --pkpo-k ${PKPO_K:-4}
  --pkpo-estimator-type ${PKPO_ESTIMATOR_TYPE:-sloo_minus_one}
  --hybrid-alpha ${HYBRID_ALPHA:-0.5}
  --hybrid-alpha-anneal-step ${HYBRID_ALPHA_ANNEAL_STEP:-200}
  --hybrid-alpha-anneal-target ${HYBRID_ALPHA_ANNEAL_TARGET:-0.8}
  --hybrid-grpo-variant ${HYBRID_GRPO_VARIANT:-grpo}
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --kl-coef -0.0
  --use-tis
)

PKPO_ARGS=(
  --advantage-estimator pkpo
  --pkpo-k ${PKPO_K:-4}
  --pkpo-estimator-type ${PKPO_ESTIMATOR_TYPE:-sloo_minus_one}
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
)

ENTROPIC_ARGS=(
  --advantage-estimator entropic
  --entropic-kl-constraint ${ENTROPIC_KL_CONSTRAINT:-0.6931471805599453}
  --kl-coef 0.1
  --entropy-coef 0.0
  --clip-grad 0
  --lr 4e-5
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-8
  --use-tis
)

if [ -n "${PKPO_K_ANNEAL_STEP}" ]; then
  PKPO_ARGS+=(--pkpo-k-anneal-step ${PKPO_K_ANNEAL_STEP} --pkpo-k-anneal-target ${PKPO_K_ANNEAL_TARGET:-1})
fi

ALG="${ADVANTAGE_ESTIMATOR_ALGORITHM:-PKPO}"
ALG_NORMALIZED=$(printf '%s' "${ALG}" | tr '[:upper:]' '[:lower:]')
case "${ALG_NORMALIZED}" in
  grpo)
    ALGORITHM_ARGS=("${GRPO_ARGS[@]}")
    ;;
  dr_grpo|dr.grpo)
    ALGORITHM_ARGS=("${DR_GRPO_ARGS[@]}")
    ;;
  hybrid_pkpo_grpo)
    ALGORITHM_ARGS=("${HYBRID_PKPO_GRPO_ARGS[@]}")
    ;;
  pkpo)
    ALGORITHM_ARGS=("${PKPO_ARGS[@]}")
    ;;
  entropic)
    ALGORITHM_ARGS=("${ENTROPIC_ARGS[@]}")
    ;;
  *)
    echo "Unsupported ADVANTAGE_ESTIMATOR_ALGORITHM: ${ALG}"
    echo "Expected one of: GRPO, DR_GRPO, HYBRID_PKPO_GRPO, PKPO, ENTROPIC"
    exit 1
    ;;
esac

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98

  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --num-gpus-per-node ${NUM_GPUS_PER_NODE}
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC}
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

if [ -n "${SGLANG_SERVER_CONCURRENCY}" ]; then
  SGLANG_ARGS+=(--sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}")
fi

MISC_ARGS=(
  ${DEBUG_ROLLOUT_ONLY}
  --seed ${SEED}
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

export TRITON_DISABLE=1

export FAST_MOUNT=$SAVE_PATH/fast_mount
export HF_DATASETS_CACHE=$FAST_MOUNT/hf/datasets
export DATASETS_CACHE=$HF_DATASETS_CACHE
export DATASETS_TMPDIR=$FAST_MOUNT/tmp
export PYARROW_TMP_DIR=$FAST_MOUNT/tmp

mkdir -p "$HF_DATASETS_CACHE" "$DATASETS_TMPDIR"
echo "[disk] HF_DATASETS_CACHE=$HF_DATASETS_CACHE TMPDIR=$TMPDIR"

RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
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
    "GOOGLE_API_KEY": "${GOOGLE_API_KEY:-}",
    "OPENAI_API_KEY": "${OPENAI_API_KEY:-}",
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
    "PACEVOLVE_EVAL_GPU_IDS": "${EVAL_GPU_IDS}"
  }
}
JSON
)"

echo "RUNTIME_ENV_JSON = $RUNTIME_ENV_JSON"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${TRAIN_GPUS_PER_NODE} \
  --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${ALGORITHM_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
