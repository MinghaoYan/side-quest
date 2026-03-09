#!/bin/bash
# General training script for PACEvolve gym with configurable parameters
# Usage: ./scripts_evolve/${MODEL_NAME}/general_pacevolve.sh WANDB_PROJECT RUN_NAME CONFIG_YAML SAVE_PATH IS_TRAINING REWARD_PROCESS_TYPE SEED [INITIAL_PROGRAM]

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
MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"

# Determine debug-rollout-only mode based on IS_TRAINING
if [ "$IS_TRAINING" = "False" ] || [ "$IS_TRAINING" = "false" ]; then
    DEBUG_ROLLOUT_ONLY="--debug-rollout-only"
    echo "Inference-only mode enabled (IS_TRAINING=$IS_TRAINING)"
else
    DEBUG_ROLLOUT_ONLY=""
    echo "Normal training mode (IS_TRAINING=$IS_TRAINING)"
fi

# Create checkpoint directory
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

source scripts/models/qwen3-8B.sh

CKPT_ARGS=(
   --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}"
   --ref-load "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
   --load "${CKPT_DIR}/"
   --save "${CKPT_DIR}/"
   --save-interval 100
)

# Build PACEvolve gym rollout args
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

# Add initial program if provided
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


PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28

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
  --eps-clip 0.2
  --eps-clip-high 0.28

  --use-tis
)

if [ -n "${PKPO_K_ANNEAL_STEP}" ]; then
  PKPO_ARGS+=(--pkpo-k-anneal-step ${PKPO_K_ANNEAL_STEP} --pkpo-k-anneal-target ${PKPO_K_ANNEAL_TARGET:-1})
fi

# Select algorithm args based on ADVANTAGE_ESTIMATOR_ALGORITHM.
ALG="${ADVANTAGE_ESTIMATOR_ALGORITHM:-PKPO}"
ALG_NORMALIZED=$(printf '%s' "${ALG}" | tr '[:upper:]' '[:lower:]')
case "${ALG_NORMALIZED}" in
  grpo)
    ALGORITHM_ARGS=("${GRPO_ARGS[@]}")
    ;;
  pkpo)
    ALGORITHM_ARGS=("${PKPO_ARGS[@]}")
    ;;
  entropic)
    ALGORITHM_ARGS=("${ENTROPIC_ARGS[@]}")
    ;;
  *)
    echo "Unsupported ADVANTAGE_ESTIMATOR_ALGORITHM: ${ALG}"
    echo "Expected one of: GRPO, PKPO, ENTROPIC"
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
  --num-gpus-per-node 16
  --rollout-num-gpus-per-engine 8
  --sglang-mem-fraction-static 0.4
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
  ${DEBUG_ROLLOUT_ONLY}
  --seed ${SEED}
  --kl-coef ${KL_COEF:-0.0}
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# Start Ray with all 16 GPUs (8 training + 8 inference, no colocate)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 16 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Disable Triton
export TRITON_DISABLE=1

export FAST_MOUNT=$SAVE_PATH/fast_mount
export HF_DATASETS_CACHE=$FAST_MOUNT/hf/datasets
export DATASETS_CACHE=$HF_DATASETS_CACHE
export DATASETS_TMPDIR=$FAST_MOUNT/tmp
export PYARROW_TMP_DIR=$FAST_MOUNT/tmp

mkdir -p "$HF_DATASETS_CACHE" "$DATASETS_TMPDIR"
echo "[disk] HF_DATASETS_CACHE=$HF_DATASETS_CACHE TMPDIR=$TMPDIR"

# Build the runtime environment JSON with proper variable substitution
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
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}"
  }
}
JSON
)"

echo "RUNTIME_ENV_JSON = $RUNTIME_ENV_JSON"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${ALGORITHM_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
