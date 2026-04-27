#!/bin/bash
# Generic launcher for standalone ThetaEvolve/OpenEvolve tasks with SLIME GRPO.
#
# Examples:
#   TASK=eplb bash run_thetaevolve_task.sh
#   TASK=multievolve_extrapolate bash run_thetaevolve_task.sh
#   TASK=kuairec SMALL_MODEL_NAME=qwen3_4b_thinking_2507 bash run_thetaevolve_task.sh
#
# This launcher uses scripts_evolve/*/general_thetaevolve.sh, which selects
# --evolving-gym and --rm-type evolving-gym. It intentionally does not use the
# PACEvolve gym or PACEvolve context-management workflow.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export SAVE_PATH="${SAVE_PATH:-/workspace/logs}"

SMALL_MODEL_NAME="${SMALL_MODEL_NAME:-dpsk_distill_qwen3_8b}"
TASK="${TASK:-eplb}"
IS_TRAINING="${IS_TRAINING:-True}"
REWARD_PROCESS_TYPE="${REWARD_PROCESS_TYPE:-rl_normalized_reward}"
LAZY_OUTPUT_PENALTY="${LAZY_OUTPUT_PENALTY:-1}"
SEED="${SEED:-3407}"
NOTE="${NOTE:-}"

THETAEVOLVE_N_SAMPLES_PER_PROMPT="${THETAEVOLVE_N_SAMPLES_PER_PROMPT:-8}"
export THETAEVOLVE_N_SAMPLES_PER_PROMPT
export THETAEVOLVE_ROLLOUT_BATCH_SIZE="${THETAEVOLVE_ROLLOUT_BATCH_SIZE:-1}"
export THETAEVOLVE_NUM_ROLLOUT="${THETAEVOLVE_NUM_ROLLOUT:-10000000}"
export THETAEVOLVE_MAX_CONCURRENT_EVALS="${THETAEVOLVE_MAX_CONCURRENT_EVALS:-8}"

MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

WANDB_API_KEY="${WANDB_API_KEY:-aaa}"
WANDB_ENTITY="${WANDB_ENTITY:-bbb}"
WANDB_PROJECT="${WANDB_PROJECT:-ccc}"
export WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT

MULTIEVOLVE_BENCHMARK_LEVEL="${MULTIEVOLVE_BENCHMARK_LEVEL:-lite}"
MULTIEVOLVE_BENCHMARK_PROTOCOL="${MULTIEVOLVE_BENCHMARK_PROTOCOL:-paper}"
MULTIEVOLVE_DATA_PATH="${MULTIEVOLVE_DATA_PATH:-${ROOT_DIR}/pacevolve/tasks/multievolve_extrapolate/data}"
MULTIEVOLVE_SOURCE_DATASET_DIR="${MULTIEVOLVE_SOURCE_DATASET_DIR:-${ROOT_DIR}/MULTI-evolve/data/benchmark/datasets}"
MULTIEVOLVE_AUTO_DOWNLOAD="${MULTIEVOLVE_AUTO_DOWNLOAD:-0}"

case "${SMALL_MODEL_NAME}" in
  dpsk_prorl_v2_1.5b)
    MODEL_FAMILY="nvidia"
    MODEL_NAME="Nemotron-Research-Reasoning-Qwen-1.5B"
    MODELS_FILE_NAME="deepseek-r1-distill-qwen-1.5B.sh"
    ;;
  dpsk_distill_qwen3_8b)
    MODEL_FAMILY="deepseek-ai"
    MODEL_NAME="DeepSeek-R1-0528-Qwen3-8B"
    MODELS_FILE_NAME="qwen3-8B.sh"
    ;;
  qwen3_4b_thinking_2507|qwen3-4B-Thinking-2507|Qwen3-4B-Thinking-2507)
    MODEL_FAMILY="Qwen"
    MODEL_NAME="Qwen3-4B-Thinking-2507"
    MODELS_FILE_NAME="qwen3-4B-Thinking-2507.sh"
    ;;
  *)
    echo "Unknown SMALL_MODEL_NAME: ${SMALL_MODEL_NAME}"
    echo "Supported values: dpsk_prorl_v2_1.5b, dpsk_distill_qwen3_8b, qwen3_4b_thinking_2507"
    exit 1
    ;;
esac

case "${TASK}" in
  eplb)
    TASK_ID="eplb"
    TASK_LABEL="EPLB"
    EXAMPLE_DIR="${ROOT_DIR}/openevolve_adapted/examples/eplb"
    CONFIG_TEMPLATE="${EXAMPLE_DIR}/configs/config_eplb_thetaevolve.yaml"
    INITIAL_PROGRAM="${EXAMPLE_DIR}/initial_programs/initial_program.py"
    EVALUATOR_FILE="${EXAMPLE_DIR}/evaluators/evaluator.py"
    EPLB_DATA_PATH="${EPLB_DATA_PATH:-${ROOT_DIR}/pacevolve/tasks/eplb/data}"
    EPLB_WORKLOAD_PATH="${EPLB_WORKLOAD_PATH:-${EPLB_DATA_PATH}/expert-load.json}"
    if [ ! -f "${EPLB_WORKLOAD_PATH}" ]; then
      echo "Missing EPLB workload: ${EPLB_WORKLOAD_PATH}"
      echo "Download: wget https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json"
      echo "Then place it under ${EPLB_DATA_PATH}, or set EPLB_WORKLOAD_PATH."
      exit 1
    fi
    ;;
  kuairec)
    TASK_ID="kuairec"
    TASK_LABEL="KUAIREC"
    EXAMPLE_DIR="${ROOT_DIR}/openevolve_adapted/examples/kuairec"
    CONFIG_TEMPLATE="${EXAMPLE_DIR}/configs/config_kuairec_thetaevolve.yaml"
    INITIAL_PROGRAM="${EXAMPLE_DIR}/initial_programs/initial_program.py"
    EVALUATOR_FILE="${EXAMPLE_DIR}/evaluators/evaluator.py"
    KUAIREC_DATASET_CSV="${KUAIREC_DATASET_CSV:-${ROOT_DIR}/pacevolve/tasks/kuairec/data/sasrec_format.csv}"
    if [ ! -f "${KUAIREC_DATASET_CSV}" ]; then
      echo "Missing KuaRec processed CSV: ${KUAIREC_DATASET_CSV}"
      echo "Set KUAIREC_DATASET_CSV=/path/to/sasrec_format.csv or prepare the task data first."
      exit 1
    fi
    ;;
  multievolve|multi-evolve|multievolve_extrapolate)
    TASK_ID="multievolve_extrapolate"
    TASK_LABEL="MULTIEVOLVE_EXTRAPOLATE"
    EXAMPLE_DIR="${ROOT_DIR}/openevolve_adapted/examples/multievolve_extrapolate"
    CONFIG_TEMPLATE="${EXAMPLE_DIR}/configs/config_multievolve_extrapolate_thetaevolve.yaml"
    INITIAL_PROGRAM="${EXAMPLE_DIR}/initial_programs/initial_program.py"
    EVALUATOR_FILE="${EXAMPLE_DIR}/evaluators/evaluator.py"
    DOWNLOAD_SCRIPT="${ROOT_DIR}/pacevolve/tasks/multievolve_extrapolate/data/download_public_data.py"
    PREPARE_SCRIPT="${ROOT_DIR}/pacevolve/tasks/multievolve_extrapolate/data/prepare_public_benchmark.py"
    PREPARED_SUMMARY="${MULTIEVOLVE_DATA_PATH}/prepared/${MULTIEVOLVE_BENCHMARK_LEVEL}/${MULTIEVOLVE_BENCHMARK_PROTOCOL}/benchmark_summary.json"
    if [ ! -f "${PREPARED_SUMMARY}" ]; then
      echo "Prepared MULTI-evolve benchmark not found: ${PREPARED_SUMMARY}"
      RAW_DIR="${MULTIEVOLVE_DATA_PATH}/raw"
      if [ ! -d "${RAW_DIR}" ] || [ -z "$(find "${RAW_DIR}" -maxdepth 1 -name '*.csv' -print -quit 2>/dev/null)" ]; then
        if [ -d "${MULTIEVOLVE_SOURCE_DATASET_DIR}" ] && [ -n "$(find "${MULTIEVOLVE_SOURCE_DATASET_DIR}" -maxdepth 1 -name '*.csv' -print -quit 2>/dev/null)" ]; then
          "${PYTHON_BIN}" "${DOWNLOAD_SCRIPT}" \
            --benchmark-level "${MULTIEVOLVE_BENCHMARK_LEVEL}" \
            --data-dir "${MULTIEVOLVE_DATA_PATH}" \
            --source-dir "${MULTIEVOLVE_SOURCE_DATASET_DIR}"
        elif [ "${MULTIEVOLVE_AUTO_DOWNLOAD}" = "1" ]; then
          "${PYTHON_BIN}" "${DOWNLOAD_SCRIPT}" \
            --benchmark-level "${MULTIEVOLVE_BENCHMARK_LEVEL}" \
            --data-dir "${MULTIEVOLVE_DATA_PATH}"
        else
          echo "No local raw MULTI-evolve CSVs were found."
          echo "Expected source dataset dir: ${MULTIEVOLVE_SOURCE_DATASET_DIR}"
          echo "Populate it or rerun with MULTIEVOLVE_AUTO_DOWNLOAD=1."
          exit 1
        fi
      fi
      "${PYTHON_BIN}" "${PREPARE_SCRIPT}" \
        --benchmark-level "${MULTIEVOLVE_BENCHMARK_LEVEL}" \
        --benchmark-protocol "${MULTIEVOLVE_BENCHMARK_PROTOCOL}" \
        --data-dir "${MULTIEVOLVE_DATA_PATH}"
    fi
    ;;
  *)
    echo "Unsupported TASK: ${TASK}"
    echo "Supported values: eplb, kuairec, multievolve_extrapolate"
    exit 1
    ;;
esac

if [ ! -f "${CONFIG_TEMPLATE}" ] || [ ! -f "${INITIAL_PROGRAM}" ] || [ ! -f "${EVALUATOR_FILE}" ]; then
  echo "Missing ThetaEvolve task files under ${EXAMPLE_DIR}"
  exit 1
fi

POSTFIX_STR="_seed${SEED}${NOTE}"
REWARD_SUFFIX="_${REWARD_PROCESS_TYPE}"
RUN_NAME="${SMALL_MODEL_NAME}_thetaevolve_${TASK_ID}_grpo${REWARD_SUFFIX}${POSTFIX_STR}"

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

RUNTIME_CONFIG_YAML="${SAVE_PATH}/${RUN_NAME}/config_${TASK_ID}_thetaevolve_runtime.yaml"
"${PYTHON_BIN}" - "${CONFIG_TEMPLATE}" "${RUNTIME_CONFIG_YAML}" "${TASK_ID}" "${SEED}" "${THETAEVOLVE_N_SAMPLES_PER_PROMPT}" "${THETAEVOLVE_MAX_CONCURRENT_EVALS}" "${MULTIEVOLVE_BENCHMARK_LEVEL}" "${MULTIEVOLVE_BENCHMARK_PROTOCOL}" <<'PY'
import os
import sys
import yaml

template_path, out_path, task_id, seed, n_samples, max_concurrent_evals, me_level, me_protocol = sys.argv[1:]

with open(template_path, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["random_seed"] = int(seed)
cfg.setdefault("database", {})
cfg["database"]["num_islands"] = int(n_samples)
cfg.setdefault("evaluator", {})
cfg["evaluator"]["parallel_evaluations"] = int(max_concurrent_evals)

variables = cfg.setdefault("variables", {})
root = os.getcwd()

def abs_path(path):
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(root, path))
    return path

if task_id == "eplb":
    data_path = os.environ.get("EPLB_DATA_PATH", variables.get("data_path"))
    workload_path = os.environ.get("EPLB_WORKLOAD_PATH", variables.get("workload_path"))
    variables["data_path"] = abs_path(data_path)
    variables["workload_path"] = abs_path(workload_path)
elif task_id == "kuairec":
    dataset_csv = os.environ.get("KUAIREC_DATASET_CSV", variables.get("dataset_csv"))
    variables["dataset_csv"] = abs_path(dataset_csv)
elif task_id == "multievolve_extrapolate":
    data_dir = os.environ.get("MULTIEVOLVE_DATA_PATH", variables.get("data_dir"))
    variables["data_dir"] = abs_path(data_dir)
    variables["benchmark_level"] = me_level
    variables["benchmark_protocol"] = me_protocol

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

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
    cp -R "${MODEL_LOCAL_PATH}" "${SAVE_SHM_DIR}/${MODEL_NAME}"
  else
    hf download "${MODEL_FAMILY}/${MODEL_NAME}" --local-dir "./${MODEL_NAME}"
    cp -r "${MODEL_NAME}" "${SAVE_SHM_DIR}/"
  fi
fi

source "scripts/models/${MODELS_FILE_NAME}"
if [ ! -d "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist" ] || [ "${FORCE_DOWNLOAD}" -eq 1 ]; then
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "${SAVE_SHM_DIR}/${MODEL_NAME}" --save "${SAVE_SHM_DIR}/${MODEL_NAME}_torch_dist"
fi

echo "=== ThetaEvolve/OpenEvolve GRPO Configuration ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "TASK: ${TASK_LABEL}"
echo "INITIAL_PROGRAM: ${INITIAL_PROGRAM}"
echo "EVALUATOR_FILE: ${EVALUATOR_FILE}"
echo "CONFIG_YAML: ${RUNTIME_CONFIG_YAML}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "MODEL: ${MODEL_NAME}"
echo "IS_TRAINING: ${IS_TRAINING}"
echo "REWARD_PROCESS_TYPE: ${REWARD_PROCESS_TYPE}"
echo "LAZY_OUTPUT_PENALTY: ${LAZY_OUTPUT_PENALTY}"
echo "GPU split: train=${THETAEVOLVE_TRAIN_GPUS_PER_NODE:-4}, rollout=${THETAEVOLVE_ROLLOUT_NUM_GPUS:-4}, eval=${THETAEVOLVE_EVAL_GPU_IDS:-8,9,10,11,12,13,14,15}"
echo "NUM_ROLLOUT: ${THETAEVOLVE_NUM_ROLLOUT}"
echo "N_SAMPLES_PER_PROMPT: ${THETAEVOLVE_N_SAMPLES_PER_PROMPT}"
echo "MAX_CONCURRENT_EVALS: ${THETAEVOLVE_MAX_CONCURRENT_EVALS}"
echo "==============================================="

bash "scripts_evolve/${MODEL_NAME}/general_thetaevolve.sh" \
  "${WANDB_PROJECT}" \
  "${RUN_NAME}" \
  "${INITIAL_PROGRAM}" \
  "${EVALUATOR_FILE}" \
  "${RUNTIME_CONFIG_YAML}" \
  "${SAVE_PATH}" \
  "${IS_TRAINING}" \
  "${LAZY_OUTPUT_PENALTY}" \
  "${REWARD_PROCESS_TYPE}" \
  "${SEED}" \
  2>&1 | tee -a "${SAVE_PATH}/${RUN_NAME}/train_log.txt"
