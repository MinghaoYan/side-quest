import dataclasses
import time
import os
from task_utils import CompletedProcess, _call_shell_command
import logging
import re
import shlex

logger = logging.getLogger("controller")
@dataclasses.dataclass
class EvalConfig:
  """Evaluation configuration for a single consistent hashing scenario."""
  dataset: str


def _get_cuda_prefix(config: dict) -> str:
    cuda_visible_devices = str(
      config.get('evaluation', {}).get('cuda_visible_devices', '')
    ).strip()
    if not cuda_visible_devices:
      return ""
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} "

def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config['compilation']
    EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
    EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
    )
    baseline_unversioned = config['evaluation']['kernel_name'].split("-")[0]
    BASELINE_PATH = os.path.join(EVAL_PATH, "baseline", baseline_unversioned+".py")
    KERNEL_BASE_PATH = os.path.join(config['paths']['src_path'], "kernels", config['evaluation']['kernel_name'])
    KERNEL_PATH = os.path.join(KERNEL_BASE_PATH, "kernel.py")
    CONDA_PREFIX = f"/opt/conda/bin/conda run -n {config['compilation']['conda_env']} "
    CUDA_PREFIX = _get_cuda_prefix(config)
    command = (
      f"{CUDA_PREFIX}{CONDA_PREFIX} python {EVAL_SCRIPT} --baseline_path {BASELINE_PATH} --kernel_path {KERNEL_PATH} --baseline_time {config['evaluation']['baseline_time']} --build_dir {KERNEL_BASE_PATH}"
    )
    logger.info(f"recompile_library: Running command: {command}")
    process_result = _call_shell_command(
      command, timeout=comp_config['recompile_timeout'], max_retries=comp_config['recompile_max_retries']
    )

    if not process_result:
      return CompletedProcess(
        args=command,
        returncode=-1,
        stdout="",
        stderr="Compilation command failed to complete.",
      )

    success = (process_result.returncode == 0)
    logger.info(f"recompile_library: Success: {success}.")
    for line in process_result.stdout.splitlines():
      logger.debug(f"recompile_library: STDOUT: {line}")
    for line in process_result.stderr.splitlines():
      logger.debug(f"recompile_library: STDERR: {line}")
    return CompletedProcess(
      args=command,
      returncode=process_result.returncode,
      stdout=process_result.stdout.strip(),
      stderr=process_result.stderr.strip()
    )


def evaluate_dataset(
  candidate_id: int,
  baseline_id: int,
  eval_config: EvalConfig,
  config: dict,
) -> CompletedProcess:
  EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
  EVAL_SCRIPT = os.path.join(
    EVAL_PATH, config['evaluation']['eval_script_name']
  )
  baseline_unversioned = config['evaluation']['kernel_name'].split("-")[0]
  BASELINE_PATH = os.path.join(EVAL_PATH, "baseline", baseline_unversioned+".py")
  KERNEL_BASE_PATH = os.path.join(config['paths']['src_path'], "kernels", config['evaluation']['kernel_name'])
  KERNEL_PATH = os.path.join(KERNEL_BASE_PATH, "kernel.py")
  CONDA_PREFIX = f"/opt/conda/bin/conda run -n {config['compilation']['conda_env']} "
  CUDA_PREFIX = _get_cuda_prefix(config)
  eval_command = (
    f"{CUDA_PREFIX}{CONDA_PREFIX} python {EVAL_SCRIPT} --baseline_path {BASELINE_PATH} --kernel_path {KERNEL_PATH} --baseline_time {config['evaluation']['baseline_time']} --build_dir {KERNEL_BASE_PATH}"
  )

  logger.info(f"evaluate_dataset: Running {eval_command}")
  process_result_eval = _call_shell_command(
    eval_command, timeout=config['evaluation']['eval_timeout'], max_retries=config['evaluation']['eval_max_retries']
  )
  if not process_result_eval:
    logger.error(
      f"evaluate_dataset: evaluate_dataset for {eval_config.dataset} failed."
    )
    return CompletedProcess(
      args=eval_command,
      returncode=-1,
      stdout="",
      stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete."
    )

  return process_result_eval


def parse_eval_results(
  eval_results: list[str] | str,
) -> list[float | None] | float | None:
  """
  Parses a string or list of strings to extract the AUC value from the 'Candidate' section.
  """
  if isinstance(eval_results, str):
    # This pattern specifically finds "AUC: " and captures the floating-point number after it.
    pattern = r"Kernel speedup:\s*(-?\d+\.\d+)"

    match = re.search(pattern, eval_results)

    if match:
        # match.group(1) is the captured AUC value string (e.g., "0.754063")
        auc_str = match.group(1)
        try:
            return float(auc_str)
        except ValueError:
            # This is unlikely to happen with this specific regex but is good practice
            logger.error(f"Could not convert captured value '{auc_str}' to a float.")
            return None

    logger.error(f"Pattern not found in the string: '{eval_results}'")
    return None
  
  elif isinstance(eval_results, list):
    parsed_results = []
    for result in eval_results:
      parsed_val = parse_eval_results(result)
      if parsed_val is not None:
          parsed_results.append(parsed_val)
          
    if len(parsed_results) == 1:
        return parsed_results[0]
    elif not parsed_results:
        return None
    else:
        return parsed_results
  
  else:
    raise ValueError("Input must be a string or a list of strings.")
