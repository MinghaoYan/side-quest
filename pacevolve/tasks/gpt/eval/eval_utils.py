# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import time
import os
from task_utils import CompletedProcess, _call_shell_command
import logging
import re

logger = logging.getLogger("controller")
@dataclasses.dataclass
class EvalConfig:
  """Evaluation configuration for a single consistent hashing scenario."""
  dataset: str

def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config['compilation']
    EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
    EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
    )
    DATA_PATH = os.path.expanduser(config['paths']['data_path'])
    TRAIN_DATA_PATH = os.path.join(
      DATA_PATH, "fineweb10B/fineweb_train_*.bin"
    )
    VAL_DATA_PATH = os.path.join(
      DATA_PATH, "fineweb10B/fineweb_val_*.bin"
    )
    command = (
      f"{EVAL_SCRIPT} {comp_config['conda_env']} '{TRAIN_DATA_PATH}' '{VAL_DATA_PATH}' true"
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
  RESULTS_PATH = os.path.expanduser(config['paths']['results_path'])
  EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
  )
  DATA_PATH = os.path.expanduser(config['paths']['data_path'])
  TRAIN_DATA_PATH = os.path.join(
    DATA_PATH, "fineweb10B/fineweb_train_*.bin"
  )
  VAL_DATA_PATH = os.path.join(
    DATA_PATH, "fineweb10B/fineweb_val_*.bin"
  )
  comp_config = config['compilation']
  eval_command = (
    f"{EVAL_SCRIPT} {comp_config['conda_env']} '{TRAIN_DATA_PATH}' '{VAL_DATA_PATH}' false"
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
    pattern = r"Candidate validation loss:.*?Training time:\s*(\d+\.\d+)"

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
    return [parse_eval_results(result) for result in eval_results]
  
  else:
    raise ValueError("Input must be a string or a list of strings.")
