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
import logging
import os
import re
import shlex
import sys

from task_utils import CompletedProcess, _call_shell_command

logger = logging.getLogger("controller")


@dataclasses.dataclass
class EvalConfig:
    """Evaluation configuration for EPLB dataset."""

    dataset: str


def _get_python_executable() -> str:
    python_executable = sys.executable or "python"
    return shlex.quote(python_executable)


def _get_cuda_prefix(config: dict) -> str:
    cuda_visible_devices = str(
        config.get("evaluation", {}).get("cuda_visible_devices", "")
    ).strip()
    if not cuda_visible_devices:
        return ""
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} "


def _build_command(config: dict) -> str:
    eval_path = os.path.expanduser(config["paths"]["eval_path"])
    target_path = os.path.expanduser(config["paths"]["src_path"])
    eval_script = os.path.join(
        eval_path, config["evaluation"]["eval_script_name"]
    )
    candidate_script = os.path.join(
        target_path, config["paths"]["target_file_path"]
    )
    data_path = os.path.expanduser(str(config["paths"]["data_path"]))
    cuda_prefix = _get_cuda_prefix(config)
    python_executable = _get_python_executable()
    return (
        f"{cuda_prefix}{python_executable} {shlex.quote(eval_script)} "
        f"--candidate_path {shlex.quote(candidate_script)} "
        f"--data_path {shlex.quote(data_path)}"
    )


def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config["compilation"]
    command = _build_command(config)
    logger.info(f"recompile_library: Running command: {command}")
    process_result = _call_shell_command(
        command,
        timeout=comp_config["recompile_timeout"],
        max_retries=comp_config["recompile_max_retries"],
    )

    if not process_result:
        return CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr="Compilation command failed to complete.",
        )

    success = process_result.returncode == 0
    logger.info(f"recompile_library: Success: {success}.")
    for line in process_result.stdout.splitlines():
        logger.debug(f"recompile_library: STDOUT: {line}")
    for line in process_result.stderr.splitlines():
        logger.debug(f"recompile_library: STDERR: {line}")
    return CompletedProcess(
        args=command,
        returncode=process_result.returncode,
        stdout=process_result.stdout.strip(),
        stderr=process_result.stderr.strip(),
    )


def evaluate_dataset(
    candidate_id: int,
    baseline_id: int,
    eval_config: EvalConfig,
    config: dict,
) -> CompletedProcess:
    del candidate_id, baseline_id
    RESULTS_PATH = os.path.expanduser(config["paths"]["results_path"])
    results_dir = os.path.join(RESULTS_PATH, eval_config.dataset)
    eval_command = _build_command(config)

    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError as e:
        logger.error(
            f"evaluate_dataset: Could not create results directory {results_dir}: {e}"
        )
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=f"Could not create results directory {results_dir}: {e}",
        )

    logger.info(f"evaluate_dataset: Running {eval_command}")
    process_result_eval = _call_shell_command(
        eval_command,
        timeout=config["evaluation"]["eval_timeout"],
        max_retries=config["evaluation"]["eval_max_retries"],
    )
    if not process_result_eval:
        logger.error(
            f"evaluate_dataset: evaluate_dataset for {eval_config.dataset} failed."
        )
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete.",
        )
    return process_result_eval


def parse_eval_results(
    eval_results: list[str] | str,
) -> list[float] | float | None:
    """Parse EPLB eval output; return combined_score (or None on failure)."""
    if isinstance(eval_results, str):
        pattern = r"Candidate:\s*({.+?})\s*"
        match = re.search(pattern, eval_results)

        if match:
            captured_value_str = match.group(1)
            try:
                clean_str = captured_value_str.replace("np.float64(", "")
                data_dict = eval(clean_str)
                return float(data_dict["combined_score"])
            except Exception as e:
                logger.error(
                    f"parse_eval_results: Failed to parse dictionary from "
                    f"string '{captured_value_str}'. Error: {e}"
                )
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

    return None
