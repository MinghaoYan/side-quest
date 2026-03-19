"""PACEvolve evaluation helpers for the MULTI-evolve extrapolation task."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Optional


logger = logging.getLogger("controller")


@dataclasses.dataclass
class CompletedProcess:
    args: str
    returncode: int
    stdout: str
    stderr: str


def _call_shell_command(command: str, timeout: int, max_retries: int) -> Optional[CompletedProcess]:
    for _ in range(max_retries):
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return CompletedProcess(
                args=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            continue
    return None


@dataclasses.dataclass
class EvalConfig:
    """Evaluation configuration for the public MULTI-evolve extrapolation benchmark."""

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


def _build_command(config: dict, syntax_only: bool = False) -> str:
    eval_path = os.path.expanduser(config["paths"]["eval_path"])
    src_path = os.path.expanduser(config["paths"]["src_path"])
    data_path = os.path.expanduser(config["paths"]["data_path"])
    eval_script = os.path.join(eval_path, config["evaluation"]["eval_script_name"])
    candidate_script = os.path.join(src_path, config["paths"]["target_file_path"])
    benchmark_level = config["evaluation"].get("benchmark_level", "lite")
    benchmark_protocol = config["evaluation"].get("benchmark_protocol", "paper")
    cuda_prefix = _get_cuda_prefix(config)
    python_executable = _get_python_executable()
    command = (
        f"{cuda_prefix}{python_executable} {shlex.quote(eval_script)} "
        f"--candidate_path {shlex.quote(candidate_script)} "
        f"--data_dir {shlex.quote(data_path)} "
        f"--benchmark_level {shlex.quote(str(benchmark_level))} "
        f"--benchmark_protocol {shlex.quote(str(benchmark_protocol))}"
    )
    if syntax_only:
        command += " --syntax_only"
    return command


def recompile_library(config: dict) -> CompletedProcess:
    command = _build_command(config, syntax_only=True)
    process_result = _call_shell_command(
        command,
        timeout=config["compilation"]["recompile_timeout"],
        max_retries=config["compilation"]["recompile_max_retries"],
    )
    if not process_result:
        return CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr="Compilation command failed to complete.",
        )
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
    results_path = os.path.expanduser(config["paths"]["results_path"])
    results_dir = os.path.join(results_path, eval_config.dataset)
    eval_command = _build_command(config, syntax_only=False)

    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError as exc:
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=f"Could not create results directory {results_dir}: {exc}",
        )

    process_result = _call_shell_command(
        eval_command,
        timeout=config["evaluation"]["eval_timeout"],
        max_retries=config["evaluation"]["eval_max_retries"],
    )
    if not process_result:
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete.",
        )
    return process_result


def parse_eval_metrics(eval_results) -> Optional[dict[str, Any]]:
    if isinstance(eval_results, str):
        match = re.search(r"Candidate:\s*(\{.+\})", eval_results)
        if not match:
            logger.error(f"Pattern not found in the string: '{eval_results}'")
            return None
        try:
            payload = json.loads(match.group(1))
            if not isinstance(payload, dict):
                logger.error("Parsed evaluation payload is not a dictionary.")
                return None
            return payload
        except Exception as exc:
            logger.error(f"Could not parse evaluation metrics: {exc}")
            return None

    if isinstance(eval_results, list):
        parsed_results = []
        for result in eval_results:
            parsed_val = parse_eval_metrics(result)
            if parsed_val is not None:
                parsed_results.append(parsed_val)
        if len(parsed_results) == 1:
            return parsed_results[0]
        if not parsed_results:
            return None
        return {"results": parsed_results}

    raise ValueError("Input must be a string or a list of strings.")


def parse_eval_results(eval_results):
    payload = parse_eval_metrics(eval_results)
    if payload is None:
        return None

    if isinstance(payload, dict) and "combined_score" in payload:
        try:
            return float(payload["combined_score"])
        except Exception as exc:
            logger.error(f"Could not read combined_score from evaluation metrics: {exc}")
            return None

    if isinstance(eval_results, str):
        return None

    if isinstance(eval_results, list):
        parsed_results = []
        for result in eval_results:
            parsed_val = parse_eval_results(result)
            if parsed_val is not None:
                parsed_results.append(parsed_val)
        if len(parsed_results) == 1:
            return parsed_results[0]
        if not parsed_results:
            return None
        return parsed_results

    raise ValueError("Input must be a string or a list of strings.")
