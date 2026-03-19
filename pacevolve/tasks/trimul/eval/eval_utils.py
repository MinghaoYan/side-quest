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
    """Evaluation configuration for the TriMul dataset."""

    dataset: str


def _get_python_executable() -> str:
    python_executable = sys.executable or "python"
    return shlex.quote(python_executable)


def _build_command(config: dict, syntax_only: bool = False) -> str:
    eval_path = os.path.expanduser(config["paths"]["eval_path"])
    src_path = os.path.expanduser(config["paths"]["src_path"])
    eval_script = os.path.join(eval_path, config["evaluation"]["eval_script_name"])
    candidate_script = os.path.join(src_path, config["paths"]["target_file_path"])
    python_executable = _get_python_executable()
    command = (
        f"{python_executable} {shlex.quote(eval_script)} "
        f"--candidate_path {shlex.quote(candidate_script)}"
    )
    cuda_visible_devices = str(
        config.get("evaluation", {}).get("cuda_visible_devices", "")
    ).strip()
    if cuda_visible_devices:
        command = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} {command}"
    if syntax_only:
        command += " --syntax_only"
    return command


def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config["compilation"]
    command = _build_command(config, syntax_only=True)
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
    results_path = os.path.expanduser(config["paths"]["results_path"])
    results_dir = os.path.join(results_path, eval_config.dataset)
    eval_command = _build_command(config, syntax_only=False)

    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError as exc:
        logger.error(
            f"evaluate_dataset: Could not create results directory {results_dir}: {exc}"
        )
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=f"Could not create results directory {results_dir}: {exc}",
        )

    logger.info(f"evaluate_dataset: Running {eval_command}")
    process_result = _call_shell_command(
        eval_command,
        timeout=config["evaluation"]["eval_timeout"],
        max_retries=config["evaluation"]["eval_max_retries"],
    )
    if not process_result:
        logger.error(
            f"evaluate_dataset: evaluate_dataset for {eval_config.dataset} failed."
        )
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete.",
        )
    return process_result


def parse_eval_results(
    eval_results: list[str] | str,
) -> list[float | None] | float | None:
    if isinstance(eval_results, str):
        match = re.search(r"Kernel speedup:\s*(-?\d+\.\d+)", eval_results)
        if not match:
            logger.error(f"Pattern not found in the string: '{eval_results}'")
            return None
        try:
            return float(match.group(1))
        except ValueError:
            logger.error(
                f"Could not convert captured value '{match.group(1)}' to a float."
            )
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
