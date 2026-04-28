"""OpenEvolve evaluator for the EPLB task.

This is intentionally a thin standalone adapter around the task evaluator used
by PACEvolve. It exposes OpenEvolve's expected ``evaluate(program_path)`` shape
without invoking any PACEvolve workflow or context-management code.
"""

from __future__ import annotations

import ast
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

_GPU_QUEUE: queue.Queue[str] | None = None
_GPU_QUEUE_KEY: str | None = None
_GPU_QUEUE_LOCK = threading.Lock()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_runtime_config() -> dict[str, Any]:
    config_path = os.environ.get("OPENEVOLVE_CONFIG_PATH")
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _score_transform_config(config: dict[str, Any]) -> dict[str, Any]:
    score_cfg = config.get("score_transform", {}) if isinstance(config, dict) else {}
    return {
        "score_range_min": float(score_cfg.get("score_range_min", 0.0)),
        "score_range_max": float(score_cfg.get("score_range_max", 1.0)),
        "alpha": float(score_cfg.get("alpha", 1.0)),
        "optimize_mode": str(score_cfg.get("optimize_mode", "maximize")).lower(),
        "positive_multiplier": float(score_cfg.get("positive_multiplier", 1.0)),
    }


def _rl_normalized_reward(score: float, config: dict[str, Any]) -> float:
    if not math.isfinite(score) or score < 0:
        return score if math.isfinite(score) else -1.0

    cfg = _score_transform_config(config)
    lo = cfg["score_range_min"]
    hi = cfg["score_range_max"]
    if hi <= lo:
        hi = lo + 1.0
    alpha = cfg["alpha"] if cfg["alpha"] > 0 else 1.0
    multiplier = cfg["positive_multiplier"] if cfg["positive_multiplier"] > 0 else 1.0

    if cfg["optimize_mode"] == "minimize":
        working = -score
        range_min = -hi
        range_max = -lo
    else:
        working = score
        range_min = lo
        range_max = hi

    clamped = max(range_min, min(range_max, working))
    linear = (clamped - range_min) / (range_max - range_min)
    return float((linear ** alpha) * multiplier)


def _eval_timeout(config: dict[str, Any]) -> int:
    evaluator_cfg = config.get("evaluator", {}) if isinstance(config, dict) else {}
    value = evaluator_cfg.get("timeout", evaluator_cfg.get("timeout_s", 600))
    try:
        return int(value)
    except Exception:
        return 600


def _get_eval_gpu_queue() -> queue.Queue[str] | None:
    global _GPU_QUEUE, _GPU_QUEUE_KEY

    gpu_ids_env = os.environ.get("THETAEVOLVE_EVAL_GPU_IDS", "")
    gpu_ids = [gpu_id.strip() for gpu_id in gpu_ids_env.split(",") if gpu_id.strip()]
    queue_key = ",".join(gpu_ids)
    if not gpu_ids:
        return None

    with _GPU_QUEUE_LOCK:
        if _GPU_QUEUE is None or _GPU_QUEUE_KEY != queue_key:
            _GPU_QUEUE = queue.Queue()
            for gpu_id in gpu_ids:
                _GPU_QUEUE.put(gpu_id)
            _GPU_QUEUE_KEY = queue_key
    return _GPU_QUEUE


class _EvalGpuLease:
    def __enter__(self) -> str | None:
        self._queue = _get_eval_gpu_queue()
        self.gpu_id = self._queue.get() if self._queue is not None else None
        return self.gpu_id

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._queue is not None and self.gpu_id is not None:
            self._queue.put(self.gpu_id)


def _parse_candidate_stdout(stdout: str) -> dict[str, Any]:
    match = re.search(r"Candidate:\s*({.+?})\s*$", stdout, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse EPLB evaluator output: {stdout[-1000:]}")

    payload = re.sub(r"np\.float64\(([^()]+)\)", r"\1", match.group(1))
    parsed = ast.literal_eval(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"EPLB evaluator returned non-dict payload: {type(parsed)}")
    return parsed


def _run_pace_eplb_subprocess(
    program_path: str,
    workload_path: Path,
    timeout: int,
    gpu_id: str | None,
) -> dict[str, Any]:
    eval_script = _repo_root() / "pacevolve" / "tasks" / "eplb" / "eval" / "evaluate_eplb.py"
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    pythonpath_parts = [str(_repo_root())]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    command = [
        sys.executable or "python3",
        str(eval_script),
        "--candidate_path",
        program_path,
        "--data_path",
        str(workload_path.parent),
        "--workload_path",
        str(workload_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(_repo_root()),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"returncode={completed.returncode}"
        raise RuntimeError(detail[-2000:])

    metrics = _parse_candidate_stdout(completed.stdout)
    if gpu_id is not None:
        metrics["eval_cuda_visible_devices"] = gpu_id
    return metrics


def _error_metrics(message: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "balancedness_score": 0.0,
        "speed_score": 0.0,
        "validity": 0.0,
        "eval_time": float(elapsed),
        "program_runtime": float(elapsed),
        "error": message,
    }


def evaluate(program_path: str, temp_dir: str | None = None) -> dict[str, Any]:
    del temp_dir
    started = time.time()
    config = _load_runtime_config()
    variables = config.get("variables", {}) if isinstance(config, dict) else {}
    data_path = _resolve_path(variables.get("data_path"))
    workload_path = _resolve_path(variables.get("workload_path"))
    if workload_path is None and data_path is not None:
        workload_path = data_path / "expert-load.json"

    if workload_path is None or not workload_path.exists():
        return _error_metrics(f"Missing EPLB workload file: {workload_path}", time.time() - started)

    try:
        with _EvalGpuLease() as gpu_id:
            raw = _run_pace_eplb_subprocess(
                program_path,
                workload_path,
                _eval_timeout(config),
                gpu_id,
            )
    except subprocess.TimeoutExpired:
        return _error_metrics(f"EPLB evaluator timed out after {_eval_timeout(config)}s", time.time() - started)
    except Exception as exc:
        return _error_metrics(str(exc), time.time() - started)

    elapsed = time.time() - started
    metrics = dict(raw or {})
    if "combined_score" not in metrics:
        return _error_metrics("EPLB evaluator did not return combined_score", elapsed)

    try:
        score = float(metrics["combined_score"])
    except Exception:
        return _error_metrics(f"Invalid combined_score: {metrics.get('combined_score')}", elapsed)

    metrics["combined_score"] = score
    metrics.setdefault("balancedness_score", 0.0)
    metrics.setdefault("speed_score", 0.0)
    metrics["validity"] = 0.0 if metrics.get("error") else 1.0
    metrics["eval_time"] = float(elapsed)
    metrics["program_runtime"] = float(elapsed)
    metrics["rl_normalized_reward"] = _rl_normalized_reward(score, config)
    return metrics
