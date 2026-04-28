"""OpenEvolve evaluator for the public MULTI-evolve extrapolation task."""

from __future__ import annotations

import json
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
_GPU_LOCK = threading.Lock()


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


def _timeout_seconds(config: dict[str, Any]) -> int:
    evaluator_cfg = config.get("evaluator", {}) if isinstance(config, dict) else {}
    try:
        timeout = int(evaluator_cfg.get("timeout", 1200))
    except Exception:
        timeout = 1200
    return max(timeout, 1)


def _get_eval_gpu_queue() -> queue.Queue[str] | None:
    global _GPU_QUEUE
    gpu_ids = [
        item.strip()
        for item in os.environ.get("THETAEVOLVE_EVAL_GPU_IDS", "").split(",")
        if item.strip()
    ]
    if not gpu_ids:
        return None
    with _GPU_LOCK:
        if _GPU_QUEUE is None:
            _GPU_QUEUE = queue.Queue()
            for gpu_id in gpu_ids:
                _GPU_QUEUE.put(gpu_id)
        return _GPU_QUEUE


def _acquire_eval_gpu() -> str | None:
    gpu_queue = _get_eval_gpu_queue()
    if gpu_queue is None:
        return None
    return gpu_queue.get()


def _release_eval_gpu(gpu_id: str | None) -> None:
    if gpu_id is None:
        return
    gpu_queue = _get_eval_gpu_queue()
    if gpu_queue is not None:
        gpu_queue.put(gpu_id)


def _parse_candidate_metrics(stdout: str) -> dict[str, Any] | None:
    matches = re.findall(r"Candidate:\s*(\{.+\})", stdout)
    if not matches:
        return None
    try:
        payload = json.loads(matches[-1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _rl_normalized_reward(score: float, config: dict[str, Any]) -> float:
    if not math.isfinite(score) or score < 0:
        return score if math.isfinite(score) else -1.0
    score_cfg = config.get("score_transform", {}) if isinstance(config, dict) else {}
    lo = float(score_cfg.get("score_range_min", 0.0))
    hi = float(score_cfg.get("score_range_max", 1.0))
    if hi <= lo:
        hi = lo + 1.0
    alpha = float(score_cfg.get("alpha", 1.0))
    if alpha <= 0:
        alpha = 1.0
    multiplier = float(score_cfg.get("positive_multiplier", 1.0))
    if multiplier <= 0:
        multiplier = 1.0
    optimize_mode = str(score_cfg.get("optimize_mode", "maximize")).lower()
    if optimize_mode == "minimize":
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


def _error_metrics(message: str, elapsed: float = 0.0, stdout: str = "", stderr: str = "") -> dict[str, Any]:
    metrics = {
        "combined_score": 0.0,
        "mean_pearson_r": 0.0,
        "mean_precision_top5": 0.0,
        "validity": 0.0,
        "eval_time": float(elapsed),
        "program_runtime": float(elapsed),
        "error": message,
    }
    if stdout:
        metrics["stdout_tail"] = stdout[-2000:]
    if stderr:
        metrics["stderr_tail"] = stderr[-2000:]
    return metrics


def evaluate(program_path: str, temp_dir: str | None = None) -> dict[str, Any]:
    started = time.time()
    config = _load_runtime_config()
    variables = config.get("variables", {}) if isinstance(config, dict) else {}
    data_dir = _resolve_path(variables.get("data_dir"))
    benchmark_level = str(variables.get("benchmark_level", "lite"))
    benchmark_protocol = str(variables.get("benchmark_protocol", "paper"))

    if data_dir is None or not data_dir.exists():
        return _error_metrics(f"Missing MULTI-evolve data dir: {data_dir}", time.time() - started)

    prepared_summary = data_dir / "prepared" / benchmark_level / benchmark_protocol / "benchmark_summary.json"
    if not prepared_summary.exists():
        return _error_metrics(f"Missing prepared benchmark summary: {prepared_summary}", time.time() - started)

    eval_script = (
        _repo_root()
        / "pacevolve"
        / "tasks"
        / "multievolve_extrapolate"
        / "eval"
        / "evaluate_multievolve_extrapolate.py"
    )
    command = [
        sys.executable,
        str(eval_script),
        "--candidate_path",
        program_path,
        "--data_dir",
        str(data_dir),
        "--benchmark_level",
        benchmark_level,
        "--benchmark_protocol",
        benchmark_protocol,
    ]

    env = os.environ.copy()
    if temp_dir:
        env["PACEVOLVE_ARTIFACT_DIR"] = str(Path(temp_dir) / "artifacts")

    eval_gpu_id = _acquire_eval_gpu()
    if eval_gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = eval_gpu_id

    try:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_timeout_seconds(config),
                env=env,
                check=False,
            )
        finally:
            _release_eval_gpu(eval_gpu_id)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        return _error_metrics(
            f"MULTI-evolve evaluation timed out after {_timeout_seconds(config)}s",
            elapsed,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
        )
    except Exception as exc:
        return _error_metrics(str(exc), time.time() - started)

    elapsed = time.time() - started
    if result.returncode != 0:
        return _error_metrics(
            f"MULTI-evolve subprocess exited with code {result.returncode}",
            elapsed,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    raw = _parse_candidate_metrics(result.stdout)
    metrics = dict(raw or {})
    if "combined_score" not in metrics:
        return _error_metrics(
            "MULTI-evolve evaluator did not return combined_score",
            elapsed,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    try:
        score = float(metrics["combined_score"])
    except Exception:
        return _error_metrics(f"Invalid combined_score: {metrics.get('combined_score')}", elapsed)

    metrics["combined_score"] = score
    metrics["validity"] = 1.0
    metrics["eval_time"] = float(elapsed)
    metrics["program_runtime"] = float(elapsed)
    metrics["rl_normalized_reward"] = _rl_normalized_reward(score, config)
    return metrics
