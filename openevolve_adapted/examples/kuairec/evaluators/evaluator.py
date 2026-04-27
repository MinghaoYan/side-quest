"""OpenEvolve evaluator for the KuaRec FuXi-linear task."""

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
    clamped = max(lo, min(hi, score))
    linear = (clamped - lo) / (hi - lo)
    return float((linear ** alpha) * multiplier)


def _timeout_seconds(config: dict[str, Any]) -> int:
    evaluator_cfg = config.get("evaluator", {}) if isinstance(config, dict) else {}
    try:
        timeout = int(evaluator_cfg.get("timeout", 1200))
    except Exception:
        timeout = 1200
    return max(timeout, 1)


def _get_eval_gpu_queue() -> queue.Queue[str] | None:
    global _GPU_QUEUE
    gpu_ids = [item.strip() for item in os.environ.get("THETAEVOLVE_EVAL_GPU_IDS", "").split(",") if item.strip()]
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


def _validation_error(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return "Could not parse KuaRec evaluation payload."
    if not payload.get("valid_run", True):
        return f"Candidate reported invalid run: {payload.get('failure_reason', 'unknown reason')}"
    if not payload.get("within_budget", False):
        return "Candidate exceeded the fixed runtime budget."
    try:
        score = float(payload["combined_score"])
    except Exception:
        return "Candidate did not report a valid combined_score."
    if not math.isfinite(score):
        return "Candidate produced a non-finite combined_score."

    required = ("ndcg@10", "ndcg@50", "hr@10", "hr@50", "mrr")
    try:
        ndcg10, ndcg50, hr10, hr50, mrr = [float(payload[name]) for name in required]
    except Exception:
        return "Candidate did not report all required KuaRec submetrics."
    if not all(math.isfinite(value) for value in (ndcg10, ndcg50, hr10, hr50, mrr)):
        return "Candidate produced a non-finite KuaRec submetric."

    tie_values = (
        float(payload.get("mean_target_tie_count", 0.0)),
        float(payload.get("max_target_tie_count", 0.0)),
        float(payload.get("frac_target_tie_gt1", 0.0)),
        float(payload.get("frac_target_tie_ge10", 0.0)),
    )
    if not all(math.isfinite(value) for value in tie_values):
        return "Candidate produced non-finite tie diagnostics."

    collapsed = max(abs(hr10 - hr50), abs(hr10 - ndcg10), abs(ndcg10 - ndcg50)) <= 1e-8
    if collapsed and abs(mrr - hr10) <= 1e-3 and score > 0.20 and (
        tie_values[0] > 100.0 or tie_values[1] > 1000.0
    ):
        return "Candidate was rejected for implausibly collapsed high-score ranking metrics."
    return None


def _error_metrics(message: str, elapsed: float = 0.0, stdout: str = "", stderr: str = "") -> dict[str, Any]:
    metrics = {
        "combined_score": 0.0,
        "ndcg@10": 0.0,
        "ndcg@50": 0.0,
        "hr@10": 0.0,
        "hr@50": 0.0,
        "mrr": 0.0,
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
    dataset_csv = _resolve_path(variables.get("dataset_csv"))
    if dataset_csv is None or not dataset_csv.exists():
        return _error_metrics(f"Missing KuaRec dataset CSV: {dataset_csv}", time.time() - started)

    env = os.environ.copy()
    if temp_dir:
        env["PACEVOLVE_ARTIFACT_DIR"] = str(Path(temp_dir) / "artifacts")
    command = [sys.executable, program_path, "--dataset_csv", str(dataset_csv)]

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
            f"KuaRec evaluation timed out after {_timeout_seconds(config)}s",
            elapsed,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
    except Exception as exc:
        return _error_metrics(str(exc), time.time() - started)

    elapsed = time.time() - started
    payload = _parse_candidate_metrics(result.stdout)
    if result.returncode != 0:
        return _error_metrics(
            f"KuaRec subprocess exited with code {result.returncode}",
            elapsed,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    invalid_reason = _validation_error(payload)
    if invalid_reason is not None:
        return _error_metrics(invalid_reason, elapsed, stdout=result.stdout, stderr=result.stderr)

    metrics = dict(payload or {})
    score = float(metrics["combined_score"])
    metrics["combined_score"] = score
    metrics["validity"] = 1.0
    metrics["eval_time"] = float(elapsed)
    metrics["program_runtime"] = float(elapsed)
    metrics["rl_normalized_reward"] = _rl_normalized_reward(score, config)
    return metrics
