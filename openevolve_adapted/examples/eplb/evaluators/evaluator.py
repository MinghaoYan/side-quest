"""OpenEvolve evaluator for the EPLB task.

This is intentionally a thin standalone adapter around the task evaluator used
by PACEvolve. It exposes OpenEvolve's expected ``evaluate(program_path)`` shape
without invoking any PACEvolve workflow or context-management code.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml


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


def _load_pace_eplb_module():
    eval_path = _repo_root() / "pacevolve" / "tasks" / "eplb" / "eval" / "evaluate_eplb.py"
    spec = importlib.util.spec_from_file_location("thetaevolve_pace_eplb_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load EPLB evaluator from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["thetaevolve_pace_eplb_eval"] = module
    spec.loader.exec_module(module)
    return module


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
        module = _load_pace_eplb_module()
        raw = module.evaluate(program_path, str(workload_path))
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
