"""OpenEvolve evaluator for the public MULTI-evolve extrapolation task."""

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


def _load_pace_multievolve_module():
    eval_dir = _repo_root() / "pacevolve" / "tasks" / "multievolve_extrapolate" / "eval"
    eval_path = eval_dir / "evaluate_multievolve_extrapolate.py"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location("thetaevolve_pace_multievolve_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MULTI-evolve evaluator from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["thetaevolve_pace_multievolve_eval"] = module
    spec.loader.exec_module(module)
    return module


def _error_metrics(message: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "mean_pearson_r": 0.0,
        "mean_precision_top5": 0.0,
        "validity": 0.0,
        "eval_time": float(elapsed),
        "program_runtime": float(elapsed),
        "error": message,
    }


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

    previous_artifact_dir = os.environ.get("PACEVOLVE_ARTIFACT_DIR")
    if temp_dir:
        os.environ["PACEVOLVE_ARTIFACT_DIR"] = str(Path(temp_dir) / "artifacts")
    try:
        module = _load_pace_multievolve_module()
        raw = module.evaluate(program_path, str(data_dir), benchmark_level, benchmark_protocol)
    except Exception as exc:
        return _error_metrics(str(exc), time.time() - started)
    finally:
        if previous_artifact_dir is None:
            os.environ.pop("PACEVOLVE_ARTIFACT_DIR", None)
        else:
            os.environ["PACEVOLVE_ARTIFACT_DIR"] = previous_artifact_dir

    elapsed = time.time() - started
    metrics = dict(raw or {})
    if "combined_score" not in metrics:
        return _error_metrics("MULTI-evolve evaluator did not return combined_score", elapsed)

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
