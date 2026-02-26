"""Reward model adapter for PACEvolve gym.

Same contract as evolving_gym_rm.py but backed by PACEvolveSingleTaskGym.
"""

import math
from typing import Dict, Any

_GYM = None


def set_gym(gym):
    global _GYM
    _GYM = gym


def _get_combined_score(metrics: Dict[str, Any], default_low_score: float) -> float:
    return metrics.get("combined_score", default_low_score)


def _get_parent_combined_score(parent_program, default_low_score: float) -> float:
    if parent_program is None:
        return default_low_score
    if hasattr(parent_program, "metrics") and parent_program.metrics:
        return parent_program.metrics.get("combined_score", default_low_score)
    return default_low_score


def _get_rl_normalized_reward(metrics: Dict[str, Any], default_low_score: float) -> float:
    """Get rl_normalized_reward from metrics."""
    val = metrics.get("rl_normalized_reward")
    if val is not None:
        return val
    print(f"[WARNING] rl_normalized_reward missing in metrics: {metrics}")
    return default_low_score


def _process_reward(
    metrics: Dict[str, Any],
    parent_program,
    reward_process_type: str,
    default_low_score: float,
) -> float:
    if reward_process_type == "original_reward":
        return _get_combined_score(metrics, default_low_score)

    elif reward_process_type == "rl_normalized_reward":
        return _get_rl_normalized_reward(metrics, default_low_score)

    elif reward_process_type == "improve_reward":
        child_score = _get_combined_score(metrics, default_low_score)
        if isinstance(child_score, (int, float)) and child_score < -0.5:
            return child_score
        parent_score = _get_parent_combined_score(parent_program, default_low_score)
        eps = 1e-6
        return 1.0 if child_score > parent_score + eps else 0.0

    else:
        return _get_combined_score(metrics, default_low_score)


def _sanitize_reward(
    reward_val: float,
    default_low_score: float,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
) -> float:
    if math.isnan(reward_val) or math.isinf(reward_val):
        print(f"[WARNING] Invalid reward {reward_val}, using {default_low_score}")
        return default_low_score
    clipped = max(clip_min, min(clip_max, reward_val))
    if abs(reward_val - clipped) > 1e-6:
        print(f"[WARNING] Reward clipped: {reward_val:.2e} -> {clipped:.2f}")
    return clipped


async def pacevolve_gym_rm(args, sample) -> Dict[str, Any]:
    """Score LLM output using PACEvolve gym and return reward dict."""
    default_low_score = -1.0
    reward_key = getattr(args, "reward_key", "reward")

    if _GYM is None:
        return {reward_key: default_low_score, "error": "gym_not_initialized"}

    metadata = sample.metadata or {}
    if isinstance(metadata, str):
        metadata = {}

    parent_program = metadata.get("parent_program", None)
    if not parent_program:
        return {reward_key: default_low_score, "error": "missing_parent_program"}

    # Set record context for per-sample API transcript logging
    if (
        getattr(_GYM, "recording_enabled", False)
        and getattr(_GYM, "set_record_context", None) is not None
    ):
        try:
            from pacevolve.evolving_gym.record_context import get_rollout_id
            rid = get_rollout_id()
            if rid is not None:
                sample_index = metadata.get("record_sample_index")
                if not isinstance(sample_index, int):
                    sample_index = int(getattr(sample, "index", 0))
                _GYM.set_record_context(rid, sample_index)
        except Exception:
            pass

    try:
        result = await _GYM.response_scorer(sample.response or "", parent_program)
    except Exception as e:
        print(f"exception in pacevolve_gym_rm: {e}")
        return {reward_key: default_low_score, "error": f"exception:{str(e)[:200]}"}

    if result is None or not getattr(result, "child_metrics", None):
        return {reward_key: default_low_score}

    metrics = result.child_metrics or {}
    reward_val = _process_reward(
        metrics, parent_program, _GYM.reward_process_type, default_low_score
    )
    reward_val = _sanitize_reward(reward_val, default_low_score)

    # Cache child to temp storage
    try:
        if getattr(result, "child_program", None) is not None:
            _GYM.database.add_temp(
                result.child_program,
                iteration=_GYM.database._iteration_counter,
            )
    except Exception as e:
        print(f"Failed to add to temp cache: {e}")

    out = {
        reward_key: reward_val,
        "metrics": metrics,
        "child_id": getattr(
            getattr(result, "child_program", None), "id", None
        ),
    }
    if getattr(result, "iteration_time", None) is not None:
        out["iteration_time"] = result.iteration_time
    if getattr(result, "artifacts", None):
        out["artifacts"] = result.artifacts

    return out
