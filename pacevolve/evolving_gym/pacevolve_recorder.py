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

"""Recorder for PACEvolve gym - writes policy outputs, metrics JSON per step."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PACEvolveRecorder:
    """Records policy outputs and metrics per rollout step."""

    def __init__(self, gym, output_dir: str):
        self.gym = gym
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"PACEvolveRecorder initialized with output_dir={output_dir}")

    def record_step(
        self,
        rollout_id: int,
        step_metrics: Optional[Dict[str, Any]] = None,
        data: Optional[List[List[Any]]] = None,
    ) -> None:
        """
        Record rollout step: write policy outputs and metrics.json.

        Args:
            rollout_id: Training step / rollout id.
            step_metrics: Rollout-level metrics (avg_reward, success_rate, batch_operations).
            data: Flattened list of sample groups. Each group is list of Sample.
        """
        step_dir = os.path.join(
            self.output_dir, f"step_{rollout_id:05d}"
        )
        step_dir = os.path.abspath(step_dir)
        os.makedirs(step_dir, exist_ok=True)
        print(f"[PACEvolveRecorder] Writing to step_dir={step_dir}", flush=True)

        # Write policy outputs (sample.response)
        candidates = []
        if data is not None:
            sample_index = 0
            for group in data:
                for sample in group:
                    policy_path = os.path.join(
                        step_dir, f"policy_{sample_index:04d}.txt"
                    )
                    try:
                        response = getattr(sample, "response", None) or ""
                        with open(policy_path, "w", encoding="utf-8") as f:
                            f.write(response)
                    except Exception as e:
                        logger.warning(
                            f"Failed to write policy_{sample_index}: {e}"
                        )

                    # Build candidate info from reward dict
                    reward_val = None
                    metrics = None
                    child_id = None
                    error_val = None
                    if hasattr(sample, "reward"):
                        r = sample.reward
                        if isinstance(r, dict):
                            reward_val = r.get("reward")
                            metrics = r.get("metrics")
                            child_id = r.get("child_id")
                            error_val = r.get("error")
                        elif isinstance(r, (int, float)):
                            reward_val = float(r)

                    score = None
                    if metrics and isinstance(metrics, dict):
                        score = metrics.get("combined_score")

                    candidates.append({
                        "sample_index": sample_index,
                        "score": score,
                        "reward": reward_val,
                        "child_id": child_id,
                        "error": error_val,
                    })
                    sample_index += 1
        else:
            print(
                f"[PACEvolveRecorder] WARNING: data is None, no policy files written",
                flush=True,
            )

        # Build metrics.json (rollout + candidates; train filled later by trainer)
        metrics_obj = {
            "rollout_id": rollout_id,
            "timestamp": time.time(),
            "rollout": step_metrics or {},
            "candidates": candidates,
            "train": {},  # Filled by model.py when training completes
        }

        metrics_path = os.path.join(step_dir, "metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_obj, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write metrics.json: {e}")

        logger.info(f"PACEvolveRecorder: recorded step {rollout_id} to {step_dir}")


def update_step_train_metrics(
    record_dir: str, rollout_id: int, log_dict: Dict[str, Any]
) -> None:
    """
    Update the train block in metrics.json for a step. Called from the trainer
    after each training step.

    Args:
        record_dir: Base records directory.
        rollout_id: Rollout/step id.
        log_dict: Training metrics (grad_norm, loss, pg_loss, etc.).
    """
    step_dir = os.path.join(record_dir, f"step_{rollout_id:05d}")
    metrics_path = os.path.join(step_dir, "metrics.json")

    # Extract train/ prefixed keys and flatten for JSON
    train_metrics = {}
    for k, v in log_dict.items():
        if k.startswith("train/"):
            key = k.replace("train/", "").replace("-", "_")
            if hasattr(v, "item"):
                train_metrics[key] = float(v.item())
            elif isinstance(v, (int, float)):
                train_metrics[key] = float(v)
            else:
                train_metrics[key] = v
        elif "/" not in k:
            if hasattr(v, "item"):
                train_metrics[k] = float(v.item())
            elif isinstance(v, (int, float)):
                train_metrics[k] = float(v)

    if not train_metrics:
        return

    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_obj = json.load(f)
        else:
            metrics_obj = {
                "rollout_id": rollout_id,
                "timestamp": time.time(),
                "rollout": {},
                "candidates": [],
                "train": {},
            }
        metrics_obj["train"] = train_metrics
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to update train metrics: {e}")
