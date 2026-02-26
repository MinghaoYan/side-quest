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

"""Context for PACEvolve per-sample recording (rollout_id, sample_index)."""

from contextvars import ContextVar
from typing import Optional

# Current record context: {rollout_id, sample_index, record_dir} when recording a specific sample.
# Set by the rollout flow before RM calls; read by the gym when creating Transcript.
_pacevolve_record_context: ContextVar[Optional[dict]] = ContextVar(
    "pacevolve_record_context", default=None
)


def get_record_context() -> Optional[dict]:
    """Return current record context, or None if not recording this sample."""
    return _pacevolve_record_context.get()


def set_record_context(rollout_id: int, sample_index: int, record_dir: str) -> None:
    """Set the record context for the current async task."""
    _pacevolve_record_context.set({
        "rollout_id": rollout_id,
        "sample_index": sample_index,
        "record_dir": record_dir,
    })


def clear_record_context() -> None:
    """Clear the record context."""
    try:
        _pacevolve_record_context.set(None)
    except LookupError:
        pass


def set_rollout_id(rollout_id: int) -> None:
    """Set rollout_id in context; sample_index set per-call in pacevolve_gym_rm."""
    _pacevolve_record_context.set({"rollout_id": rollout_id})


def get_rollout_id() -> Optional[int]:
    """Get rollout_id from context, if set."""
    ctx = _pacevolve_record_context.get()
    return ctx.get("rollout_id") if ctx else None
