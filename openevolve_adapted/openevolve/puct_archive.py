"""
PUCT-based state archive for TTT-Discover.

Implements the PUCT (Predictor + Upper Confidence bounds applied to Trees)
reuse heuristic from the TTT-Discover paper. Each state in the archive is
scored by:

    score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s))

where:
    - Q(s) = max child reward (or R(s) if unexpanded)
    - P(s) = rank-based linear prior
    - n(s) = visitation count (backpropagated through ancestors)
    - T = total number of expansions
    - c = exploration coefficient
    - scale = R_max - R_min over the archive
"""

import logging
import math
import os
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PUCTState:
    """A state in the PUCT archive."""
    state_id: str
    reward: float
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    visit_count: int = 0
    max_child_reward: Optional[float] = None
    is_seed: bool = False
    code: Optional[str] = None
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def q_value(self) -> float:
        """Q(s): max child reward if expanded, else own reward."""
        if self.visit_count > 0 and self.max_child_reward is not None:
            return self.max_child_reward
        return self.reward


class PUCTArchive:
    """
    PUCT-inspired archive for TTT-Discover reuse.

    Maintains an archive of discovered states and selects initial states
    for the next batch using a PUCT-like scoring rule. Tracks lineage
    (parent/child relationships) and backpropagates visitation counts.

    Args:
        exploration_coef: PUCT exploration coefficient c (default 1.0)
        archive_max_size: Maximum number of states to keep (default 1000)
        top_children_per_parent: Number of top children to keep per expansion (default 2)
    """

    def __init__(
        self,
        exploration_coef: float = 1.0,
        archive_max_size: int = 1000,
        top_children_per_parent: int = 2,
    ):
        self.exploration_coef = exploration_coef
        self.archive_max_size = archive_max_size
        self.top_children_per_parent = top_children_per_parent

        # State storage: state_id -> PUCTState
        self.states: Dict[str, PUCTState] = {}

        # Seed state IDs (always retained)
        self.seed_ids: Set[str] = set()

        # Total expansion count
        self.total_expansions: int = 0

        # Thread safety
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_state(
        self,
        state_id: str,
        reward: float,
        parent_id: Optional[str] = None,
        is_seed: bool = False,
        code: Optional[str] = None,
        language: str = "python",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new state to the archive."""
        with self._lock:
            state = PUCTState(
                state_id=state_id,
                reward=reward,
                parent_id=parent_id,
                is_seed=is_seed,
                code=code,
                language=language,
                metadata=metadata or {},
            )
            self.states[state_id] = state

            if is_seed:
                self.seed_ids.add(state_id)

            # Register as child of parent
            if parent_id is not None and parent_id in self.states:
                parent_state = self.states[parent_id]
                if state_id not in parent_state.children_ids:
                    parent_state.children_ids.append(state_id)

            self._enforce_archive_limit()

    def select_states(self, batch_size: int = 1) -> List[str]:
        """
        Select initial states for the next batch using PUCT scoring.

        Selects `batch_size` states, blocking lineage between selections
        within the same batch to encourage diversity.

        Returns:
            List of state_ids selected for expansion.
        """
        with self._lock:
            if not self.states:
                return []

            selected = []
            blocked: Set[str] = set()

            for _ in range(min(batch_size, len(self.states))):
                best_id = None
                best_score = -math.inf

                for sid, state in self.states.items():
                    if sid in blocked:
                        continue
                    score = self._compute_puct_score(sid)
                    if score > best_score:
                        best_score = score
                        best_id = sid

                if best_id is None:
                    break

                selected.append(best_id)

                # Block the selected state's full lineage for this batch
                lineage = self._get_lineage(best_id)
                blocked.update(lineage)
                blocked.add(best_id)

            return selected

    def update_after_expansion(
        self,
        parent_id: str,
        children: List[Tuple[str, float]],
    ) -> None:
        """
        Update archive after expanding a parent state.

        Keeps only the top-K children by reward. Updates Q-values and
        backpropagates visitation counts.

        Args:
            parent_id: The expanded parent state ID.
            children: List of (child_id, child_reward) tuples for all
                      children generated from this parent in this expansion.
        """
        with self._lock:
            if parent_id not in self.states:
                return

            # Sort children by reward (descending) and keep top-K
            children_sorted = sorted(children, key=lambda x: x[1], reverse=True)
            children_to_keep = children_sorted[: self.top_children_per_parent]

            # The children should already be added via add_state; just prune
            kept_ids = {cid for cid, _ in children_to_keep}
            parent_state = self.states[parent_id]

            # Remove non-kept children from this expansion
            new_children_from_expansion = {cid for cid, _ in children}
            to_remove = new_children_from_expansion - kept_ids
            for rid in to_remove:
                if rid in self.states and rid not in self.seed_ids:
                    # Remove from parent's children list
                    if rid in parent_state.children_ids:
                        parent_state.children_ids.remove(rid)
                    del self.states[rid]

            # Update max_child_reward for parent
            best_child_reward = max(
                (self.states[cid].reward for cid in parent_state.children_ids if cid in self.states),
                default=None,
            )
            if best_child_reward is not None:
                if parent_state.max_child_reward is None:
                    parent_state.max_child_reward = best_child_reward
                else:
                    parent_state.max_child_reward = max(parent_state.max_child_reward, best_child_reward)

            # Backpropagate visitation count to parent and ancestors
            self._backpropagate_visits(parent_id)

            self.total_expansions += 1
            self._enforce_archive_limit()

    # ------------------------------------------------------------------
    # PUCT scoring
    # ------------------------------------------------------------------

    def _compute_puct_score(self, state_id: str) -> float:
        """
        Compute PUCT score for a state:

            score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s))
        """
        state = self.states[state_id]

        q = state.q_value()
        n = state.visit_count
        T = self.total_expansions
        c = self.exploration_coef

        # Reward scale
        rewards = [s.reward for s in self.states.values()]
        r_max = max(rewards)
        r_min = min(rewards)
        scale = r_max - r_min if r_max > r_min else 1.0

        # Rank-based prior P(s)
        p = self._rank_prior(state_id)

        exploration_bonus = c * scale * p * math.sqrt(1.0 + T) / (1.0 + n)

        return q + exploration_bonus

    def _rank_prior(self, state_id: str) -> float:
        """
        Compute rank-based linear prior P(s).

        P(s) = (|H| - rank(s)) / sum_{s' in H}(|H| - rank(s'))

        where rank(s) in {0, ..., |H|-1} with rank 0 being the best.
        """
        # Sort states by reward descending
        sorted_ids = sorted(
            self.states.keys(),
            key=lambda sid: self.states[sid].reward,
            reverse=True,
        )
        H = len(sorted_ids)
        rank_map = {sid: r for r, sid in enumerate(sorted_ids)}

        rank_s = rank_map.get(state_id, H - 1)
        weight = H - rank_s
        total_weight = sum(H - r for r in range(H))

        if total_weight == 0:
            return 1.0 / max(H, 1)

        return weight / total_weight

    # ------------------------------------------------------------------
    # Lineage and visitation
    # ------------------------------------------------------------------

    def _get_ancestors(self, state_id: str) -> Set[str]:
        """Get all ancestor state IDs (parents, grandparents, ...)."""
        ancestors = set()
        current = state_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            state = self.states.get(current)
            if state is None or state.parent_id is None:
                break
            ancestors.add(state.parent_id)
            current = state.parent_id
        return ancestors

    def _get_descendants(self, state_id: str) -> Set[str]:
        """Get all descendant state IDs (children, grandchildren, ...)."""
        descendants = set()
        stack = [state_id]
        while stack:
            current = stack.pop()
            state = self.states.get(current)
            if state is None:
                continue
            for child_id in state.children_ids:
                if child_id not in descendants and child_id in self.states:
                    descendants.add(child_id)
                    stack.append(child_id)
        return descendants

    def _get_lineage(self, state_id: str) -> Set[str]:
        """Get full lineage: ancestors + descendants."""
        return self._get_ancestors(state_id) | self._get_descendants(state_id)

    def _backpropagate_visits(self, state_id: str) -> None:
        """Increment visit count for state and all its ancestors."""
        current = state_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            state = self.states.get(current)
            if state is None:
                break
            state.visit_count += 1
            current = state.parent_id

    # ------------------------------------------------------------------
    # Archive maintenance
    # ------------------------------------------------------------------

    def _enforce_archive_limit(self) -> None:
        """Keep only top-K states by reward, always retaining seed states."""
        if len(self.states) <= self.archive_max_size:
            return

        # Sort non-seed states by reward
        non_seed = [
            (sid, self.states[sid].reward)
            for sid in self.states
            if sid not in self.seed_ids
        ]
        non_seed.sort(key=lambda x: x[1], reverse=True)

        # Determine how many non-seed states to keep
        keep_count = self.archive_max_size - len(self.seed_ids)
        to_remove = {sid for sid, _ in non_seed[keep_count:]}

        for sid in to_remove:
            # Clean up parent references
            state = self.states[sid]
            if state.parent_id and state.parent_id in self.states:
                parent = self.states[state.parent_id]
                if sid in parent.children_ids:
                    parent.children_ids.remove(sid)
            del self.states[sid]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_state(self, state_id: str) -> Optional[PUCTState]:
        """Get state by ID."""
        return self.states.get(state_id)

    def get_best_state(self) -> Optional[PUCTState]:
        """Get the state with the highest reward."""
        if not self.states:
            return None
        best_id = max(self.states.keys(), key=lambda sid: self.states[sid].reward)
        return self.states[best_id]

    def size(self) -> int:
        """Current archive size."""
        return len(self.states)

    def get_reward_stats(self) -> Dict[str, float]:
        """Get reward statistics of the archive."""
        if not self.states:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
        rewards = [s.reward for s in self.states.values()]
        return {
            "min": min(rewards),
            "max": max(rewards),
            "mean": sum(rewards) / len(rewards),
            "count": len(rewards),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save archive to disk."""
        os.makedirs(path, exist_ok=True)
        data = {
            "exploration_coef": self.exploration_coef,
            "archive_max_size": self.archive_max_size,
            "top_children_per_parent": self.top_children_per_parent,
            "total_expansions": self.total_expansions,
            "seed_ids": list(self.seed_ids),
            "states": {},
        }
        for sid, state in self.states.items():
            data["states"][sid] = {
                "state_id": state.state_id,
                "reward": state.reward,
                "parent_id": state.parent_id,
                "children_ids": state.children_ids,
                "visit_count": state.visit_count,
                "max_child_reward": state.max_child_reward,
                "is_seed": state.is_seed,
                "language": state.language,
                "metadata": state.metadata,
            }
            # Save code separately (can be large)
            if state.code is not None:
                code_path = os.path.join(path, f"code_{sid}.txt")
                with open(code_path, "w") as f:
                    f.write(state.code)

        meta_path = os.path.join(path, "puct_archive.json")
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved PUCT archive with {len(self.states)} states to {path}")

    def load(self, path: str) -> None:
        """Load archive from disk."""
        meta_path = os.path.join(path, "puct_archive.json")
        if not os.path.exists(meta_path):
            logger.warning(f"No PUCT archive found at {meta_path}")
            return

        with open(meta_path, "r") as f:
            data = json.load(f)

        self.exploration_coef = data.get("exploration_coef", self.exploration_coef)
        self.archive_max_size = data.get("archive_max_size", self.archive_max_size)
        self.top_children_per_parent = data.get("top_children_per_parent", self.top_children_per_parent)
        self.total_expansions = data.get("total_expansions", 0)
        self.seed_ids = set(data.get("seed_ids", []))

        self.states.clear()
        for sid, sdata in data.get("states", {}).items():
            code = None
            code_path = os.path.join(path, f"code_{sid}.txt")
            if os.path.exists(code_path):
                with open(code_path, "r") as f:
                    code = f.read()

            self.states[sid] = PUCTState(
                state_id=sdata["state_id"],
                reward=sdata["reward"],
                parent_id=sdata.get("parent_id"),
                children_ids=sdata.get("children_ids", []),
                visit_count=sdata.get("visit_count", 0),
                max_child_reward=sdata.get("max_child_reward"),
                is_seed=sdata.get("is_seed", False),
                code=code,
                language=sdata.get("language", "python"),
                metadata=sdata.get("metadata", {}),
            )

        logger.info(f"Loaded PUCT archive with {len(self.states)} states from {path}")

    def __repr__(self) -> str:
        stats = self.get_reward_stats()
        return (
            f"PUCTArchive(size={self.size()}, expansions={self.total_expansions}, "
            f"reward_range=[{stats['min']:.4f}, {stats['max']:.4f}])"
        )
