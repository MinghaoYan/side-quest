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

import random
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger("controller")

class AdaptiveCrossoverScheduler:
    """
    Implements the adaptive crossover logic in a stateful, efficient manner.
    - Triggers based on: Momentum (rate of progress)
    - Samples based on: Absolute Progress (position on path to target)
    
    Refactored to separate state updates from action sampling.
    """
    def __init__(
        self, 
        num_islands: int,
        target_score: float,
        beta: float = 0.85,
        epsilon_rel: float = 0.001,
        reverse: int = 1,
    ):
        self.reverse = reverse
        self.num_islands = num_islands
        self.target_score = self.reverse * target_score
        self.beta = beta
        self.epsilon_rel = epsilon_rel
        
        # Internal state trackers
        self.momentums = [0.0] * num_islands
        self.initial_scores = [float('inf')] * num_islands
        self.best_scores = [float('inf')] * num_islands    
        self.update_counts = [0] * num_islands
        
        logger.info(f"Scheduler initialized for {num_islands} islands with target_score={target_score}.")

    def _get_all_absolute_progress(self) -> List[float]:
        """
        Calculates the "Absolute Progress" (A_t) for all islands.
        A_t = (s_0 - s_t) / (s_0 - r)
        This represents the fraction of the total gap closed.
        """
        all_progress = []
        for i in range(self.num_islands):
            s_0 = self.initial_scores[i]
            s_t = self.best_scores[i]
            r = self.target_score
            
            denominator = s_0 - r
            numerator = s_0 - s_t
            
            if s_0 == float('inf') or s_t == float('inf'):
                 # No progress if island has never been updated
                 all_progress.append(0.0)
            elif denominator <= 1e-9:
                # If initial score was already at or below target
                all_progress.append(1.0 if s_t <= r else 0.0)
            else:
                progress = numerator / denominator
                all_progress.append(progress)
                
        # Clamp progress between 0 and 1
        return [max(0.0, min(1.0, p)) for p in all_progress]

    def update_score(
        self, 
        island_id: int,
        new_score: float
    ):
        """
        Takes a new score for a *single island* and updates its
        internal momentum and best score trackers.
        """
        if not (0 <= island_id < self.num_islands):
            logger.warning(f"Invalid island_id: {island_id}. Ignoring update.")
            return
            
        relative_progress = 0.0
        
        s_t_candidate = self.reverse * new_score
        s_tm1_best = self.best_scores[island_id]

        # Check for new best score (minimization problem)
        if s_t_candidate < s_tm1_best:
            s_t_best = s_t_candidate
            self.best_scores[island_id] = s_t_best 

            # If this is the very first score for this island, set initial score
            if self.update_counts[island_id] == 0:
                self.initial_scores[island_id] = s_t_best
                logger.info(f"Island {island_id}: Initial score set to {s_t_best}")
            else:
                denominator = s_tm1_best - self.target_score
                if denominator > 1e-9: 
                    # Only calculate progress if the previous best was better than the target
                    r_t = (s_tm1_best - s_t_best) / denominator
                    relative_progress = r_t

        # Update momentum (m_t)
        self.momentums[island_id] = (
            self.beta * self.momentums[island_id] + 
            (1 - self.beta) * relative_progress
        )
        logger.info(f"Island {island_id}: Momentum updated to {self.momentums[island_id]}")
                    
        self.update_counts[island_id] += 1
        
    def check_trigger(self, island_id: int) -> bool:
        """
        Checks if the specified island's momentum is below the trigger
        threshold.
        """
        if not (0 <= island_id < self.num_islands):
            logger.warning(f"Invalid island_id: {island_id}. Cannot check trigger.")
            return False
            
        # Don't trigger on the very first update
        if self.update_counts[island_id] <= 1:
            return False
            
        return self.momentums[island_id] < self.epsilon_rel

    def sample_action(
        self, 
        triggered_island_idx: int
    ) -> Tuple[str, Optional[int]]:
        """
        Implements the Unified Utility Sampling (Scheme 1).
        You call this *after* check_trigger(island_id) returns True.
        
        It uses the scheduler's current internal state to sample an action.
        """
        if not (0 <= triggered_island_idx < self.num_islands):
            logger.warning(f"Invalid island_id: {triggered_island_idx}. Defaulting to BACKTRACK.")
            return ("BACKTRACK", None)
            
        if self.num_islands <= 1:
            return ("BACKTRACK", None)

        # 1. Get the Absolute Progress (A_t) for all islands
        absolute_progresses = self._get_all_absolute_progress()
        
        # Use A_t as the metric for sampling
        m_i = absolute_progresses[triggered_island_idx]

        # 2. Find the best partner
        m_best = -float('inf')
        j_best = -1
        for j in range(self.num_islands):
            if j == triggered_island_idx:
                continue
            if absolute_progresses[j] > m_best:
                m_best = absolute_progresses[j]
                j_best = j

        # No other island exists or all are worse
        if j_best == -1: 
            logger.info(f"Island {triggered_island_idx}: Triggered, but no better islands exist. BACKTRACK.")
            return ("BACKTRACK", None)

        # 3. Calculate all weights based on Absolute Progress (A_t)
        
        # Similarity
        similarity = max(0, 1 - abs(m_i - m_best))
        
        # Principle 2: Dominance (w_bt_dom)
        w_bt_dom = max(0, m_i - m_best)
        # Principle 3: Low-Sim Stagnation (w_bt_stag)
        w_bt_stag = similarity * (1 - m_i) * (1 - m_best)
        w_bt = w_bt_dom + w_bt_stag

        # Crossover weights
        crossover_weights: Dict[int, float] = {}
        for j in range(self.num_islands):
            if j == triggered_island_idx:
                continue
            
            # Principle 1: Gain (w_c_base)
            w_c_base = max(0, absolute_progresses[j] - m_i)
            
            # Principle 4: High-Sim Synergy (w_c_syn)
            w_c_syn = 0.0
            if j == j_best:
                w_c_syn = similarity * m_i * m_best
            
            crossover_weights[j] = w_c_base + w_c_syn

        # 4. Assemble pools and sample
        action_pool = [("BACKTRACK", None)]
        weight_pool = [w_bt]
        
        for j, w in crossover_weights.items():
            action_pool.append(("CROSSOVER", j))
            weight_pool.append(w)

        total_weight = sum(weight_pool)
        if total_weight < 1e-9:
            # If all weights are zero (e.g., all progresses are 0), default to backtrack
            logger.info(f"Island {triggered_island_idx}: Triggered, but all weights are 0. BACKTRACK.")
            return ("BACKTRACK", None)

        # 5. Perform weighted random choice
        chosen_action = random.choices(action_pool, weights=weight_pool, k=1)[0]
        
        logger.info(f"Island {triggered_island_idx}: Triggered. Action weights: {list(zip(action_pool, weight_pool))}. Chosen: {chosen_action}")
        return chosen_action

    # --- Inspection Helpers ---
    def get_momentums(self) -> List[float]:
        """Helper to inspect the scheduler's internal state."""
        return self.momentums

    def get_scheduler_best_scores(self) -> List[float]:
        """Helper to inspect the scheduler's *internal* best scores (s_t)."""
        return self.best_scores

    def get_scheduler_initial_scores(self) -> List[float]:
        """Helper to inspect the scheduler's *internal* initial scores (s_0)."""
        return self.initial_scores
        
    def get_all_absolute_progress(self) -> List[float]:
        """Helper to inspect the scheduler's *internal* A_t calc."""
        return self._get_all_absolute_progress()

