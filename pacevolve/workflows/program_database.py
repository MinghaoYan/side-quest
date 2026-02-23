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

"""A programs database that implements a parallelized tournament selection evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any
import numpy as np
import logging

logger = logging.getLogger("controller")
Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


@dataclasses.dataclass
class ProgramsDatabaseConfig:
  """Configuration for the ProgramsDatabase."""
  num_islands: int = 1
  tournament_size: int = 2
  top_k: int = 4
  max_queue_size: int = 100


@dataclasses.dataclass(frozen=True)
class Candidate:
  """A prompt produced by the ProgramsDatabase, to be sent to Samplers."""
  code: str
  version_generated: int
  island_id: int


class ProgramsDatabase:
  """A collection of programs, organized as islands."""

  def __init__(
      self,
      config: ProgramsDatabaseConfig,
      template: str,
      function_to_evolve: str,
      metric_direction: str,
  ) -> None:
    self._config: ProgramsDatabaseConfig = config
    self._function_to_evolve: str = function_to_evolve

    if metric_direction == "max":
      self._reverse_sort = True
    else:
      self._reverse_sort = False
    self._islands: list[Island] = [
        Island(config, template, function_to_evolve, self._reverse_sort)
        for _ in range(config.num_islands)
    ]

  def get_candidate(self) -> Candidate:
    """Returns a prompt containing a parent from one chosen island.
    
    This method is designed to be called by up to M parallel workers.
    """
    island_id = np.random.randint(len(self._islands))
    code, version_generated = self._islands[island_id].get_candidate()
    return code, island_id

  def register_program(
      self,
      program: str,
      island_id: int,
      score: float,
  ) -> None:
    """Registers `program` in the database.

    This method is designed to be called by up to M parallel workers.
    """
    self._islands[island_id].register_program(program, score)


class Island:
  """A sub-population of programs that evolves using tournament selection."""

  def __init__(
      self,
      config: ProgramsDatabaseConfig,
      template: str,
      function_to_evolve: str,
      reverse_sort: bool
  ) -> None:
    self._config: ProgramsDatabaseConfig = config
    self._template: str = template
    self._function_to_evolve: str = function_to_evolve
    
    # This is our priority queue, storing tuples of (score, program).
    # It's kept sorted by score in descending order.
    self._candidates: list[tuple[float, str]] = []
    self._version_counter: int = 0

    self._reverse_sort = reverse_sort

  def get_candidate(self) -> tuple[str, int]:
    """Selects a parent via tournament selection and returns a prompt."""
    if not self._candidates:
      # If the queue is empty, use the function from the base template as the parent.
      parent = self._template
    else:

      candidates_snapshot = self._candidates
      top_k_candidates = candidates_snapshot[:self._config.top_k]
      tournament_size = min(self._config.tournament_size, len(top_k_candidates))
      
      if tournament_size == 0:
        # Fallback if there are no candidates to select from.
        parent = self._template
      else:
        indices = np.random.choice(
            len(top_k_candidates), size=tournament_size, replace=False)
        tournament = [top_k_candidates[i] for i in indices]

        parent = max(tournament, key=lambda x: x[0])[1]

    version_generated = self._version_counter
    self._version_counter += 1
    
    new_function_name = f'{self._function_to_evolve}_v{version_generated}'
    
    return parent, new_function_name

  def register_program(
      self,
      program: str,
      score: float,
  ) -> None:
    """Stores a program in the priority queue."""

    if score is None:
      logger.error(f"Program score is None, skipping program registration.")
      return
    self._candidates.append((score, program))
    
    # Sort to maintain ascending order of scores.
    self._candidates.sort(key=lambda x: x[0], reverse=self._reverse_sort)
    
    # Trim the queue to maintain its maximum size.
    if len(self._candidates) > self._config.max_queue_size:
      self._candidates = self._candidates[:self._config.max_queue_size]
