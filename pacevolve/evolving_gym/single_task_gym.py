"""PACEvolve Evolving Gym for SLIME integration.

Exposes the same interface as OpenEvolve's SingleTaskEvolvingGym:
  - problem_generator()  -> (prompt_dict, parent_program_wrapper)
  - response_scorer(response, parent) -> Result

The trained policy handles ONLY idea generation (3 hypotheses) + idea selection.
PACEvolve's API model handles classification, code implementation,
compile/eval retry loops, and summarization -- preserving the full workflow.
"""

import asyncio
import dataclasses
import importlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
import yaml
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from pacevolve.evolving_gym import record_context  # noqa: E402

logger = logging.getLogger(__name__)


IDEA_SELECTION_SUFFIX = """
After generating your ideas, you must select the most promising one to implement.
Use the following format:

Selected Idea: <idea number, e.g. 1, 2, or 3>
Experiment description: <concise description of the experiment to run>
"""


# ---------------------------------------------------------------------------
# Lightweight data wrappers (compatible with SLIME reward model expectations)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ParentProgram:
    """Minimal Program-like wrapper so the reward model can access .id, .metrics, .code."""
    id: str
    code: str
    island_id: int
    metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    generation: int = 0

    @property
    def metadata(self):
        return {"island": self.island_id}


@dataclasses.dataclass
class Result:
    """Result of scoring a response, compatible with SLIME evolving_gym_rm expectations."""
    child_program: Any = None
    parent: Any = None
    child_metrics: Dict[str, Any] = None
    iteration_time: float = None
    prompt: Dict[str, str] = None
    llm_response: str = None
    artifacts: Dict[str, Any] = None
    runtime_environment_path: str = None


@dataclasses.dataclass
class ChildProgram:
    """Minimal child program wrapper."""
    id: str
    code: str
    metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    parent_id: str = None
    generation: int = 0
    island_id: int = 0


# ---------------------------------------------------------------------------
# Serialisable database wrapper for save/load
# ---------------------------------------------------------------------------

class PACEvolveDatabase:
    """Wrapper around ProgramsDatabase that adds save/load and temp-cache
    support, matching the interface SLIME's RolloutDataSource expects."""

    def __init__(self, programs_db, idea_repo_db, config: dict):
        self.programs_db = programs_db
        self.idea_repo_db = idea_repo_db
        self._config = config
        self.programs: Dict[str, Any] = {}
        self.temp_programs: List[Any] = []
        self._temp_lock = threading.Lock()
        self._iteration_counter = 0

    def add_temp(self, child_program, iteration=0):
        with self._temp_lock:
            self.temp_programs.append(child_program)

    def get_temp_cache_size(self):
        with self._temp_lock:
            return len(self.temp_programs)

    def update_database_from_temp(self, verbose: bool = True):
        """SLIME-compatible alias: flush temp cache to main database."""
        n_before = len(self.temp_programs)
        self.flush_temp_to_db()
        if verbose and n_before > 0:
            logger.info(f"[PACEvolve] Flushed {n_before} temp programs to database")

    def flush_temp_to_db(self):
        """Move temp programs into the real ProgramsDatabase."""
        with self._temp_lock:
            for prog in self.temp_programs:
                score = prog.metrics.get("combined_score")
                if score is not None and not isinstance(score, str):
                    self.programs_db.register_program(
                        program=prog.code,
                        island_id=prog.island_id,
                        score=score,
                    )
            self.temp_programs.clear()

    def save(self, path, rollout_id=None):
        os.makedirs(path, exist_ok=True)
        state = {
            "config": self._config,
            "iteration_counter": self._iteration_counter,
            "islands": [],
        }
        for island in self.programs_db._islands:
            state["islands"].append({
                "candidates": list(island._candidates),
                "version_counter": island._version_counter,
            })
        if self.idea_repo_db is not None:
            state["best_scores_history"] = [
                list(h) for h in self.idea_repo_db.best_scores_history
            ]
        with open(os.path.join(path, "db_state.json"), "w") as f:
            json.dump(state, f, default=str)
        logger.info(f"PACEvolve database saved to {path}")

    def load(self, path):
        state_path = os.path.join(path, "db_state.json")
        if not os.path.exists(state_path):
            logger.warning(f"No database state found at {state_path}")
            return
        with open(state_path, "r") as f:
            state = json.load(f)
        self._iteration_counter = state.get("iteration_counter", 0)
        for i, island_state in enumerate(state.get("islands", [])):
            if i < len(self.programs_db._islands):
                island = self.programs_db._islands[i]
                island._candidates = [tuple(c) for c in island_state["candidates"]]
                island._version_counter = island_state["version_counter"]
        if self.idea_repo_db is not None and "best_scores_history" in state:
            for i, hist in enumerate(state["best_scores_history"]):
                if i < len(self.idea_repo_db.best_scores_history):
                    self.idea_repo_db.best_scores_history[i] = hist
        logger.info(f"PACEvolve database loaded from {path}")


# ---------------------------------------------------------------------------
# Main Gym class
# ---------------------------------------------------------------------------

class PACEvolveSingleTaskGym:
    """
    Evolving gym backed by PACEvolve's island-based evolution.

    The trained policy (SLIME model) handles idea generation + selection.
    PACEvolve's API model handles classification, code implementation,
    compile/eval retry loops, and summarization.

    Compatible with SLIME's EvolvingGymManager interface via:
      - problem_generator() -> (prompt_dict, parent_program)
      - response_scorer(response, parent) -> Result
      - database.save() / database.load()
      - initialize_sync()
    """

    def __init__(
        self,
        config_path: str,
        initial_program_path: Optional[str] = None,
        max_concurrent_evaluations: int = 1,
        reward_process_type: str = "original_reward",
        seed: Optional[int] = None,
        log_prompts: bool = True,
        n_samples_per_prompt: int = 1,
        **kwargs,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.config["config_path"] = os.path.abspath(config_path)

        self.task_id = self.config["experiment"]["task_id"]
        self.reward_process_type = reward_process_type
        self._initialized = False
        self.log_prompts = log_prompts
        self._llm_name = self.config["llm"]["name"]
        self._rl_reward_cfg = self._build_rl_reward_config()

        # Resolve paths
        pacevolve_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if pacevolve_root not in sys.path:
            sys.path.insert(0, pacevolve_root)
        workflows_dir = os.path.join(pacevolve_root, "workflows")
        if workflows_dir not in sys.path:
            sys.path.insert(0, workflows_dir)

        # Load initial program
        if initial_program_path:
            with open(initial_program_path, "r") as f:
                self._initial_code = f.read()
        else:
            prompt_filename = self.config["experiment"].get("prompts_file", "prompts")
            self._prompts_module = importlib.import_module(
                f"tasks.{self.task_id}.config.{prompt_filename}"
            )
            algo_name = self.config["experiment"]["sota_algo_name"]
            self._initial_code = getattr(self._prompts_module, algo_name)

        # Import task modules
        self._prompts_module = self._import_prompts()
        self._task_eval_utils = importlib.import_module(
            f"tasks.{self.task_id}.eval.eval_utils"
        )

        # Build PACEvolve databases
        import program_database as pdb_mod
        import idea_select_utils

        num_islands = self.config["database"]["num_islands"]
        metric_dir = self.config["evaluation"]["metric_direction"]

        db_config = pdb_mod.ProgramsDatabaseConfig(
            num_islands=num_islands,
            tournament_size=self.config["database"]["tournament_size"],
            top_k=self.config["database"]["top_k"],
            max_queue_size=self.config["database"]["max_queue_size"],
        )
        self._programs_db = pdb_mod.ProgramsDatabase(
            config=db_config,
            template=self._initial_code,
            function_to_evolve=self.task_id,
            metric_direction=metric_dir,
        )
        self._idea_repo_db = idea_select_utils.IdeaRepoDatabase(
            num_islands=num_islands,
            target_score=self.config["evaluation"]["target_score"],
            metric_direction=metric_dir,
        )

        # Compile config
        import task_utils
        src_path = os.path.expanduser(self.config["paths"]["src_path"])
        self._compile_config = task_utils.CompilationConfig(
            target_file_path=os.path.join(
                src_path, self.config["paths"]["target_file_path"]
            ),
            pip_path=None,
        )

        # Eval configs
        EvalConfig = self._task_eval_utils.EvalConfig
        self._eval_configs = [
            EvalConfig(**d)
            for d in self.config["evaluation"]["eval_configs"]
        ]

        # Concurrency -- serialize evals to avoid file/GPU conflicts.
        # If a dedicated eval GPU pool is provided, cap concurrency to pool size.
        eval_gpu_ids_env = os.environ.get("PACEVOLVE_EVAL_GPU_IDS", "")
        self._eval_gpu_ids = [
            gpu_id.strip()
            for gpu_id in eval_gpu_ids_env.split(",")
            if gpu_id.strip()
        ]
        if self._eval_gpu_ids:
            max_concurrent_evaluations = min(
                max_concurrent_evaluations, len(self._eval_gpu_ids)
            )
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self._eval_semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
        self._eval_gpu_queue = asyncio.Queue()
        for gpu_id in self._eval_gpu_ids:
            self._eval_gpu_queue.put_nowait(gpu_id)
        self._generation_counter = 0
        self._lock = threading.Lock()
        self._pending_context: Dict[str, Dict[str, Any]] = {}  # one context per parent
        self._n_samples_per_prompt = n_samples_per_prompt
        self._ablation_list: List[str] = []
        self.num_islands = num_islands

        # PACEvolve workflow parameters (mirror run_experiment.py args)
        self._use_idea_repo = kwargs.get("use_idea_repo", True)
        self._use_integrated_sampling = kwargs.get("use_integrated_sampling", True)
        self._freeze_period = kwargs.get("freeze_period", 15)
        self._max_attempt = kwargs.get("max_attempt", 3)
        self._backtrack_freq = kwargs.get("backtrack_freq", -1)
        self._backtrack_len = kwargs.get("backtrack_len", 5)
        self._power_alpha = kwargs.get("power_alpha", 1.5)
        self._idea_cap = kwargs.get("idea_cap", 5)
        self._merge_freq = kwargs.get("merge_freq", 1)
        self._summarize_freq = kwargs.get("summarize_freq", 20)

        self._per_island_count = [0] * num_islands
        self._last_crossover_idx = [0] * num_islands

        # Backtrack state (persists across iterations, per-island)
        self._backtrack_triggered_idx: Dict[int, int] = {}
        self._repo_idx_before_backtrack: Dict[int, int] = {}

        # Transcript log directory for API model calls
        log_dir = os.path.expanduser(self.config["paths"].get("log_dir", "/tmp/pacevolve_gym_logs"))
        os.makedirs(log_dir, exist_ok=True)
        self._transcript_file = os.path.join(log_dir, "gym_transcript.txt")

        # Wrap database for SLIME compatibility
        self.database = PACEvolveDatabase(
            self._programs_db, self._idea_repo_db, self.config
        )

        # Recorder (optional, SLIME compatibility)
        self._recorder = None
        self.recording_enabled = False

        # PUCT stubs for SLIME compat
        self.use_puct_reuse = False
        self.puct_archive = None

        print(
            f"[PACEvolveSingleTaskGym] Initialized for task={self.task_id}, "
            f"islands={num_islands}, reward_type={reward_process_type}, "
            f"max_concurrent_evaluations={self.max_concurrent_evaluations}, "
            f"eval_gpu_ids={self._eval_gpu_ids}",
            flush=True,
        )

    def _get_transcript_filename(self) -> str:
        """Return per-sample transcript file when recording, else shared transcript."""
        ctx = record_context.get_record_context()
        if ctx and self.recording_enabled:
            step_dir = os.path.join(
                ctx["record_dir"], f"step_{ctx['rollout_id']:05d}"
            )
            os.makedirs(step_dir, exist_ok=True)
            return os.path.join(step_dir, f"api_{ctx['sample_index']:04d}.txt")
        return self._transcript_file

    def set_record_context(self, rollout_id: int, sample_index: int) -> None:
        """Set record context for current sample (used by pacevolve_gym_rm)."""
        if self.recording_enabled and hasattr(self, "_record_dir"):
            record_context.set_record_context(
                rollout_id, sample_index, self._record_dir
            )

    def clear_record_context(self) -> None:
        """Clear record context after rollout."""
        record_context.clear_record_context()

    def _create_sample_workspace(self, eval_gpu_id: Optional[str] = None):
        """Create an isolated temp workspace for this sample to avoid race conditions
        when multiple samples edit the same target file concurrently.
        Returns (config_override, compile_config_override, cleanup_fn).
        """
        import task_utils

        src_path = os.path.expanduser(self.config["paths"]["src_path"])
        results_path = os.path.expanduser(self.config["paths"].get("results_path", "/tmp"))
        target_file = self.config["paths"]["target_file_path"]

        temp_root = tempfile.mkdtemp(prefix="pacevolve_sample_")
        temp_src = os.path.join(temp_root, "src")
        temp_results = os.path.join(temp_root, "results")

        try:
            shutil.copytree(src_path, temp_src)
        except Exception as e:
            logger.error("Failed to copy src to temp workspace: %s", e)
            try:
                shutil.rmtree(temp_root, ignore_errors=True)
            except Exception:
                pass
            raise

        os.makedirs(temp_results, exist_ok=True)

        config_override = deepcopy(self.config)
        config_override["paths"] = dict(config_override["paths"])
        config_override["evaluation"] = dict(config_override["evaluation"])
        config_override["paths"]["src_path"] = temp_src
        config_override["paths"]["results_path"] = temp_results
        if eval_gpu_id is not None:
            config_override["evaluation"]["cuda_visible_devices"] = str(eval_gpu_id)

        compile_config_override = task_utils.CompilationConfig(
            target_file_path=os.path.join(temp_src, target_file),
            pip_path=None,
        )

        def cleanup():
            try:
                shutil.rmtree(temp_root, ignore_errors=True)
            except Exception as e:
                logger.warning("Failed to cleanup temp workspace %s: %s", temp_root, e)

        return config_override, compile_config_override, cleanup

    def _import_prompts(self):
        prompt_filename = self.config["experiment"].get("prompts_file", "prompts")
        dataset_id = self.config.get("_dataset_id", ".")
        if dataset_id == ".":
            return importlib.import_module(
                f"tasks.{self.task_id}.config.{prompt_filename}"
            )
        else:
            return importlib.import_module(
                f"tasks.{self.task_id}.config.{dataset_id}.{prompt_filename}"
            )

    def _build_rl_reward_config(self) -> Dict[str, float]:
        """Build score->reward transform config for rl_normalized_reward."""
        eval_cfg = self.config.get("evaluation", {})
        score_cfg = self.config.get("score_transform", {})

        init_score = float(eval_cfg.get("init_score", 0.0))
        target_score = float(eval_cfg.get("target_score", 1.0))
        auto_min = min(init_score, target_score)
        auto_max = max(init_score, target_score)

        score_min = float(score_cfg.get("score_range_min", auto_min))
        score_max = float(score_cfg.get("score_range_max", auto_max))
        if score_max <= score_min:
            score_max = score_min + 1.0

        alpha = float(score_cfg.get("alpha", 1.0))
        if alpha <= 0:
            alpha = 1.0

        positive_multiplier = float(score_cfg.get("positive_multiplier", 5.0))
        if positive_multiplier <= 0:
            positive_multiplier = 1.0

        metric_direction = str(eval_cfg.get("metric_direction", "max")).lower()
        if metric_direction not in ("max", "min"):
            metric_direction = "max"

        return {
            "score_min": score_min,
            "score_max": score_max,
            "alpha": alpha,
            "positive_multiplier": positive_multiplier,
            "metric_direction": metric_direction,
        }

    def _compute_rl_normalized_reward(self, combined_score: float, is_error: bool = False) -> float:
        """Convert combined_score to rl_normalized_reward with deterministic scaling."""
        if is_error:
            return float(combined_score)
        if math.isnan(combined_score) or math.isinf(combined_score):
            return -1.0

        score_min = self._rl_reward_cfg["score_min"]
        score_max = self._rl_reward_cfg["score_max"]
        alpha = self._rl_reward_cfg["alpha"]
        positive_multiplier = self._rl_reward_cfg["positive_multiplier"]
        direction = self._rl_reward_cfg["metric_direction"]

        clamped = max(score_min, min(score_max, combined_score))
        if direction == "min":
            linear = (score_max - clamped) / (score_max - score_min)
        else:
            linear = (clamped - score_min) / (score_max - score_min)

        linear = max(0.0, min(1.0, linear))
        return float((linear**alpha) * positive_multiplier)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_sync(self):
        """Run the initial program through eval and seed the database."""
        if self._initialized:
            return
        import idea_select_utils
        IdeaRepo = idea_select_utils.IdeaRepo

        init_score = self.config["evaluation"]["init_score"]
        num_islands = self.config["database"]["num_islands"]
        for island_id in range(num_islands):
            self._programs_db.register_program(
                program=self._initial_code,
                island_id=island_id,
                score=init_score,
            )
            self._idea_repo_db.best_scores_history[island_id].append(init_score)
            self._idea_repo_db.scheduler.update_score(island_id, init_score)
            initial_repo = IdeaRepo()
            initial_repo.sota = self._initial_code
            self._idea_repo_db.idea_repos[island_id].append(initial_repo)

        self._initialized = True
        print(f"[PACEvolveSingleTaskGym] Initialized with init_score={init_score}")

    async def initialize(self):
        self.initialize_sync()

    # ------------------------------------------------------------------
    # Parsing helpers for trained policy output
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_selected_hypothesis(response: str) -> Tuple[Optional[int], Optional[str]]:
        """Parse 'Selected Idea: N' and 'Experiment description: ...' from trained policy output."""
        match = re.search(
            r"Selected Idea:\s*(\d+)\s*\n\s*Experiment description:\s*(.*)",
            response, re.DOTALL,
        )
        if match:
            return int(match.group(1)), match.group(2).strip()
        return None, None

    @staticmethod
    def _extract_json_object(text: str) -> Optional[dict]:
        """Extract the first JSON object from model text output."""
        if not text:
            return None
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    @staticmethod
    def _emit_debug(msg: str):
        """Emit diagnostics to both logger and stdout for Ray job logs."""
        logger.warning(msg)
        print(msg, flush=True)

    @staticmethod
    def _log_policy_ideas(stage: str, hypotheses: List[str], selected_num: Optional[int], exp_description: Optional[str]):
        """Log all policy-generated ideas for sanity checking at each stage."""
        if not hypotheses:
            PACEvolveSingleTaskGym._emit_debug(f"[policy-ideas][{stage}] no hypotheses")
            return
        lines = [f"[policy-ideas][{stage}] total={len(hypotheses)} selected={selected_num}"]
        for i, hypo in enumerate(hypotheses, start=1):
            marker = " <SELECTED>" if selected_num == i else ""
            lines.append(f"  {i}. {hypo}{marker}")
        if exp_description is not None:
            lines.append(f"  experiment_description={exp_description}")
        PACEvolveSingleTaskGym._emit_debug("\n".join(lines))

    def _normalize_policy_output_via_api(
        self,
        raw_policy_response: str,
        llm_utils,
        Transcript,
        ContentChunk,
    ) -> Tuple[List[str], Optional[int], Optional[str]]:
        """Use API model to normalize free-form policy output into strict JSON fields."""
        normalize_prompt = f"""
You are a strict information extractor.
Given a free-form policy output, extract exactly these fields and return JSON only:
{{
  "hypotheses": ["idea 1", "idea 2", "idea 3"],
  "selected_idea": 1,
  "experiment_description": "..."
}}

Rules:
- "hypotheses" must contain exactly 3 concise strings. If there are more, keep the best 3.
- "selected_idea" must be 1, 2, or 3 and must correspond to one hypothesis.
- "experiment_description" must be a concise plain-text experiment plan.
- Output valid JSON only. No markdown, no explanations.

Policy output:
```text
{raw_policy_response}
```
"""
        tpath = self._get_transcript_filename()
        transcript = Transcript(
            log_filename=tpath,
            readable_log=(tpath != self._transcript_file),
        )
        transcript.append(ContentChunk(normalize_prompt, "user", tags=["policy_output_normalization_prompt"]))
        normalized_text = llm_utils.generate_completion(self._llm_name, transcript, self.config)
        transcript.append(ContentChunk(str(normalized_text), "model", tags=["policy_output_normalization_response"]))

        def _parse_payload(payload: dict) -> Tuple[List[str], Optional[int], Optional[str]]:
            hypotheses = payload.get("hypotheses")
            selected_num = payload.get("selected_idea")
            exp_description = payload.get("experiment_description")

            if not isinstance(hypotheses, list):
                hypotheses = []
            hypotheses = [str(x).strip() for x in hypotheses if str(x).strip()]
            if len(hypotheses) > 3:
                hypotheses = hypotheses[:3]

            try:
                selected_num = int(selected_num)
            except Exception:
                selected_num = None
            if selected_num not in (1, 2, 3):
                selected_num = None

            if exp_description is None:
                exp_description = ""
            exp_description = str(exp_description).strip()
            if not exp_description:
                exp_description = None
            return hypotheses, selected_num, exp_description

        data = self._extract_json_object(normalized_text or "")
        if not isinstance(data, dict):
            self._emit_debug(
                f"[policy-normalization] failed to parse JSON from normalization output; "
                f"response_snippet={(normalized_text or '')[:600]!r}"
            )
            return [], None, None

        hypotheses, selected_num, exp_description = _parse_payload(data)
        self._emit_debug(f"[policy-normalization] detected {len(hypotheses)}/3 ideas from policy output")

        # If fewer than 3 ideas are detected, ask API model to generate/complete to exactly 3.
        if len(hypotheses) < 3:
            completion_prompt = f"""
You are repairing/augmenting policy ideas into a strict output schema.

You were given a policy output and extracted only {len(hypotheses)} ideas out of 3.
Please produce exactly 3 concrete hypotheses by preserving extracted ideas and generating missing ones.

Return JSON only:
{{
  "hypotheses": ["idea 1", "idea 2", "idea 3"],
  "selected_idea": 1,
  "experiment_description": "..."
}}

Constraints:
- hypotheses must be exactly 3 concise and distinct ideas.
- Keep existing extracted ideas whenever possible:
{json.dumps(hypotheses)}
- selected_idea must be 1..3 and correspond to one of the final hypotheses.
- If current selected idea is unavailable, choose the closest one.
- experiment_description should align with selected_idea.
- Output valid JSON only.

Original policy output:
```text
{raw_policy_response}
```
"""
            transcript.append(ContentChunk(completion_prompt, "user", tags=["policy_output_completion_prompt"]))
            completion_text = llm_utils.generate_completion(self._llm_name, transcript, self.config)
            transcript.append(ContentChunk(str(completion_text), "model", tags=["policy_output_completion_response"]))

            completion_data = self._extract_json_object(completion_text or "")
            if isinstance(completion_data, dict):
                hypotheses, selected_num, exp_description = _parse_payload(completion_data)
                self._emit_debug(f"[policy-normalization] completed to {len(hypotheses)}/3 ideas")
            else:
                self._emit_debug(
                    f"[policy-normalization] completion step returned non-JSON output; "
                    f"response_snippet={(completion_text or '')[:600]!r}"
                )

        if len(hypotheses) < 3:
            self._emit_debug(
                f"[policy-normalization] final idea count still <3; count={len(hypotheses)} "
                f"selected={selected_num} exp_desc_present={bool(exp_description)}"
            )
            return [], None, None

        return hypotheses, selected_num, exp_description

    # ------------------------------------------------------------------
    # problem_generator  (SLIME interface)
    # ------------------------------------------------------------------

    def _problem_generator_impl(
        self, forced_island_id: Optional[int] = None
    ) -> Tuple[Dict[str, str], ParentProgram]:
        """Build the idea-generation + selection prompt for the trained policy.

        The trained policy will generate 3 hypotheses and select one.
        No API model calls are made here -- all heavy lifting moves to
        response_scorer.
        """
        if not self._initialized:
            self.initialize_sync()

        import idea_select_utils

        with self._lock:
            self._generation_counter += 1
            gen = self._generation_counter

            trigger_backtrack = False
            last_bt_iter = False

            if forced_island_id is None:
                parent_code, island_id = self._programs_db.get_candidate()
            else:
                island_id = forced_island_id
                parent_code, _ = self._programs_db.get_candidate_for_island(island_id)

            # Check ongoing backtrack for this island
            bt_idx = self._backtrack_triggered_idx.get(island_id, -1)
            if bt_idx > -1 and self._per_island_count[island_id] - bt_idx < self._backtrack_len:
                if self._per_island_count[island_id] - bt_idx == self._backtrack_len - 1:
                    last_bt_iter = True
                parent_code = self._idea_repo_db.idea_repos[island_id][-1].sota
                new_idea_repo = deepcopy(self._idea_repo_db.idea_repos[island_id][-1])
            else:
                # Clear stale backtrack state
                if bt_idx > -1:
                    self._backtrack_triggered_idx[island_id] = -1

                if len(self._idea_repo_db.idea_repos[island_id]) == 0:
                    new_idea_repo = idea_select_utils.IdeaRepo()
                else:
                    new_idea_repo = deepcopy(self._idea_repo_db.idea_repos[island_id][-1])

                # Integrated crossover/backtrack trigger
                if (
                    self._use_integrated_sampling
                    and self._per_island_count[island_id] - self._last_crossover_idx[island_id] > self._freeze_period
                    and self._idea_repo_db.scheduler.check_trigger(island_id)
                ):
                    action, cross_id = self._idea_repo_db.scheduler.sample_action(island_id)
                    if action == "CROSSOVER" and cross_id is not None:
                        self._last_crossover_idx[island_id] = self._per_island_count[island_id]
                        _, cross_sota = self._programs_db.get_best_program_for_island(cross_id)
                        if cross_sota is not None:
                            parent_code = cross_sota
                        try:
                            best_idea_repo = self._idea_repo_db.get_best_idea_repo(cross_id)
                            new_idea_repo.ideas.extend(best_idea_repo.ideas)
                            new_idea_repo.reindex_ideas()
                        except Exception:
                            pass
                    elif action == "BACKTRACK":
                        self._last_crossover_idx[island_id] = self._per_island_count[island_id]
                        trigger_backtrack = True

                # Periodic or triggered backtrack
                count = self._per_island_count[island_id]
                if (self._backtrack_freq != -1 and (count + 1) % self._backtrack_freq == 0) or trigger_backtrack:
                    if len(self._idea_repo_db.idea_repos[island_id]) > 0:
                        self._backtrack_triggered_idx[island_id] = count
                        self._repo_idx_before_backtrack[island_id] = len(self._idea_repo_db.idea_repos[island_id]) - 1
                        sampled_idx = idea_select_utils.sample_power_law(
                            len(self._idea_repo_db.idea_repos[island_id]),
                            alpha=self._power_alpha,
                        )
                        parent_code = self._idea_repo_db.idea_repos[island_id][sampled_idx].sota
                        new_idea_repo = deepcopy(self._idea_repo_db.idea_repos[island_id][sampled_idx])

            new_idea_repo.sota = parent_code

            self._per_island_count[island_id] += 1

            # Merge/idea cap check
            trigger_merge = False
            if self._idea_cap > -1 and len(new_idea_repo.ideas) > self._idea_cap and self._merge_freq > -1:
                if self._use_integrated_sampling and self._per_island_count[island_id] - self._last_crossover_idx[island_id] == self._freeze_period:
                    trigger_merge = True
                elif self._backtrack_freq == -1 or (self._per_island_count[island_id]) % self._backtrack_freq == self._backtrack_freq - 1:
                    trigger_merge = True

        # Build the prompt for the trained policy
        prompts = self._prompts_module
        if self._use_idea_repo:
            user_text = prompts.construct_idea_gen_prompt(parent_code, new_idea_repo)
            user_text += IDEA_SELECTION_SUFFIX
        else:
            user_text = prompts.construct_mutation_prompt(parent_code, self._ablation_list)

        parent_id = str(uuid.uuid4())
        best_metric = self.config["evaluation"].get("init_score", 0)
        try:
            best_score, _ = self._programs_db.get_best_program_for_island(island_id)
            if best_score is not None:
                best_metric = best_score
        except Exception:
            pass

        parent = ParentProgram(
            id=parent_id,
            code=parent_code,
            island_id=island_id,
            metrics={"combined_score": best_metric},
            generation=gen,
        )

        context = {
            "island_id": island_id,
            "idea_repo": deepcopy(new_idea_repo),
            "trigger_merge": trigger_merge,
            "last_bt_iter": last_bt_iter,
            "repo_idx_before_backtrack": self._repo_idx_before_backtrack.get(island_id, 0),
        }
        with self._lock:
            self._pending_context[parent_id] = context

        prompt_dict = {
            "system": "",
            "user": user_text,
        }
        return prompt_dict, parent

    def problem_generator(self) -> Tuple[Dict[str, str], ParentProgram]:
        return self._problem_generator_impl(forced_island_id=None)

    def problem_generator_for_island(
        self, island_id: int
    ) -> Tuple[Dict[str, str], ParentProgram]:
        if island_id < 0 or island_id >= self.num_islands:
            raise ValueError(
                f"Invalid island_id={island_id}, expected [0, {self.num_islands - 1}]"
            )
        return self._problem_generator_impl(forced_island_id=island_id)

    # ------------------------------------------------------------------
    # response_scorer  (SLIME interface)
    # ------------------------------------------------------------------

    async def response_scorer(
        self, response: str, parent_program: ParentProgram
    ) -> Optional[Result]:
        """Score a trained-policy response by running the full PACEvolve
        workflow: classify hypotheses, implement via API model, compile/eval
        retry loops, summarize, and update the evolutionary state."""
        # Capture record context before run_in_executor - contextvars don't propagate to threads
        ctx = record_context.get_record_context()

        async with self._eval_semaphore:
            eval_gpu_id = None
            if self._eval_gpu_ids:
                eval_gpu_id = await self._eval_gpu_queue.get()

            def _run_with_context():
                if ctx and "sample_index" in ctx and "record_dir" in ctx:
                    record_context.set_record_context(
                        ctx["rollout_id"],
                        ctx["sample_index"],
                        ctx["record_dir"],
                    )
                return self._score_response_sync(
                    response, parent_program, eval_gpu_id=eval_gpu_id
                )

            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, _run_with_context
                )
            finally:
                if eval_gpu_id is not None:
                    self._eval_gpu_queue.put_nowait(eval_gpu_id)

    def _make_error_result(
        self, parent: ParentProgram, error: str, code: str = "", t0: float = 0.0,
    ) -> Result:
        combined_score = -1.0
        child_metrics = {
            "combined_score": combined_score,
            "rl_normalized_reward": self._compute_rl_normalized_reward(combined_score, is_error=True),
            "error": error,
        }
        child = ChildProgram(
            id=str(uuid.uuid4()),
            code=code,
            metrics=child_metrics,
            parent_id=parent.id,
            island_id=parent.island_id,
        )
        result = Result()
        result.parent = parent
        result.child_program = child
        result.child_metrics = child_metrics
        result.iteration_time = time.time() - t0
        return result

    def _score_response_sync(
        self,
        response: str,
        parent_program: ParentProgram,
        eval_gpu_id: Optional[str] = None,
    ) -> Optional[Result]:
        """Full PACEvolve iteration driven by the trained policy's ideas.

        Mirrors run_experiment.py lines 360-502:
        1. Parse hypotheses + selection from trained policy
        2. Classify hypotheses via API model (scratch_pad phase 2)
        3. Build implementation prompt with selected idea
        4. API model generates code
        5. edit_until_compile  (API model retry loop)
        6. edit_until_successful_eval  (API model retry loop)
        7. Summarisation  (API model)
        8. Update idea repo, DB, scheduler, ablation list
        """
        t0 = time.time()

        try:
            import llm_utils
            import idea_select_utils
            import workflow_utils

            Transcript = llm_utils.Transcript
            ContentChunk = llm_utils.ContentChunk
            AlgorithmTrial = workflow_utils.AlgorithmTrial

            # Retrieve pending context stored by problem_generator (one context per parent)
            with self._lock:
                pending = self._pending_context.pop(parent_program.id, None)
            if pending is None:
                logger.error("No pending context for parent_program %s", parent_program.id)
                return self._make_error_result(parent_program, "no_pending_context", t0=t0)

            island_id = pending["island_id"]
            new_idea_repo = pending["idea_repo"]
            trigger_merge = pending["trigger_merge"]
            last_bt_iter = pending["last_bt_iter"]
            repo_idx_before_backtrack = pending["repo_idx_before_backtrack"]
            parent_code = parent_program.code
            prompts = self._prompts_module

            # ----- Step 1: Normalize and parse trained policy output -----
            hypotheses, selected_num, exp_description = self._normalize_policy_output_via_api(
                response, llm_utils, Transcript, ContentChunk
            )
            if not hypotheses:
                # Fallback to legacy parser if API normalization fails
                hypotheses = idea_select_utils.parse_hypothesis(response)
                selected_num, exp_description = self._parse_selected_hypothesis(response)
            self._log_policy_ideas("parsed", hypotheses, selected_num, exp_description)

            if not hypotheses:
                logger.warning("Trained policy produced no parseable hypotheses.")
                self._emit_debug(
                    f"[policy-normalization] Raw policy response snippet (no_hypotheses): {(response or '')[:1200]!r}"
                )
                self._finalize_idea_repo(island_id, new_idea_repo, last_bt_iter,
                                         repo_idx_before_backtrack, trigger_merge, append=False)
                return self._make_error_result(parent_program, "no_hypotheses", t0=t0)

            if selected_num is None or exp_description is None:
                logger.warning("Trained policy did not produce a valid idea selection.")
                self._emit_debug(
                    f"[policy-selection] invalid selection: selected_num={selected_num} "
                    f"exp_description_present={bool(exp_description)} "
                    f"response_snippet={(response or '')[:1200]!r}"
                )
                self._finalize_idea_repo(island_id, new_idea_repo, last_bt_iter,
                                         repo_idx_before_backtrack, trigger_merge, append=False)
                return self._make_error_result(parent_program, "no_selection", t0=t0)

            if selected_num < 1 or selected_num > len(hypotheses):
                logger.warning("Selected idea number %d out of range [1, %d].",
                               selected_num, len(hypotheses))
                selected_num = min(max(selected_num, 1), len(hypotheses))

            selected_hypothesis = hypotheses[selected_num - 1]
            self._log_policy_ideas("pre_classification", hypotheses, selected_num, exp_description)

            # ----- Step 2: Classify hypotheses via API model -----
            hypo_to_idea_id: Dict[int, Optional[int]] = {}
            for idx, hypo in enumerate(hypotheses):
                tpath = self._get_transcript_filename()
                classification_transcript = Transcript(
                    log_filename=tpath,
                    readable_log=(tpath != self._transcript_file),
                )
                classify_prompt = idea_select_utils.construct_idea_classification_prompt(
                    new_idea_repo, hypo,
                )
                classification_transcript.append(
                    ContentChunk(classify_prompt, "user", tags=["idea_classification_prompt"])
                )
                try:
                    next_id_before = new_idea_repo.get_next_id()
                    classify_response = llm_utils.generate_completion(
                        self._llm_name, classification_transcript, self.config,
                    )
                    if idea_select_utils.update_idea_repo(classify_response, new_idea_repo):
                        # Determine which idea_id this hypothesis mapped to
                        if new_idea_repo.get_next_id() > next_id_before:
                            hypo_to_idea_id[idx] = next_id_before
                        else:
                            id_match = re.search(r"Idea ID:\s*(\d+)", classify_response)
                            hypo_to_idea_id[idx] = int(id_match.group(1)) if id_match else None
                    else:
                        hypo_to_idea_id[idx] = None
                except Exception as e:
                    logger.error("Classification failed for hypothesis %d: %s", idx, e)
                    hypo_to_idea_id[idx] = None

            # Resolve the idea_id for the selected hypothesis
            idea_id = hypo_to_idea_id.get(selected_num - 1)
            if idea_id is None:
                logger.warning("Could not resolve idea_id for selected hypothesis %d. "
                               "Falling back to newest idea.", selected_num)
                if new_idea_repo.ideas:
                    idea_id = new_idea_repo.ideas[-1].id

            # ----- Step 3: Build implementation prompt (API model) -----
            self._log_policy_ideas("pre_implementation", hypotheses, selected_num, exp_description)
            impl_prompt = prompts.construct_code_impl_prompt(
                parent_code, idea_id, exp_description,
                selected_idea_text=selected_hypothesis,
            )
            tpath = self._get_transcript_filename()
            transcript = Transcript(
                log_filename=tpath,
                readable_log=(tpath != self._transcript_file),
            )
            transcript.append(ContentChunk(impl_prompt, "user", tags=["idea_selection_prompt"]))
            llm_impl_response = llm_utils.generate_completion(
                self._llm_name, transcript, self.config,
            )
            transcript.append(ContentChunk(llm_impl_response, "model", tags=["initial_response"]))

            # ----- Step 4 & 5: edit_until_compile + edit_until_successful_eval -----
            # Use per-sample isolated workspace to avoid race when n_samples_per_prompt > 1
            config_override, compile_config_override, cleanup_workspace = (
                self._create_sample_workspace(eval_gpu_id=eval_gpu_id)
            )
            print(
                f"[PACEvolve] Using temp workspace target_file={compile_config_override.target_file_path} "
                f"eval_gpu_id={eval_gpu_id}",
                flush=True,
            )
            try:
                self._log_policy_ideas("pre_compile", hypotheses, selected_num, exp_description)
                trial = AlgorithmTrial()
                trial.idea_id = idea_id if idea_id is not None else -1
                trial = workflow_utils.edit_until_compile(
                    self._llm_name, trial, transcript,
                    compile_config_override, config_override,
                    loop_config=self.config["workflow_loops"]["initial_compile"],
                    use_idea_repo=False,
                )
                transcript.hide_by_tag(tags=["initial_compile_loop"])

                if not trial.compile_success:
                    logger.warning("Compilation failed after retries.")
                    self._finalize_idea_repo(island_id, new_idea_repo, last_bt_iter,
                                             repo_idx_before_backtrack, trigger_merge, append=False)
                    return self._make_error_result(parent_program, "compile_error",
                                                  code=trial.algorithm_implementation, t0=t0)

                # ----- Step 5: edit_until_successful_eval -----
                self._log_policy_ideas("pre_eval", hypotheses, selected_num, exp_description)
                baseline_id = self.config["experiment"]["initial_baseline_id"]
                candidate_id = self._generation_counter

                trial = workflow_utils.edit_until_successful_eval(
                    self._llm_name, trial, transcript, compile_config_override,
                    self._eval_configs, config_override,
                    candidate_id, baseline_id,
                    loop_config=self.config["workflow_loops"]["initial_eval"],
                )
            finally:
                cleanup_workspace()
            transcript.hide_by_tag(tags=["initial_eval_loop"])

            if not all(trial.eval_success):
                logger.warning("Evaluation failed after retries.")
                self._finalize_idea_repo(island_id, new_idea_repo, last_bt_iter,
                                         repo_idx_before_backtrack, trigger_merge, append=False)
                return self._make_error_result(parent_program, "eval_error",
                                              code=trial.algorithm_implementation, t0=t0)

            # ----- Step 6: Log eval results + parse score -----
            transcript.append(ContentChunk(
                prompts.EVAL_DESCRIPTION_PROMPT, "user", tags=["initial_eval_results"],
            ))
            eval_results_text = "\n".join(["```"] + trial.eval_results + ["```"])
            transcript.append(ContentChunk(
                eval_results_text, "system", tags=["initial_eval_results"],
            ))

            eval_metrics = None
            if hasattr(self._task_eval_utils, "parse_eval_metrics"):
                try:
                    eval_metrics = self._task_eval_utils.parse_eval_metrics(trial.eval_results)
                except Exception as e:
                    logger.warning("Failed to parse eval metrics payload: %s", e)

            eval_score = self._task_eval_utils.parse_eval_results(trial.eval_results)
            parse_failed = eval_score is None
            if eval_score is None:
                eval_score = -1.0

            # ----- Step 7: Summarisation (API model) -----
            summary_prompt = prompts.SUMMARIZE_EVAL_PROMPT
            transcript.append(ContentChunk(
                summary_prompt, "user", tags=["final_summary_request"],
            ))
            bullets = []
            for attempt_idx in range(self._max_attempt):
                llm_summary = llm_utils.generate_completion(
                    self._llm_name, transcript, self.config,
                )
                if llm_summary:
                    transcript.append(ContentChunk(
                        llm_summary, "model", tags=["final_summary_response"],
                    ))
                    bullets = workflow_utils.extract_summary(llm_summary)
                    break
                else:
                    logger.warning("Summarisation attempt %d failed.", attempt_idx + 1)

            self._ablation_list.extend(bullets)

            # ----- Step 8: Update idea repo -----
            if idea_id is not None:
                idea_obj = new_idea_repo.find_idea_by_id(idea_id)
                if idea_obj is not None:
                    idea_obj.exp_history.extend(bullets)
                    idea_obj.exp_count += 1
                    if idea_obj.exp_count % self._summarize_freq == 0:
                        try:
                            tpath = self._get_transcript_filename()
                            idea_select_utils.summarize(
                                idea_obj, self._llm_name, self.config,
                                Transcript(
                                    log_filename=tpath,
                                    readable_log=(tpath != self._transcript_file),
                                ),
                                Transcript(
                                    log_filename=tpath,
                                    readable_log=(tpath != self._transcript_file),
                                ),
                            )
                        except Exception as e:
                            logger.error("Idea summarisation failed: %s", e)

            self._finalize_idea_repo(island_id, new_idea_repo, last_bt_iter,
                                     repo_idx_before_backtrack, trigger_merge)

            # ----- Step 9: Register program in DB -----
            combined_score = float(eval_score)
            child_metrics = {
                "combined_score": combined_score,
                "rl_normalized_reward": self._compute_rl_normalized_reward(
                    combined_score, is_error=parse_failed
                ),
            }
            if isinstance(eval_metrics, dict):
                for key, value in eval_metrics.items():
                    if key not in child_metrics:
                        child_metrics[key] = value
            child = ChildProgram(
                id=str(uuid.uuid4()),
                code=trial.algorithm_implementation,
                metrics=child_metrics,
                parent_id=parent_program.id,
                generation=parent_program.generation + 1,
                island_id=island_id,
            )

            with self._lock:
                if eval_score is not None and eval_score != -1.0:
                    self._programs_db.register_program(
                        program=trial.algorithm_implementation,
                        island_id=island_id,
                        score=float(eval_score),
                    )
                    self._idea_repo_db.best_scores_history[island_id].append(float(eval_score))
                    self._idea_repo_db.scheduler.update_score(island_id, float(eval_score))

            result = Result()
            result.parent = parent_program
            result.child_program = child
            result.child_metrics = child_metrics
            result.llm_response = response
            result.iteration_time = time.time() - t0
            return result

        except Exception as e:
            logger.exception("Error in response_scorer: %s", e)
            return self._make_error_result(parent_program, f"exception: {e}", t0=t0)

    # ------------------------------------------------------------------
    # Idea repo finalisation helpers
    # ------------------------------------------------------------------

    def _finalize_idea_repo(
        self, island_id: int, new_idea_repo, last_bt_iter: bool,
        repo_idx_before_backtrack: int, trigger_merge: bool,
        append: bool = True,
    ):
        """Handle backtrack merge, idea-cap merge, and optionally append repo.
        Mirrors run_experiment.py lines 495-502. append=False on failure (no persist)."""
        import llm_utils
        Transcript = llm_utils.Transcript

        with self._lock:
            if last_bt_iter and self._merge_freq > -1:
                try:
                    pre_bt_repo = self._idea_repo_db.idea_repos[island_id][repo_idx_before_backtrack]
                    new_idea_repo.ideas.extend(pre_bt_repo.ideas)
                    new_idea_repo.reindex_ideas()
                except (IndexError, Exception) as e:
                    logger.error("Backtrack merge failed: %s", e)

            if trigger_merge:
                try:
                    import workflow_utils
                    workflow_utils.merge_ideas(
                        self._llm_name, self._get_transcript_filename(), self.config,
                        new_idea_repo, self._idea_cap,
                    )
                except Exception as e:
                    logger.error("merge_ideas failed: %s", e)

            if append:
                self._idea_repo_db.idea_repos[island_id].append(new_idea_repo)

    # ------------------------------------------------------------------
    # Recording stubs (SLIME compatibility)
    # ------------------------------------------------------------------

    def enable_recording(self, output_dir: str = "./gym_output"):
        self.recording_enabled = True
        self._record_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        from pacevolve.evolving_gym.pacevolve_recorder import PACEvolveRecorder
        self._recorder = PACEvolveRecorder(self, output_dir)
        # Redirect PACEvolve logger output to a file (logger.info is hidden by default)
        self._setup_pacevolve_file_logging(output_dir)
        print(f"[PACEvolve] Recording enabled: output_dir={os.path.abspath(output_dir)}", flush=True)

    def _setup_pacevolve_file_logging(self, output_dir: str) -> None:
        """Add a file handler so logger.info from PACEvolve modules is written to a file."""
        log_path = os.path.join(output_dir, "pacevolve.log")
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            for name in ("controller", logger.name, "pacevolve.evolving_gym.pacevolve_recorder"):
                lg = logging.getLogger(name)
                lg.setLevel(min(lg.level or logging.INFO, logging.DEBUG))
                lg.addHandler(file_handler)
            self._pacevolve_log_handler = file_handler
            print(f"[PACEvolve] Logs written to {log_path}", flush=True)
        except Exception as e:
            logger.warning("Could not set up PACEvolve file logging: %s", e)

    def record_progress(
        self,
        training_step: int,
        step_metrics: Optional[Dict[str, Any]] = None,
        data: Optional[List[List[Any]]] = None,
        **kwargs,
    ):
        if not self.recording_enabled:
            return
        if self._recorder is not None:
            self._recorder.record_step(
                rollout_id=training_step,
                step_metrics=step_metrics,
                data=data,
            )
        best = self._programs_db.get_best_program()
        best_score = best[0] if best[0] is not None else "N/A"
        print(f"[PACEvolve Record] step={training_step} best_score={best_score}", flush=True)

    def seed_puct_archive(self):
        pass

    async def check_and_reinit_database(self, *args, **kwargs):
        return False
