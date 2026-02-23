"""Async parallel multi-island evolution runner for PACEvolve.

Uses concurrent.futures.ProcessPoolExecutor to run island iterations in
parallel. Each worker process independently runs the full pipeline:
  candidate selection -> LLM call -> compile -> eval -> return result

The main process coordinates database updates and crossover scheduling.
"""

import asyncio
import dataclasses
import importlib
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, Future
from copy import deepcopy
from typing import Optional

logger = logging.getLogger("controller")

# ---------------------------------------------------------------------------
# Worker-side globals (initialised per process via _worker_init)
# ---------------------------------------------------------------------------
_worker_config = None
_worker_compile_config = None
_worker_eval_configs = None
_worker_llm_name = None
_worker_prompts = None
_worker_task_eval_utils = None


@dataclasses.dataclass
class IterationResult:
    """Serialisable result returned from a worker process."""
    iteration: int
    island_id: int
    program_code: Optional[str] = None
    eval_score: Optional[float] = None
    eval_results: list = dataclasses.field(default_factory=list)
    summary_bullets: list = dataclasses.field(default_factory=list)
    idea_id: int = -1
    success: bool = False
    error: Optional[str] = None
    elapsed: float = 0.0


def _worker_init(config_dict: dict, project_root: str):
    """Initialise heavy, non-picklable objects once per worker process."""
    global _worker_config, _worker_compile_config, _worker_eval_configs
    global _worker_llm_name, _worker_prompts, _worker_task_eval_utils

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import yaml
    import task_utils

    _worker_config = config_dict
    _worker_llm_name = config_dict['llm']['name']

    src_path = os.path.expanduser(config_dict['paths']['src_path'])
    _worker_compile_config = task_utils.CompilationConfig(
        target_file_path=os.path.join(src_path, config_dict['paths']['target_file_path']),
        pip_path=None,
    )

    task_id = config_dict['experiment']['task_id']
    _worker_task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")
    EvalConfig = _worker_task_eval_utils.EvalConfig
    _worker_eval_configs = [EvalConfig(**d) for d in config_dict['evaluation']['eval_configs']]

    prompt_filename = config_dict['experiment'].get('prompts_file', 'prompts')
    dataset_id = config_dict.get('_dataset_id', '.')
    if dataset_id == ".":
        _worker_prompts = importlib.import_module(f"tasks.{task_id}.config.{prompt_filename}")
    else:
        _worker_prompts = importlib.import_module(f"tasks.{task_id}.config.{dataset_id}.{prompt_filename}")


def _run_island_iteration(
    iteration: int,
    island_id: int,
    parent_code: str,
    idea_repo_snapshot,
    use_idea_repo: bool,
    use_idea_filter: bool,
    max_attempt: int,
    baseline_id: int,
    transcript_file: str,
) -> IterationResult:
    """Worker function executed in a child process.

    Runs the full pipeline for one iteration on a given island:
    idea generation -> LLM code generation -> compile -> eval -> summarise.
    """
    t0 = time.time()
    result = IterationResult(iteration=iteration, island_id=island_id)

    try:
        import llm_utils, workflow_utils

        Transcript = llm_utils.Transcript
        ContentChunk = llm_utils.ContentChunk
        AlgorithmTrial = workflow_utils.AlgorithmTrial

        config = _worker_config
        llm_name = _worker_llm_name
        prompts = _worker_prompts
        compile_config = _worker_compile_config
        eval_configs = _worker_eval_configs

        transcript = Transcript(log_filename=transcript_file)
        transcript.log_debug_message(f"### Starting parallel iteration {iteration} on island {island_id}")
        trial = AlgorithmTrial()

        sota_algo = parent_code
        new_idea_repo = deepcopy(idea_repo_snapshot) if idea_repo_snapshot else None

        if use_idea_repo and new_idea_repo is not None:
            new_idea_repo.sota = sota_algo
            import idea_select_utils
            idea_gen_prompt_text = prompts.construct_idea_gen_prompt(sota_algo, new_idea_repo)
            new_hypo = idea_select_utils.scratch_pad(new_idea_repo, llm_name, transcript, config, idea_gen_prompt_text)
            if not new_hypo:
                result.error = "Failed to generate new hypothesis"
                result.elapsed = time.time() - t0
                return result

            if use_idea_filter:
                for gen_hypo in range(max_attempt):
                    try:
                        prompt_text = prompts.construct_idea_select_no_code_prompt(sota_algo, new_idea_repo)
                        transcript.append(ContentChunk(prompt_text, "user", tags=[f"idea_selection_prompt_no_code_{gen_hypo}"]))
                        llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
                        transcript.append(ContentChunk(llm_response_text, "model", tags=[f"initial_response_no_code_{gen_hypo}"]))
                        if llm_response_text is None:
                            transcript.hide_by_tag(tags=[f"idea_selection_prompt_no_code_{gen_hypo}", f"initial_response_no_code_{gen_hypo}"])
                            continue
                        else:
                            idea_id, _ = idea_select_utils.parse_selected_idea(llm_response_text)
                            if idea_id is None:
                                transcript.hide_by_tag(tags=[f"idea_selection_prompt_no_code_{gen_hypo}", f"initial_response_no_code_{gen_hypo}"])
                                continue
                        break
                    except Exception:
                        continue
            else:
                prompt_text = prompts.construct_idea_select_prompt(sota_algo, new_idea_repo)
                transcript.append(ContentChunk(prompt_text, "user", tags=["idea_selection_prompt"]))
                llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
                transcript.append(ContentChunk(llm_response_text, "model", tags=["initial_response"]))
        else:
            prompt_text = prompts.construct_mutation_prompt(sota_algo, [])
            transcript.append(ContentChunk(prompt_text, "user", tags=["initial_prompt"]))
            llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
            transcript.append(ContentChunk(llm_response_text, "model", tags=["initial_response"]))

        # Compile
        trial = workflow_utils.edit_until_compile(
            llm_name, trial, transcript, compile_config, config,
            loop_config=config['workflow_loops']['initial_compile'],
            use_idea_repo=use_idea_repo,
        )
        transcript.hide_by_tag(tags=["initial_compile_loop"])
        if not trial.compile_success:
            result.error = "Compilation failed"
            result.elapsed = time.time() - t0
            return result

        # Eval
        trial = workflow_utils.edit_until_successful_eval(
            llm_name, trial, transcript, compile_config, eval_configs, config,
            iteration + 1, baseline_id,
            loop_config=config['workflow_loops']['initial_eval'],
        )
        transcript.hide_by_tag(tags=["initial_eval_loop"])
        if not all(trial.eval_success):
            result.error = "Evaluation failed"
            result.elapsed = time.time() - t0
            return result

        # Parse score
        eval_score = _worker_task_eval_utils.parse_eval_results(trial.eval_results)

        # Summarise
        transcript.append(ContentChunk(prompts.EVAL_DESCRIPTION_PROMPT, "user", tags=["initial_eval_results"]))
        eval_results_text = "\n".join(["```"] + trial.eval_results + ["```"])
        transcript.append(ContentChunk(eval_results_text, "system", tags=["initial_eval_results"]))

        summary_prompt = prompts.SUMMARIZE_EVAL_PROMPT
        transcript.append(ContentChunk(summary_prompt, "user", tags=["final_summary_request"]))
        bullets = []
        for idx in range(max_attempt):
            llm_summary = llm_utils.generate_completion(llm_name, transcript, config)
            if llm_summary:
                transcript.append(ContentChunk(llm_summary, "model", tags=["final_summary_response"]))
                try:
                    bullets = workflow_utils.extract_summary(llm_summary)
                except Exception:
                    pass
                break

        result.program_code = trial.algorithm_implementation
        result.eval_score = eval_score
        result.eval_results = trial.eval_results
        result.summary_bullets = bullets
        result.idea_id = trial.idea_id
        result.success = True
        result.elapsed = time.time() - t0
        return result

    except Exception as e:
        result.error = str(e)
        result.elapsed = time.time() - t0
        return result


# ---------------------------------------------------------------------------
# Main-process coordinator
# ---------------------------------------------------------------------------

async def run_parallel_evolution(
    config: dict,
    db,
    idea_repo_db,
    prompts_module,
    args,
    transcript_file: str,
    project_root: str,
    num_workers: int = 4,
):
    """Run multi-island evolution with true process-level parallelism.

    Submits island iterations to a ProcessPoolExecutor and processes results
    as they complete, updating the shared databases in the main process.
    """
    max_iters = config['experiment']['max_iters']
    baseline_id = config['experiment']['initial_baseline_id']
    num_islands = config['database']['num_islands']

    config_for_workers = deepcopy(config)
    config_for_workers['_dataset_id'] = args.dataset_id

    per_island_count = [0] * num_islands
    last_crossover_idx = [0] * num_islands
    in_flight_per_island = [0] * num_islands
    forced_parent_for_island: list[Optional[str]] = [None] * num_islands

    completed = 0
    submitted = 0
    next_island_rr = 0

    logger.info(
        f"Starting parallel evolution: {max_iters} iterations, "
        f"{num_workers} workers, {num_islands} islands"
    )

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(config_for_workers, project_root),
    )

    try:
        pending: dict[int, tuple[Future, int]] = {}

        def _choose_ready_island() -> Optional[int]:
            nonlocal next_island_rr
            for _ in range(num_islands):
                island_id = next_island_rr
                next_island_rr = (next_island_rr + 1) % num_islands
                if in_flight_per_island[island_id] == 0:
                    return island_id
            return None

        def _submit_one() -> bool:
            nonlocal submitted
            if submitted >= max_iters:
                return False
            if len(pending) >= num_workers:
                return False
            island_id_for_iter = _choose_ready_island()
            if island_id_for_iter is None:
                return False

            # One-step off-policy crossover parent override.
            if forced_parent_for_island[island_id_for_iter] is not None:
                parent_code = forced_parent_for_island[island_id_for_iter]
                forced_parent_for_island[island_id_for_iter] = None
            else:
                parent_code, _ = db.get_candidate_for_island(island_id_for_iter)

            idea_snapshot = None
            if args.use_idea_repo and idea_repo_db.idea_repos[island_id_for_iter]:
                idea_snapshot = deepcopy(idea_repo_db.idea_repos[island_id_for_iter][-1])

            iter_id = submitted
            fut = executor.submit(
                _run_island_iteration,
                iter_id,
                island_id_for_iter,
                parent_code,
                idea_snapshot,
                args.use_idea_repo,
                args.use_idea_filter,
                args.max_attempt,
                baseline_id,
                transcript_file,
            )
            pending[iter_id] = (fut, island_id_for_iter)
            submitted += 1
            in_flight_per_island[island_id_for_iter] += 1
            return True

        # Prime workers.
        while _submit_one():
            pass

        while completed < max_iters and pending:
            done_iter = None
            done_future = None
            done_island = None
            for it, (fut, island_id) in list(pending.items()):
                if fut.done():
                    done_iter = it
                    done_future = fut
                    done_island = island_id
                    break

            if done_iter is None:
                await asyncio.sleep(0.05)
                continue

            pending.pop(done_iter)
            in_flight_per_island[done_island] = max(0, in_flight_per_island[done_island] - 1)

            try:
                result: IterationResult = done_future.result()
            except Exception as e:
                logger.error(f"Iteration {done_iter} raised exception: {e}")
                completed += 1
                while _submit_one():
                    pass
                continue

            island_id = result.island_id
            if result.success and result.eval_score is not None:
                db.register_program(
                    program=result.program_code,
                    island_id=island_id,
                    score=result.eval_score,
                )
                idea_repo_db.best_scores_history[island_id].append(result.eval_score)
                idea_repo_db.scheduler.update_score(island_id, result.eval_score)
                per_island_count[island_id] += 1

                # Keep crossover trigger semantics close to sequential:
                # trigger after freeze period and after score update.
                if (
                    args.use_integrated_sampling
                    and per_island_count[island_id] - last_crossover_idx[island_id] > args.freeze_period
                    and idea_repo_db.scheduler.check_trigger(island_id)
                ):
                    action, cross_id = idea_repo_db.scheduler.sample_action(island_id)
                    if action == "CROSSOVER" and cross_id is not None:
                        last_crossover_idx[island_id] = per_island_count[island_id]
                        # One-step off-policy: next sample on this island uses partner island SOTA.
                        _, cross_sota = db.get_best_program_for_island(cross_id)
                        if cross_sota is not None:
                            forced_parent_for_island[island_id] = cross_sota
                        if idea_repo_db.idea_repos[island_id]:
                            current_repo = deepcopy(idea_repo_db.idea_repos[island_id][-1])
                            best_repo = idea_repo_db.get_best_idea_repo(cross_id)
                            current_repo.ideas.extend(best_repo.ideas)
                            current_repo.reindex_ideas()
                            idea_repo_db.idea_repos[island_id].append(current_repo)

                logger.info(
                    f"Iteration {result.iteration} (island {island_id}) completed in "
                    f"{result.elapsed:.1f}s  score={result.eval_score}"
                )
                if result.summary_bullets:
                    logger.info("Summary: " + " | ".join(result.summary_bullets[:3]))
            else:
                logger.warning(
                    f"Iteration {result.iteration} (island {island_id}) failed: "
                    f"{result.error or 'unknown'}"
                )

            completed += 1
            while _submit_one():
                pass

        logger.info(f"Parallel evolution finished. {completed}/{max_iters} iterations completed.")

    finally:
        executor.shutdown(wait=False)
