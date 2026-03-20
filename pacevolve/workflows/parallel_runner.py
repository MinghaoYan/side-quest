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
    updated_idea_repo: Optional[object] = None
    analysis_success: bool = False
    analysis_attempts: int = 0
    analysis_metrics: dict = dataclasses.field(default_factory=dict)
    analysis_errors: list = dataclasses.field(default_factory=list)


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


def _finalize_idea_repo_worker(
    new_idea_repo,
    last_bt_iter: bool,
    pre_bt_idea_repo,
    trigger_merge: bool,
    idea_cap: int,
    merge_freq: int,
    transcript_file: str,
):
    """Handle backtrack merge and idea-cap merge inside the worker process.

    Mirrors run_experiment.py lines 495-500 and single_task_gym._finalize_idea_repo.
    """
    if last_bt_iter and merge_freq > -1 and pre_bt_idea_repo is not None:
        try:
            new_idea_repo.ideas.extend(pre_bt_idea_repo.ideas)
            new_idea_repo.reindex_ideas()
        except Exception as e:
            logger.error("Backtrack merge failed in worker: %s", e)

    if trigger_merge and new_idea_repo is not None:
        try:
            import workflow_utils
            workflow_utils.merge_ideas(
                _worker_llm_name, transcript_file, _worker_config,
                new_idea_repo, idea_cap,
            )
        except Exception as e:
            logger.error("merge_ideas failed in worker: %s", e)


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
    trigger_merge: bool = False,
    last_bt_iter: bool = False,
    pre_bt_idea_repo_snapshot=None,
    idea_cap: int = -1,
    merge_freq: int = -1,
    summarize_freq: int = 20,
) -> IterationResult:
    """Worker function executed in a child process.

    Runs the full pipeline for one iteration on a given island:
    idea generation -> LLM code generation -> compile -> eval -> summarise.

    Also handles idea repo updates (experiment history, summarization),
    backtrack merges, and idea-cap merges -- matching sequential mode.
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
                _finalize_idea_repo_worker(
                    new_idea_repo, last_bt_iter, pre_bt_idea_repo_snapshot,
                    trigger_merge, idea_cap, merge_freq, transcript_file,
                )
                result.updated_idea_repo = new_idea_repo
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
            _finalize_idea_repo_worker(
                new_idea_repo, last_bt_iter, pre_bt_idea_repo_snapshot,
                trigger_merge, idea_cap, merge_freq, transcript_file,
            )
            result.updated_idea_repo = new_idea_repo
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
        if bool(config.get("analysis", {}).get("enabled", True)):
            try:
                analysis_prompt = workflow_utils.resolve_post_eval_analysis_prompt(
                    prompts, trial, transcript, config
                )
                trial = workflow_utils.run_post_eval_analysis(
                    llm_name=llm_name,
                    trial=trial,
                    transcript=transcript,
                    config=config,
                    analysis_prompt=analysis_prompt,
                    max_attempts=max(1, min(max_attempt, 3)),
                )
            except Exception as exc:
                logger.warning(
                    "Parallel post-eval analysis crashed for iter %s island %s: %s",
                    iteration,
                    island_id,
                    exc,
                )
                trial.analysis_success = False
                trial.analysis_errors.append(f"Post-eval analysis crashed: {exc}")
            transcript.hide_by_tag(tags=["post_eval_analysis_loop"])
            result.analysis_success = trial.analysis_success
            result.analysis_attempts = trial.analysis_attempts
            result.analysis_metrics = trial.analysis_metrics
            result.analysis_errors = trial.analysis_errors
        if not all(trial.eval_success):
            _finalize_idea_repo_worker(
                new_idea_repo, last_bt_iter, pre_bt_idea_repo_snapshot,
                trigger_merge, idea_cap, merge_freq, transcript_file,
            )
            result.updated_idea_repo = new_idea_repo
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

        # Update idea repo: experiment history + summarization
        if use_idea_repo and new_idea_repo is not None and trial.idea_id != -1:
            try:
                import idea_select_utils
                idea_obj = new_idea_repo.find_idea_by_id(trial.idea_id)
                if idea_obj is not None:
                    idea_obj.exp_history.extend(bullets)
                    idea_obj.exp_count += 1
                    if summarize_freq > 0 and idea_obj.exp_count % summarize_freq == 0:
                        idea_select_utils.summarize(
                            idea_obj, llm_name, config,
                            Transcript(log_filename=transcript_file),
                            Transcript(log_filename=transcript_file),
                        )
            except Exception as e:
                logger.error("Idea repo update/summarize failed: %s", e)

        # Backtrack merge + idea-cap merge
        _finalize_idea_repo_worker(
            new_idea_repo, last_bt_iter, pre_bt_idea_repo_snapshot,
            trigger_merge, idea_cap, merge_freq, transcript_file,
        )

        result.program_code = trial.algorithm_implementation
        result.eval_score = eval_score
        result.eval_results = trial.eval_results
        result.summary_bullets = bullets
        result.idea_id = trial.idea_id
        result.success = True
        result.elapsed = time.time() - t0
        result.updated_idea_repo = new_idea_repo
        result.analysis_success = trial.analysis_success
        result.analysis_attempts = trial.analysis_attempts
        result.analysis_metrics = trial.analysis_metrics
        result.analysis_errors = trial.analysis_errors
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

    Now includes backtracking (momentum-based + periodic), merge_ideas,
    idea summarization, and failure-path merges -- matching sequential mode.
    """
    import idea_select_utils

    max_iters = config['experiment']['max_iters']
    baseline_id = config['experiment']['initial_baseline_id']
    num_islands = config['database']['num_islands']

    config_for_workers = deepcopy(config)
    config_for_workers['_dataset_id'] = args.dataset_id

    # Per-island counters (incremented on submit, matching sequential)
    per_island_count = [0] * num_islands
    last_crossover_idx = [0] * num_islands
    in_flight_per_island = [0] * num_islands
    forced_parent_for_island: list[Optional[str]] = [None] * num_islands

    # Backtrack state per island
    backtrack_triggered_idx: list[int] = [-1] * num_islands
    repo_idx_before_backtrack: list[int] = [0] * num_islands

    # Workflow params (mirror args used in sequential mode)
    backtrack_freq = getattr(args, 'backtrack_freq', -1)
    backtrack_len = getattr(args, 'backtrack_len', 5)
    power_alpha = getattr(args, 'power_alpha', 1.5)
    idea_cap = getattr(args, 'idea_cap', 5)
    merge_freq = getattr(args, 'merge_freq', 1)
    summarize_freq = getattr(args, 'summarize_freq', 20)
    use_integrated_sampling = getattr(args, 'use_integrated_sampling', True)
    freeze_period = getattr(args, 'freeze_period', 15)

    completed = 0
    submitted = 0
    next_island_rr = 0

    logger.info(
        f"Starting parallel evolution: {max_iters} iterations, "
        f"{num_workers} workers, {num_islands} islands"
    )
    logger.info(
        f"  backtrack_freq={backtrack_freq} backtrack_len={backtrack_len} "
        f"power_alpha={power_alpha} idea_cap={idea_cap} merge_freq={merge_freq} "
        f"summarize_freq={summarize_freq}"
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

            island = island_id_for_iter
            trigger_backtrack = False
            last_bt_iter = False
            pre_bt_snapshot = None

            # ---- Backtrack / parent selection logic (mirrors sequential) ----
            bt_idx = backtrack_triggered_idx[island]
            if bt_idx > -1 and per_island_count[island] - bt_idx < backtrack_len:
                # Ongoing backtrack
                if per_island_count[island] - bt_idx == backtrack_len - 1:
                    last_bt_iter = True
                parent_code = idea_repo_db.idea_repos[island][-1].sota
                idea_snapshot = deepcopy(idea_repo_db.idea_repos[island][-1])
                if last_bt_iter:
                    pre_bt_snapshot = deepcopy(
                        idea_repo_db.idea_repos[island][repo_idx_before_backtrack[island]]
                    )
            else:
                # Clear stale backtrack state
                if bt_idx > -1:
                    backtrack_triggered_idx[island] = -1

                # One-step off-policy crossover parent override
                if forced_parent_for_island[island] is not None:
                    parent_code = forced_parent_for_island[island]
                    forced_parent_for_island[island] = None
                else:
                    parent_code, _ = db.get_candidate_for_island(island)

                if len(idea_repo_db.idea_repos[island]) == 0:
                    idea_snapshot = None
                else:
                    idea_snapshot = deepcopy(idea_repo_db.idea_repos[island][-1])

                # Integrated sampling: crossover / backtrack trigger
                if (
                    use_integrated_sampling
                    and per_island_count[island] - last_crossover_idx[island] > freeze_period
                    and idea_repo_db.scheduler.check_trigger(island)
                ):
                    action, cross_id = idea_repo_db.scheduler.sample_action(island)
                    if action == "CROSSOVER" and cross_id is not None:
                        last_crossover_idx[island] = per_island_count[island]
                        _, cross_sota = db.get_best_program_for_island(cross_id)
                        if cross_sota is not None:
                            parent_code = cross_sota
                        if idea_repo_db.idea_repos[island]:
                            best_repo = idea_repo_db.get_best_idea_repo(cross_id)
                            if idea_snapshot is None:
                                idea_snapshot = deepcopy(best_repo)
                            else:
                                idea_snapshot.ideas.extend(best_repo.ideas)
                                idea_snapshot.reindex_ideas()
                            idea_repo_db.idea_repos[island].append(deepcopy(idea_snapshot))
                    elif action == "BACKTRACK":
                        last_crossover_idx[island] = per_island_count[island]
                        trigger_backtrack = True

                # Periodic or triggered backtrack
                count = per_island_count[island]
                if (backtrack_freq != -1 and (count + 1) % backtrack_freq == 0) or trigger_backtrack:
                    if len(idea_repo_db.idea_repos[island]) > 0:
                        backtrack_triggered_idx[island] = count
                        repo_idx_before_backtrack[island] = len(idea_repo_db.idea_repos[island]) - 1
                        sampled_idx = idea_select_utils.sample_power_law(
                            len(idea_repo_db.idea_repos[island]),
                            alpha=power_alpha,
                        )
                        parent_code = idea_repo_db.idea_repos[island][sampled_idx].sota
                        idea_snapshot = deepcopy(idea_repo_db.idea_repos[island][sampled_idx])

            # ---- Compute trigger_merge (mirrors sequential) ----
            trigger_merge = False
            if idea_cap > -1 and idea_snapshot is not None and len(idea_snapshot.ideas) > idea_cap and merge_freq > -1:
                if use_integrated_sampling and per_island_count[island] - last_crossover_idx[island] == freeze_period:
                    trigger_merge = True
                elif backtrack_freq == -1 or (per_island_count[island] + 1) % backtrack_freq == backtrack_freq - 1:
                    trigger_merge = True

            logger.info(
                f"Submitting iter {submitted} on island {island}: "
                f"trigger_merge={trigger_merge} last_bt_iter={last_bt_iter} "
                f"bt_active={backtrack_triggered_idx[island] > -1}"
            )

            # Increment per-island count on submit (matching sequential)
            per_island_count[island] += 1

            iter_id = submitted
            fut = executor.submit(
                _run_island_iteration,
                iter_id,
                island,
                parent_code,
                idea_snapshot,
                args.use_idea_repo,
                args.use_idea_filter,
                args.max_attempt,
                baseline_id,
                transcript_file,
                trigger_merge,
                last_bt_iter,
                pre_bt_snapshot,
                idea_cap,
                merge_freq,
                summarize_freq,
            )
            pending[iter_id] = (fut, island)
            submitted += 1
            in_flight_per_island[island] += 1
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

            # Apply updated idea repo from the worker (covers both success & failure)
            if result.updated_idea_repo is not None and args.use_idea_repo:
                idea_repo_db.idea_repos[island_id].append(result.updated_idea_repo)

            if result.success and result.eval_score is not None:
                db.register_program(
                    program=result.program_code,
                    island_id=island_id,
                    score=result.eval_score,
                )
                idea_repo_db.best_scores_history[island_id].append(result.eval_score)
                idea_repo_db.scheduler.update_score(island_id, result.eval_score)

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
