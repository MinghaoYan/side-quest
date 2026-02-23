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

import copy
import dataclasses
import traceback
import logging
import os
import re

from concurrent.futures import ThreadPoolExecutor, as_completed

import llm_utils
import task_utils, idea_select_utils
import importlib
from copy import deepcopy

logger = logging.getLogger("controller")

Transcript = llm_utils.Transcript
ContentChunk = llm_utils.ContentChunk
CompilationConfig = task_utils.CompilationConfig
CompletedProcess = task_utils.CompletedProcess

NUM_CPU_CORES = 32


@dataclasses.dataclass
class AlgorithmTrial:
  algorithm_implementation: str = ""
  compile_success: bool = False
  eval_success: list[bool] = dataclasses.field(default_factory=list)
  eval_results: list[str] = dataclasses.field(default_factory=list)
  idea_id: int = -1


def _summarize_compile_error(process: CompletedProcess, config: dict) -> str:
  summ_config = config['summarization']
  error_summary_lines = []
  lines = process.stdout.splitlines() + process.stderr.splitlines()
  context_lines_after_error = summ_config['context_lines_after_error']

  indices_with_errors = [i for i, line in enumerate(lines) if "error:" in line.lower()]

  if not indices_with_errors and process.stderr.strip():
    max_lines_fallback = 40
    fallback_summary = "\n".join(lines[-max_lines_fallback:])
    error_description = f"No specific 'error:' lines found. Showing last {max_lines_fallback} lines of output:\n{fallback_summary}"
  elif not indices_with_errors and not process.stderr.strip():
    error_description = "Compilation failed, but no stderr/stdout output was captured or it was empty."
  else:
    line_indices_to_include = set()
    for error_idx in indices_with_errors:
      for i in range(max(0, error_idx - 1), min(len(lines), error_idx + context_lines_after_error + 1)):
        line_indices_to_include.add(i)
    
    sorted_indices = sorted(list(line_indices_to_include))
    max_summary_lines = summ_config['max_summary_lines']
    last_printed_idx = -2
    for idx in sorted_indices:
      if idx > last_printed_idx + 1 and last_printed_idx != -2 :
        if len(error_summary_lines) < max_summary_lines: error_summary_lines.append("[...]")
      if len(error_summary_lines) < max_summary_lines: error_summary_lines.append(lines[idx])
      last_printed_idx = idx
      if len(error_summary_lines) >= max_summary_lines: break
    
    if len(error_summary_lines) >= max_summary_lines and sorted_indices[-1] > last_printed_idx:
      error_summary_lines.append("\n[Error summary truncated due to length...]")
    error_description = "\n".join(error_summary_lines)

    # Log full compile output to DEBUG to avoid flooding INFO, summary still INFO
    logger.debug(f"summarize_compile_error: Full compile output (stdout/stderr combined):\n{process.stderr}")
    logger.info(f"summarize_compile_error: Summarized compile error sent to LLM:\n{error_description}")

  return error_description


def attempt_compile(
  trial: AlgorithmTrial,
  compile_config: CompilationConfig,
  config: dict,
) -> tuple[AlgorithmTrial, str]:
  try:
    edit_library(
      compile_config.target_file_path,
      algorithm_implementation=trial.algorithm_implementation,
      config=config,
    )
  except ValueError as e:
    error_message = f"INTERNAL ERROR: Failed to edit library: {e}"
    logger.critical(f"attempt_compile: {error_message}")
    return trial, error_message
  
  task_id = config['experiment']['task_id']
  # Dynamically import task-specific eval_utils
  task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")

  compile_output = task_eval_utils.recompile_library(config)
  success = (compile_output.returncode == 0)

  output_message = "Code compiled successfully."
  if not success:
    error_description = _summarize_compile_error(compile_output, config)
    output_message = (
      f"The code provided did not compile. Compiler output (summarized):\n{error_description}\n"
      "Please analyze these errors and provide a corrected version of the code. "
      "Ensure your response contains a complete code block. Focus on fixing the problem."
    )
  command_output = (
    compile_output.stdout.splitlines() + compile_output.stderr.splitlines()
  )
  for line in command_output:
    logger.debug(f"attempt_compile: {line}")
  output_trial = copy.deepcopy(trial)
  output_trial.compile_success = success
  return output_trial, output_message


def edit_until_compile(
  llm_name,
  trial: AlgorithmTrial,
  transcript: Transcript,
  compile_config: CompilationConfig,
  config: dict,
  loop_config: dict,
  use_idea_repo: bool=False,
) -> AlgorithmTrial:
  max_compile_attempts = loop_config['max_attempts']
  compile_loop_tag = loop_config['loop_tag']
  compile_summary_tag = loop_config['summary_tag']
  num_attempts = 0
  code_was_revised = False
  recovery_prompt = None
  trial = copy.deepcopy(trial)  # Do not modify the original trial object.
  while num_attempts < max_compile_attempts:
    num_attempts += 1
    logger.info(f"edit_until_compile: {num_attempts}/{max_compile_attempts}")
    # If we have a recovery prompt from the last loop, re-prompt the LLM.
    if recovery_prompt:
      code_was_revised = True
      transcript.append(
        ContentChunk(recovery_prompt, "user", tags=[compile_loop_tag])
      )
      recovery_response = llm_utils.generate_completion(llm_name, transcript, config)
      transcript.append(
        ContentChunk(recovery_response, "model", tags=[compile_loop_tag])
      )

    if not transcript or transcript[-1].role != "model":
      # If we do not have a message from the model at the end of the transcript,
      # we need to re-prompt the LLM to provide the C++ code.
      logger.critical(
        "edit_unil_compile: Expected latest message to be from 'model'."
      )
      recovery_prompt = "Error: No model response found. Please respond."
      continue

    current_llm_response = transcript[-1].content
    logger.debug(f"edit_until_compile: LLM Response:\n{current_llm_response}")
    # print(f"edit_until_compile: LLM Response:\n{current_llm_response}")
    if not current_llm_response:
      logger.warning("edit_until_compile: No response.")
      recovery_prompt = (
        "Your output did not contain any markdown-formatted code blocks. "
        "Please provide one."
      )
      continue

    if use_idea_repo:
      idea_id = idea_select_utils.extract_idea_id(current_llm_response)
      if not idea_id:
        logger.warning("edit_until_compile: Idea ID not found in response.")
        recovery_prompt = (
          "Your output did not contain Idea ID for the selected idea. "
          "Please provide one."
        )
        continue
      else:
        logger.info(f"edit_until_compile: Setting idea ID to {idea_id}.")
        trial.idea_id = idea_id


    code_blocks = llm_utils.extract_code_blocks(current_llm_response)

    if not code_blocks:
      logger.warning("edit_until_compile: Code blocks not found in response.")
      recovery_prompt = (
        "Your output did not contain any markdown-formatted code blocks. "
        "Please provide one."
      )
      continue

    trial.algorithm_implementation = code_blocks[0]  # Use the first block.
    trial, recovery_prompt = attempt_compile(trial, compile_config, config)

    if trial.compile_success:
      logger.info(f"edit_until_compile: Attempt {num_attempts} successful")
      # All of the state is contained within the trial object.
      break

  if trial.compile_success and code_was_revised:
    transcript.log_debug_message(
      "Code had compilation errors/was missing but compiled after revisions."
    )
    summary_text = (
      "Code had errors but compiled after revisions. The code that was finally "
      f"installed:\n```cpp\n{trial.algorithm_implementation}\n```"
    )
    transcript.append(
      ContentChunk(summary_text, "system", tags=[compile_summary_tag])
    )
  elif trial.compile_success and not code_was_revised:
    summary_text = (
      "Code compiled successfully and was installed."
    )
    transcript.log_debug_message("Code compiled on the first attempt.")
    transcript.append(
      ContentChunk(summary_text, "system", tags=[compile_summary_tag])
    )
  else:
    error_message = (
      f"Compilation failed even after {num_attempts} attempts to fix the code."
    )
    logger.warning(error_message)
    transcript.log_debug_message(error_message)
    transcript.append(
      ContentChunk(error_message, "system", tags=[compile_summary_tag])
    )

  return trial

# Map file extensions to their single-line comment markers
COMMENT_MARKERS: dict = {
    'cpp': '//', 'h': '//', 'cc': '//', 'cxx': '//', 'c': '//', 'java': '//',
    'js': '//', 'mjs': '//', 'cjs': '//', 'ts': '//', 'go': '//',
    'py': '#', 'sh': '#', 'bash': '#', 'zsh': '#',
    'rb': '#', 'pl': '#', 'pm': '#', 'r': '#', 'yaml': '#', 'yml': '#',
}

def get_comment_marker_for_file(file_path: str) -> str:
    """Infers the comment marker from the file extension."""
    _, ext = os.path.splitext(file_path)
    extension = ext.lower().lstrip('.')
    marker = COMMENT_MARKERS.get(extension)
    if marker is None:
        raise ValueError(
            f"Unsupported file extension '{extension}' for {file_path}. "
            "Cannot determine comment style."
        )
    return marker

def edit_library(
  target_file_path: str,
  algorithm_implementation: str,
  config: dict,
):
  """
  Edits a target file by replacing content between special comment tags.

  Determines the comment style based on the file extension.
  """
  start_tag: str = config['compilation']['edit_start_tag']
  end_tag: str = config['compilation']['edit_end_tag']

  try:
    comment_marker = get_comment_marker_for_file(target_file_path)
    logger.info(f"Using comment marker '{comment_marker}' for {target_file_path}")
  except ValueError as e:
    logger.critical(f"edit_library: {e}")
    raise

  logger.info(
    "edit_library: Preparing to write to the library file: "
    f"{target_file_path}"
  )
  logger.debug("edit_library: --- Code to be inserted: ---")
  logger.debug(f"CODE: \n{algorithm_implementation}")
  logger.debug("edit_library: --- End of code to be inserted ---")

  with open(target_file_path, 'r') as file:
    content = file.read()

  esc_comment = re.escape(comment_marker)
  esc_start_tag = re.escape(start_tag)
  esc_end_tag = re.escape(end_tag)

  # Regex to find the block:
  # Group 1: The entire start tag line, including the comment marker and any whitespace.
  # Group 2: The content between the start and end tags (what will be replaced).
  # Group 3: The entire end tag line.
  pattern = re.compile(
      f"(^[ \t]*{esc_comment}[ \t]*{esc_start_tag}[ \t]*\n)"  # Group 1: Start tag line
      f"(.*?)"  # Group 2: Content between tags
      f"(^[ \t]*{esc_comment}[ \t]*{esc_end_tag}[ \t]*$)",  # Group 3: End tag line
      re.DOTALL | re.MULTILINE
  )

  if not pattern.search(content):
      logger.critical(
          f"edit_library: Tag block not found for '{comment_marker} {start_tag}' "
          f"and '{comment_marker} {end_tag}' in {target_file_path}."
      )
      raise ValueError(f"Start/end tag block not found in {target_file_path}.")

  def replacer(match):
      return f"{match.group(1)}{algorithm_implementation}\n{match.group(3)}"

  new_content, num_subs = pattern.subn(replacer, content, count=1)

  if num_subs == 0:
      # This case should ideally not be reached if the search above passed.
      logger.critical(
          "edit_library: Tags found but pattern substitution failed for "
          f"'{start_tag}'/'{end_tag}' in {target_file_path}."
      )
      raise ValueError("Tags found, but pattern substitution failed.")

  with open(target_file_path, 'w') as file:
    file.write(new_content)

  logger.info(
    "edit_library: Successfully wrote to the library file: "
    f"{target_file_path}"
  )


def attempt_evals(
  eval_configs: list,
  trial: AlgorithmTrial,
  candidate_id: int,
  baseline_id: int,
  config: dict,
  max_parallel_evals: int = 5,
) -> AlgorithmTrial:
  trial = copy.deepcopy(trial)  # Do not modify the original trial object.
  trial.eval_success = [False for _ in eval_configs]
  trial.eval_results = ["" for _ in eval_configs]
  # num_build_threads = max(1, config['experiment']['num_cpu_cores'] // max_parallel_evals)
  logger.info(
    "attempt_evals: Starting eval for datasets: "
    f"{','.join([config.dataset for config in eval_configs])} "
    f"(Cand ID: {candidate_id})"
  )
  num_workers = min(len(eval_configs), max_parallel_evals)
  task_id = config['experiment']['task_id']
  # Dynamically import task-specific eval_utils
  task_eval_utils = importlib.import_module(f"tasks.{task_id}.eval.eval_utils")
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_idx = {
      executor.submit(task_eval_utils.evaluate_dataset, candidate_id, baseline_id, cfg, config): i 
      for i, cfg in enumerate(eval_configs)
    }
    for future in as_completed(future_to_idx):
      idx = future_to_idx[future]
      try:
        completed_process = future.result()
        result_text = "\n".join([
          completed_process.stdout.strip(),
          completed_process.stderr.strip(),
        ])
        trial.eval_success[idx] = (completed_process.returncode == 0)
        trial.eval_results[idx] = result_text
      except Exception as exc:
        trial.eval_success[idx] = False
        trial.eval_results[idx] = (
          f"Eval for {eval_configs[idx].dataset} failed in thread: "
          f"{exc}\n{traceback.format_exc()}"
        )
  return trial


def edit_until_successful_eval(
  llm_name,
  trial: AlgorithmTrial,
  transcript: Transcript,
  compile_config: CompilationConfig,
  eval_configs: list,
  config: dict,
  candidate_id: int,
  baseline_id: int,
  loop_config: dict,
) -> AlgorithmTrial:
  trial = copy.deepcopy(trial)  # Do not modify the original trial object.
  # If the code was not been compiled yet, attempt to do so.
  if not trial.compile_success:
    raise ValueError("Trial must have been compiled before evals.")

  max_eval_attempts = loop_config['max_attempts']
  eval_loop_tag = loop_config['loop_tag']
  eval_summary_tag = loop_config['summary_tag']
  compile_loop_config = {
      'max_attempts': loop_config['max_compile_attempts'],
      'loop_tag': loop_config['compile_loop_tag'],
      'summary_tag': loop_config['compile_summary_tag'],
  }

  code_was_revised = False
  num_attempts = 0
  while num_attempts < max_eval_attempts:
    num_attempts += 1
    logger.info(
      f"edit_until_successful_eval: {num_attempts}/{max_eval_attempts} "
      f"for candidate ID {candidate_id}."
    )

    # Attempt evals, assuming that the current code is installed to the library.
    trial = attempt_evals(
      eval_configs, trial, candidate_id, baseline_id, config, max_parallel_evals=5
    )
    # If all trials were successful, we can break out of the loop; eval is done.
    if all(trial.eval_success):
      logger.info(
        f"edit_until_successful_eval: Attempt {num_attempts} succeeded."
      )
      break
    # If we are out of attempts, we should break and not try to fix the code.
    if num_attempts >= max_eval_attempts:
      break

    # If we reach here, at least one eval failed and we still have attempts left
    # to fix it. We re-prompt the LLM to fix the code and attempt a re-compile.
    logger.info(
      f"edit_until_successful_eval: Eval failed for candidate {candidate_id} "
      f"on attempt {num_attempts}. Requesting LLM fix."
    )
    transcript.append(
      ContentChunk(
        "The code compiled, but evaluation has run-time errors.",
        "user",
        tags=[eval_loop_tag]
      )
    )
    failed_eval_messages = [
      msg for msg, flag in zip(trial.eval_results, trial.eval_success)
      if not flag
    ]
    transcript.append(
      ContentChunk(
        "\n".join(f"- {m}" for m in failed_eval_messages),
        "system",
        tags=[eval_loop_tag]
      )
    )
    transcript.append(
      ContentChunk(
        "Analyze the issue and provide corrected code in a markdown block.",
        "user",
        tags=[eval_loop_tag]
      )
    )
    # Generate a new code block from the LLM to fix the eval issues.
    llm_fix_response = llm_utils.generate_completion(llm_name, transcript, config)
    transcript.append(
      ContentChunk(llm_fix_response, "model", tags=[eval_loop_tag])
    )
    # Attempt to compile the new code provided by the LLM
    trial = edit_until_compile(
      llm_name, trial, transcript, compile_config, config,
      loop_config=compile_loop_config,
    )
    # Squash any compile edits from the transcript by hiding those chunks.
    transcript.hide_by_tag(tags=[compile_loop_config['loop_tag']])
    code_was_revised = True
    # If compile failed, then we will have the same eval issue and will loop
    # again. This is fine, because it gives the LLM another chance to fix it.

  # If we reach here, we either succeeded in evals or exhausted the attempts.
  final_success = all(trial.eval_success)
  if final_success:
    logger.info(
      f"edit_until_successful_eval: All evals ran for candidate {candidate_id} "
      f"after {num_attempts} attempt(s)."
    )
    if not code_was_revised:
      summary_text = "Evals ran without modifications to the code."
    else:
      summary_text = (
        "Evals ran successfully after code revisions. The final code that "
        f"was evaluated:\n```cpp\n{trial.algorithm_implementation}\n```"
      )
  else:
    logger.error(
      f"edit_until_successful_eval: Failed for candidate {candidate_id} "
      f"after {num_attempts} attempt(s)."
    )
    summary_text = (
      "Evals failed even after multiple attempts to fix the code. "
    )
  transcript.append(
    ContentChunk(summary_text, "system", tags=[eval_summary_tag])
  )
  return trial


def extract_summary(response_text: str) -> list[str]:
  bullets = []
  lines = response_text.strip().split("\n")
  for line in lines:
    line = line.strip()
    if line.startswith(("- ", "* ")) :
      bullets.append(line)
  return bullets



def backtrack_idea_tts(
    idea_repo_db: idea_select_utils.IdeaRepoDatabase,
    llm_name: str,
    config: dict,
    transcript: Transcript,
    prompts: object,
    transcript_file: str,
    ablation_list: list,
    power_alpha: float = 1.5,
    num_samples: int = 10,
) -> tuple[idea_select_utils.Idea, idea_select_utils.IdeaRepo] | None:
  """
  Performs an advanced backtracking search.

  1. Samples `num_samples` historical states using a power-law distribution.
  2. Generates one promising new idea from each sampled state.
  3. Asks the LLM to provide a rationale for each of the `num_samples` ideas and select the best one.
  4. Returns the selected idea and its parent repository.

  Args:
    idea_repo_db: The database containing all historical IdeaRepo states.
    llm_name: The name of the language model to use.
    config: The experiment configuration dictionary.
    transcript: The main transcript for logging interactions.
    prompts: The imported prompts module for the task.
    power_alpha: The alpha parameter for the power-law sampling.
    num_samples: The number of historical states to sample (default: 10).

  Returns:
    A tuple containing the selected Idea object and a deepcopy of its
    parent IdeaRepo, or None if the process fails.
  """
  logger.info(f"--- Starting Advanced Backtracking with {num_samples} samples ---")
  if not idea_repo_db.idea_repos:
    logger.error("IdeaRepoDatabase is empty. Cannot perform backtracking.")
    return None

  candidate_ideas = []
  parent_repos = []
  for i in range(num_samples):
    # Step 1: Sample a historical state
    sampled_idx = idea_select_utils.sample_power_law(len(idea_repo_db.idea_repos), alpha=power_alpha)
    parent_repo = deepcopy(idea_repo_db.idea_repos[sampled_idx])
    sota_algo = idea_repo_db.idea_repos[sampled_idx].sota
    logger.info(f"Sample {i+1}/{num_samples}: Selected historical repo index {sampled_idx}.")

    # Step 2: Generate a new idea from this state
    idea_gen_prompt = prompts.construct_idea_gen_prompt(parent_repo.sota, parent_repo)
    
    # Using scratch_pad to generate a set of new ideas for the sampled repo
    new_hypotheses_generated = idea_select_utils.scratch_pad(parent_repo, llm_name, transcript, config, idea_gen_prompt)

    if not new_hypotheses_generated:
      logger.warning(f"Failed to generate new hypotheses for sampled repo index {sampled_idx}. Skipping sample.")
      continue
    
    prompt_text = prompts.construct_idea_tournament_prompt(sota_algo, parent_repo)
    transcript.append(
      ContentChunk(prompt_text, "user", tags=["idea_tournament_prompt"])
    )
    llm_response_text = llm_utils.generate_completion(llm_name, transcript, config)
    transcript.append(
      ContentChunk(llm_response_text, "model", tags=["initial_response"])
    )
    candidate_ideas.append(llm_response_text)
    parent_repos.append(parent_repo)

  if not candidate_ideas:
    logger.error("Failed to generate any candidate ideas after sampling. Aborting backtracking.")
    return None
  
  max_attempts = 3
  filtered_proposals = []
  filtered_repos = []
  for attempt in range(max_attempts):
    try:
      filtered_proposals = []
      filtered_repos = []
      idea_filtering_transcript = Transcript(log_filename=transcript_file)
      idea_filtering_prompt = idea_select_utils.construct_idea_filter_prompt(ablation_list, candidate_ideas)
      idea_filtering_transcript.append(ContentChunk(idea_filtering_prompt, "user", tags=["filter_ideas"]))
      
      ideas_to_filter = llm_utils.generate_completion(llm_name, idea_filtering_transcript, config)
      
      ideas_to_filter_list = idea_select_utils.parse_llm_idea_list(ideas_to_filter)
      if ideas_to_filter_list is None:
        logger.error(f"Parsed output was not a valid list of idea IDs on attempt {attempt + 1}. Retrying.")
        continue  # Go to the next attempt in the loop
      elif len(ideas_to_filter_list) == 0:
        break
      else:
        for filter_idx in range(len(candidate_ideas)):
          if filter_idx not in ideas_to_filter_list:
            filtered_proposals.append(candidate_ideas[filter_idx])
            filtered_repos.append(parent_repos[filter_idx])
      break
    except:
      logger.error(f"Unexpected error occurs in on attempt {attempt + 1}. Retrying.")
      continue


  if len(filtered_proposals != filtered_repos) or filtered_proposals == []:
    logger.error(f"Filtering proposals failed. Exiting")
    return None

  # Step 3: Have the LLM justify and select the best candidate
  
  for attempt in range(max_attempts):
    proposal_eval_transcript = Transcript(log_filename=transcript_file)
    final_selection_prompt = tournament_prompts.construct_proposal_writing_prompt(candidate_ideas)
    proposal_eval_transcript.append(
      ContentChunk(final_selection_prompt, "user", tags=["advanced_backtrack_selection"])
    )
    
    llm_response = llm_utils.generate_completion(llm_name, proposal_eval_transcript, config)
    
    if not llm_response:
      logger.error("LLM failed to provide a response for final selection. Aborting.")
      continue

    proposal_eval_transcript.append(
      ContentChunk(llm_response, "model", tags=["advanced_backtrack_response"])
    )

  return None

def merge_ideas(
  llm_name: str, 
  transcript_name: str, 
  config: dict,
  new_idea_repo, 
  idea_cap: int,
  max_attempts: int = 3,
) -> None:
  for attempt in range(max_attempts):

    idea_dropping_transcript = Transcript(log_filename=transcript_name)
    idea_dropping_prompt = idea_select_utils.construct_idea_drop_prompt(new_idea_repo, idea_cap)
    idea_dropping_transcript.append(ContentChunk(idea_dropping_prompt, "user", tags=["drop_ideas"]))
    
    ideas_to_drop = llm_utils.generate_completion(llm_name, idea_dropping_transcript, config)

    if not ideas_to_drop:
      logger.error(f"LLM failed to produce ideas to drop on attempt {attempt + 1}. Moving on.")
      continue
    
    ideas_to_drop_list = idea_select_utils.parse_llm_idea_list(ideas_to_drop)
    if not ideas_to_drop_list:
      logger.error(f"Parsed output was not a valid list of idea IDs on attempt {attempt + 1}. Retrying.")
      continue  # Go to the next attempt in the loop
        
    ideas_to_keep = [idea for idea in new_idea_repo.ideas if idea.id not in ideas_to_drop_list]
    new_idea_repo.ideas = ideas_to_keep
    new_idea_repo.reindex_ideas()

    if len(new_idea_repo.ideas) <= idea_cap:
      logger.info(f"Successfully dropped ideas and met cap of {idea_cap} on attempt {attempt + 1}.")
      break  # Goal achieved, exit the loop
    else:
      logger.warning(f"Did not drop enough ideas. Current count: {len(new_idea_repo.ideas)}. Retrying...")

  if len(new_idea_repo.ideas) > idea_cap:
    logger.error(f"Failed to meet idea cap of {idea_cap} after {max_attempts} attempts. Current idea count is {len(new_idea_repo.ideas)}. Proceeding with extra ideas.")
