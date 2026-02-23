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

import re
from dataclasses import dataclass, field
from typing import Literal
from llm_utils import ContentChunk
import llm_utils
from copy import deepcopy
import logging
import numpy as np
import ast

import crossover_utils
AdaptiveCrossoverScheduler = crossover_utils.AdaptiveCrossoverScheduler
logger = logging.getLogger("controller")

# --- DataClass Definitions ---


@dataclass
class Idea:
    """Represents a single idea with its history and status."""
    id: int
    description: str
    exp_history: list[str] = field(default_factory=list)
    exp_count: int = 0
    status: Literal['new', 'in_progress', 'validated', 'invalidated'] = 'new'

@dataclass
class IdeaRepo:
    """A repository to store and manage a list of ideas."""
    ideas: list[Idea] = field(default_factory=list)
    sota: str = None

    def get_next_id(self):
        """Returns the next available unique ID."""
        if not self.ideas:
            return 1
        return max(idea.id for idea in self.ideas) + 1

    def find_idea_by_id(self, idea_id: int):
        """Finds an idea in the repository by its ID."""
        for idea in self.ideas:
            if idea.id == idea_id:
                return idea
        return None

    def __str__(self):
        """String representation for printing the idea database."""
        if not self.ideas:
            return "No ideas in the database."
        return "\n".join(
            # A main f-string for the static info of each idea
            f"Idea ID: {i.id}\n"
            f"Idea Description: {i.description}\n"
            f"Experiment Count: {i.exp_count}\n" +
            f"Experiment History:\n" +
            # Join the history items, formatting each one.
            # This gracefully handles an empty list by producing an empty string.
            "\n".join(f"\t {exp}" for exp in i.exp_history)
            for i in self.ideas
        )
    
    def get_lowest_nmse(self, exp_history: list[str]) -> float | None:
        """Parses a list of experiment history strings and returns the lowest NMSE value found."""
        nmse_values = []
        pattern = re.compile(r'nmse: (\d+\.?\d*e?-?\d*)')
        for exp_str in exp_history:
            match = pattern.search(exp_str)
            if match:
                nmse_values.append(float(match.group(1)))
        
        if not nmse_values:
            return None
        
        return min(nmse_values)

    def get_next_idea_id_ucb(self):
        pass
    
    def reindex_ideas(self):
        new_id = 1
        for idea in self.ideas:
          idea.id = new_id
          new_id += 1

@dataclass
class IdeaRepoDatabase:
    num_islands: int
    target_score: float # Added: target_score (r)
    metric_direction: str
    idea_repos: list[list[IdeaRepo]] = None
    best_scores_history: list[list[float]] = None # Renamed for clarity
    
    # Internal scheduler, initialized in __post_init__
    scheduler: AdaptiveCrossoverScheduler = field(init=False, repr=False)

    def __post_init__(self):
        if self.idea_repos is None:
            self.idea_repos = [[] for _ in range(self.num_islands)]
        if self.best_scores_history is None:
            self.best_scores_history = [[] for _ in range(self.num_islands)]
        
        # Initialize the adaptive scheduler
        # Multiply score by -1 if the target is max.
        if self.metric_direction == "max":
            reverse = -1
        elif self.metric_direction == "min":
            reverse = 1
        self.scheduler = AdaptiveCrossoverScheduler(
            num_islands=self.num_islands,
            target_score=self.target_score,
            reverse=reverse
        )

    def get_best_idea_repo(self, island_id: int) -> IdeaRepo:
        """
        Convenience function to get the IdeaRepo object associated
        with the best score for a given island.
        """
        if not (0 <= island_id < self.num_islands):
            raise ValueError(f"Invalid island_id: {island_id}")
            
        island_scores = self.best_scores_history[island_id]
        if not island_scores:
            raise ValueError(f"No score history for island_id: {island_id}")
        best_idx = int(np.argmin(island_scores))
        
        return self.idea_repos[island_id][best_idx]

def construct_idea_classification_prompt(idea_repo, hypothesis):
    prompt =f"""
Below is the database of ideas we have explored:
{idea_repo}

Here is the hypothesis to be classified:
{hypothesis}

Your job is to classify the newly generated idea and merge into the idea repo.
If you believe the following idea is similar or identical to one of the ideas in the database, you should respond in the following format by identifying the ID of the similar idea. For instance, if your idea leverage a similar math or computer science concepts, but require different implementations or hyperparameter tuning, then they should be grouped under the same idea.
If your idea is similar but not identical, provide an updated description that combines both your idea and the original idea and keep the description concise. Otherwise, you can reuse the same idea description in the updated description section.
If your idea is a combination of two existing but very different ideas, then your idea should count as a new idea. 
If you find it hard to articulate the new hypothesis together with the original idea in TWO SENTENCES after merging, you should classify the hypothesis as a new idea. Your updated description / idea description should be NO MORE THAN TWO SENTENCES. But you should also AVOID using overly broad terms and BE SPECIFIC about the ideas while keeping the idea description concise.

Idea Exists: True
Idea ID: <Idea ID>
Updated description: <Updated description here>

If you believe your idea is new and orthogonal to existing ideas, respond in the following format:
Idea Exists: False
Idea description: <Provide your hypothesis here>

"""
    return prompt


def construct_idea_summarization_prompt(idea):
    prompt =f"""
Below is the idea we want to summarize:
{idea.description}

Your job is to summarize and shorten the idea description into NO MORE THAN TWO SENTENCES. You should AVOID using overly broad terms and BE SPECIFIC about the ideas while keeping the idea description concise. Your ideas will later be used to generate concrete experiment hypothesis.
"""
    return prompt

def construct_history_summarization_prompt(idea):
    prompt =f"""
Below is the idea we want to summarize:
{idea.exp_history}

Your job is to summarize and shorten the experiment history into one bullet point. You should keep the result of the best trial so far, followed by summarizing what ideas work and what do not work. 

Format like this:

- Results: nmse: xxx, oos_nmse: xxx. <Summary of what ideas work and what do not work>

"""
    return prompt

def construct_idea_drop_prompt(idea_repo, idea_cap):
    prompt =f"""
Below is the database of ideas we have explored:
{idea_repo}

We want to cap the number of ideas under-consideration at {idea_cap}, therefore, we need to drop some ideas.

Your job is to evaluate which ideas should we job to reduce the number of ideas under consideration to {idea_cap}.

Your criteria should be as follows:

1. If thorough experiment results from the experiment history of this and other ideas clearly invalidate this idea?
2. If there is any potential in a breakthrough in performance if we try more hypothesis based on this idea?
3. Has this idea been thoroughly investigated (indicated by experiment count) and the performance clearly underperform the current best results in the experiment history?
4. With the amount of experiments performed on this idea, is it possible to reduce the gap between the current best of class performance of this idea and the target metric?
5. You should priorize dropping ideas that are either old, lack the potential to improve, or has been explored extensively but still do not see breakthrough improvements.

You should output the idea(s) to drop in a Python list format (e.g. if you want to drop idea i, you should output [i])

Format like this:
Dropping Ideas: <list of idea(s) to drop>
"""
    return prompt


def construct_idea_filter_prompt(ablation_list, proposals):
    prompt =f"""
Below is the database of experiment hypothesis we have explored:
{ablation_list}

And below are the proposals we want to evaluate:
{proposals}

We want to filter out repetitive experiment hypothesis so that we do not waste compute on hypothesis we already know. Your job is to identify proposals that are either 1. identical to one of the ideas already explored or 2. have been tested enough similar hypothesis that you have high confidence in inferring the results of the hypothesis.

Your criteria should be as follows:

1. If thorough experiment results from the experiment history invalidate this idea?
2. If there are enough experiment results from the experiment history that we can confidently infer the result of this experiment?
3. Was there already an experiment about an identical experiment hypothesis? If so, running it again would yield the same result and we should not be running it.

You should output the idea(s) to drop in a Python list format (e.g. if you want to drop idea i, you should output [i]. if you believe no ideas should be filtered out, then output [])

Format like this:
Dropping Ideas: <list of idea(s) to drop>
"""
    return prompt


def construct_idea_repetition_detection_prompt(ablation_list, proposals):
    prompt =f"""
Below is the database of experiment hypothesis we have explored:
{ablation_list}

And below is the proposal we want to evaluate:
{proposals}

We want to filter out repetitive experiment hypothesis so that we do not waste compute on hypothesis we already know. Your job is to identify proposals that are either identical to one of the ideas already explored or have been tested enough similar hypothesis that you have high confidence in inferring the results of the hypothesis.

Your criteria should be as follows:

1. If thorough experiment results from the experiment history invalidate this idea?
2. If there are enough experiment results from the experiment history that we can confidently infer the result of this experiment?
3. Was there already an experiment about an identical experiment hypothesis? If so, running it again would yield the same result and we should not be running it.

You should output True or False after "Dropping Ideas: " once you finish your reasoning.

Format like this:
Reasoning: <Your reasoning goes here>
Dropping Ideas: <True if this hypothesis is repetitive and we should drop it, False otherwise>
"""
    return prompt


def parse_llm_idea_list(llm_output: str) -> list | None:
    """
    Parses a string from an LLM to extract and convert a Python list.

    Uses a regex to find the list string and then ast.literal_eval()
    to safely convert it.
    
    Parameters:
        llm_output (str): The raw output string from the LLM.

    Returns:
        list | None: A Python list of the extracted idea IDs, or None if parsing fails.
    """
    # Define a regex pattern to find the list
    # It looks for "Dropping Ideas: " followed by a list structure.
    # The [^\[\]]*? is a non-greedy match for any characters that aren't brackets.
    # The \[\s*(\d+(?:\s*,\s*\d+)*)\s*\] is a more robust way to capture only numbers.
    # The provided example is a simple list of numbers, so we'll use a simplified pattern
    # that captures content between the first [ and last ]
    pattern = re.compile(r'Dropping Ideas:\s*(\[.*?\])', re.DOTALL)
    
    match = pattern.search(llm_output)
    
    if not match:
        logging.error("Failed to find 'Dropping Ideas:' pattern in LLM output.")
        return None
        
    # Extract the string representation of the list from the matched group
    list_str = match.group(1)
    
    try:
        # Safely convert the string to a Python list
        ideas_to_drop = ast.literal_eval(list_str)
        
        # Verify the result is actually a list and contains integers
        if isinstance(ideas_to_drop, list) and all(isinstance(x, int) for x in ideas_to_drop):
            return ideas_to_drop
        else:
            logging.error(f"ast.literal_eval did not return a list of integers: {ideas_to_drop}")
            return None
            
    except (ValueError, SyntaxError) as e:
        logging.error(f"Failed to safely evaluate list string '{list_str}': {e}")
        return None


# --- Core Functions ---

def parse_hypothesis(llm_response: str) -> list[str]:
    """
    Parses hypotheses from a formatted LLM response string.
    Now includes a basic check for empty input.
    """
    if not llm_response or not llm_response.strip():
        logger.warning("parse_hypothesis received an empty or null response.")
        return []
        
    pattern = re.compile(r"Hypothesis:\s*(.*?)\s*Reasoning:", re.DOTALL)
    hypotheses = pattern.findall(llm_response)
    return [h.strip() for h in hypotheses]


def parse_selected_idea(llm_response_text: str) -> tuple:
    """
    Parses the selected idea ID and experiment description from an LLM response string.

    Args:
        llm_response_text: The text output from the language model.

    Returns:
        A tuple containing the idea ID and the experiment description.
        Returns (None, None) if parsing fails or the input is empty.
    """
    if not llm_response_text or not llm_response_text.strip():
        logger.warning("parse_selected_idea received an empty or null response.")
        return None, None

    # This pattern looks for "Idea ID:" and "Experiment description:" headers.
    # - `re.DOTALL` is crucial because it allows `.` to match newline characters,
    #   capturing a multi-line experiment description.
    # - `(.*?)` is a non-greedy capture for the ID.
    # - `\s*` matches any whitespace including newlines between the two fields.
    pattern = re.compile(
        r"Idea ID:\s*(.*?)\s*Experiment description:\s*(.*)", re.DOTALL
    )

    match = pattern.search(llm_response_text)

    if match:
        # group(1) captures the text for Idea ID
        # group(2) captures the text for Experiment description
        idea_id = match.group(1).strip()
        exp_description = match.group(2).strip()

        # Ensure that the captured strings are not empty
        if idea_id and exp_description:
            return idea_id, exp_description
        else:
            logger.error(f"Regex matched but captured empty ID or description from response: '{llm_response_text}'")
            return None, None
    else:
        logger.error(f"Failed to parse idea and description from LLM response: '{llm_response_text}'")
        return None, None
    

def parse_repetition_detection(llm_response: str):
    """
    Parses the boolean value from an LLM response for repetition detection.
    This version explicitly checks that the parsed value is 'true' or 'false'.

    Args:
        llm_response: The text output from the language model, which is expected
                      to contain a line like "Dropping Ideas: <True/False>".

    Returns:
        True if the response indicates the idea should be dropped.
        False if the response indicates the idea should not be dropped.
        None if parsing fails, the input is empty, or the value is not 'True' or 'False'.
    """
    if not llm_response or not llm_response.strip():
        logger.warning("parse_repetition_detection received an empty or null response.")
        return None

    # This more robust pattern finds the line and captures the word that follows.
    # It doesn't restrict the capture to only "True" or "False" at the regex level.
    pattern = re.compile(r"Dropping Ideas:\s*(\w+)", re.IGNORECASE)

    match = pattern.search(llm_response)

    if match:
        # We capture the value first...
        value_string = match.group(1).lower()
        
        # ...then explicitly check if it's one of the valid boolean strings.
        if value_string == 'true':
            return True
        elif value_string == 'false':
            return False
        else:
            # This handles cases like "Dropping Ideas: Maybe" or "Dropping Ideas: Unsure"
            logger.error(
                f"Found 'Dropping Ideas:' but the value was not 'True' or 'False'. "
                f"Received: '{match.group(1)}'"
            )
            return None
    else:
        logger.error(
            f"Failed to find the 'Dropping Ideas: <value>' pattern in LLM response: "
            f"'{llm_response}'"
        )
        return None


def update_idea_repo(llm_classification_response: str, idea_repo: IdeaRepo) -> bool:
    """
    Updates the idea database based on a classification response.
    This function is now more robust against parsing errors and returns a
    status boolean instead of printing errors directly.
    """
    if not llm_classification_response:
        logger.warning("update_idea_repo received an empty response.")
        return False

    try:
        if "Idea Exists: True" in llm_classification_response:
            idea_id_match = re.search(r"Idea ID:\s*(\d+)", llm_classification_response)
            desc_match = re.search(r"Updated description:\s*(.*)", llm_classification_response, re.DOTALL)

            if not idea_id_match or not desc_match:
                logger.warning("LLM claimed idea exists, but failed to provide ID or description.")
                return False

            idea_id = int(idea_id_match.group(1))
            updated_desc = desc_match.group(1).strip()
            
            idea_to_update = idea_repo.find_idea_by_id(idea_id)
            if idea_to_update:
                idea_to_update.description = updated_desc
                logger.info(f"Merged new concept into Idea ID {idea_id}.")
                return True
            else:
                logger.error(f"Could not find Idea ID {idea_id} to update, though LLM suggested it.")
                return False

        elif "Idea Exists: False" in llm_classification_response:
            desc_match = re.search(r"Idea description:\s*(.*)", llm_classification_response, re.DOTALL)
            
            if not desc_match:
                logger.warning("LLM claimed idea is new, but failed to provide a description.")
                return False

            new_desc = desc_match.group(1).strip()
            new_id = idea_repo.get_next_id()
            idea_repo.ideas.append(Idea(id=new_id, description=new_desc))
            logger.info(f"Added new Idea ID {new_id}: '{new_desc}'")
            return True
        else:
            logger.warning(f"Unrecognized classification format in LLM response:\n{llm_classification_response}")
            return False
            
    except (AttributeError, ValueError, TypeError) as e:
        logger.error(f"Critical error while parsing LLM classification response: {e}")
        return False

def extract_idea_id(llm_response_text: str) -> int:
  """
  Extracts the integer value following "Idea ID:" from a response text.

  Args:
    llm_response_text: The text generated by the LLM.

  Returns:
    The extracted idea ID as an integer.

  Raises:
    ValueError: If the "Idea ID:" pattern is not found in the text.
  """
  # This regex looks for the literal string "Idea ID:", followed by
  # optional whitespace (\s*), and then captures one or more digits (\d+).
  pattern = r"Idea ID:\s*(\d+)"
  
  match = re.search(pattern, llm_response_text)
  
  if match:
    # match.group(1) returns the part of the string inside the parentheses
    # in the pattern, which is our number.
    idea_id_str = match.group(1)
    return int(idea_id_str)
  else:
    return None


def scratch_pad(idea_repo: IdeaRepo, llm_name: str, transcript: list[ContentChunk], config: dict, idea_gen_prompt: str) -> None:
    """
    Orchestrates idea generation and classification with robust retry logic.
    """
    # --- Configuration for the retry loop ---
    max_attempts = 3
    
    # 1. Generate new ideas with a retry loop
    logger.info("--- Phase 1: Generating new ideas (with retry logic) ---")
    num_attempts = 0
    new_hypotheses = []
    recovery_prompt = None
    
    generation_transcript = deepcopy(transcript)
    while num_attempts < max_attempts:
        num_attempts += 1
        logger.info(f"Idea generation attempt {num_attempts}/{max_attempts}...")
        
        if recovery_prompt:
            generation_transcript.append(ContentChunk(recovery_prompt, "user", tags=["recovery_prompt"]))
        else:
            generation_transcript.append(ContentChunk(idea_gen_prompt, "user", tags=["idea_generation_prompt"]))

        try:
            generation_response = llm_utils.generate_completion(llm_name, generation_transcript, config)
            generation_transcript.append(ContentChunk(generation_response, "model", tags=["idea_generation_response"]))
            
            new_hypotheses = parse_hypothesis(generation_response)

            if new_hypotheses:
                logger.info(f"Successfully generated {len(new_hypotheses)} new hypotheses.")
                break # Success, exit the loop
            else:
                logger.warning(f"Attempt {num_attempts}: LLM response did not contain valid hypotheses.")
                recovery_prompt = "Your last response did not follow the required format. Please follow the format guideline closely."
        
        except Exception as e:
            logger.error(f"scratch_pad: An exception occurred during LLM call on attempt {num_attempts}: {e}")
            recovery_prompt = "An error occurred. Please try generating the response again."
            continue

    if not new_hypotheses:
        logger.error("Failed to generate new hypotheses after multiple attempts. Aborting scratch pad cycle.")
        return None

    # 2. Classify each new hypothesis and update the database
    logger.info(f"\n--- Phase 2: Classifying and updating database ---, there are {len(new_hypotheses)} hypotheses to be processed")
    successful_classifications = 0
    for hypo in new_hypotheses:
        logger.info(f"Classifying hypothesis: '{hypo[:80]}...'")
        classification_transcript = deepcopy(transcript)
        prompt_text = construct_idea_classification_prompt(idea_repo, hypo)
        classification_transcript.append(
            ContentChunk(prompt_text, "user", tags=["idea_classification_prompt"])
        )
        
        try:
            classification_response = llm_utils.generate_completion(llm_name, classification_transcript, config)
            if update_idea_repo(classification_response, idea_repo):
                successful_classifications += 1
        except Exception as e:
            logger.error(f"An exception occurred during LLM call for classification: {e}")
            continue

    logger.info(f"Finished classification. Successfully updated repo for {successful_classifications}/{len(new_hypotheses)} hypotheses.")
    return 1


def summarize(idea, llm_name, config, idea_transcript, history_transcript):
    max_attempts = 3
    num_attempts = 0
    while num_attempts < max_attempts:
        num_attempts += 1
        idea_summarization_prompt = construct_idea_summarization_prompt(idea)
        idea_transcript.append(
            ContentChunk(idea_summarization_prompt, "user", tags=["idea_summarization_prompt"])
        )
        try:
            generation_response = llm_utils.generate_completion(llm_name, idea_transcript, config)
            if not generation_response:
                logger.error("summarize: LLM call did not produce any results")
                continue
            idea.description = generation_response
        except Exception as e:
            logger.error(f"summarize: An exception occurred during LLM call on attempt {num_attempts}: {e}")
            # recovery_prompt = "An error occurred. Please try generating the response again."
            continue


        history_summarization_prompt = construct_history_summarization_prompt(idea)
        history_transcript.append(
            ContentChunk(history_summarization_prompt, "user", tags=["idea_summarization_prompt"])
        )
        try:
            generation_response = llm_utils.generate_completion(llm_name, history_transcript, config)
            if not generation_response:
                logger.error("summarize: LLM call did not produce any results")
                continue
            idea.exp_history = [generation_response]
        except Exception as e:
            logger.error(f"summarize: An exception occurred during LLM call on attempt {num_attempts}: {e}")
            # recovery_prompt = "An error occurred. Please try generating the response again."
            continue
        

def get_lowest_nmse(exp_history: list[str]) -> float | None:
    """
    Parses a list of experiment history strings and returns the lowest NMSE value found.

    Parameters:
        exp_history (list[str]): A list of strings, each representing an experiment's history.

    Returns:
        float | None: The lowest NMSE value or None if no NMSE is found.
    """
    nmse_values = []
    # Regex to find 'nmse: ' followed by a number
    pattern = re.compile(r'nmse: (\d+\.?\d*e?-?\d*)')
    for exp_str in exp_history:
        match = pattern.search(exp_str)
        if match:
            nmse_values.append(float(match.group(1)))
    
    if not nmse_values:
        return None
    
    return min(nmse_values)



def sample_power_law(n, alpha=1.5):
    """
    Samples a single number from the range [0, n-1] according to a 
    power-law distribution.

    Args:
        n (int): The upper bound of the range [0, n-1]. Must be a positive integer.
        alpha (float): The exponent of the power law. A higher alpha creates a
                     stronger bias towards sampling smaller numbers. Default is 1.5.

    Returns:
        int: A single integer sampled from the specified distribution.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Create the indices [0, 1, 2, ..., n-1]
    indices = np.arange(n)
    
    # Calculate the probability weights for each index.
    # We use (index + 1) to avoid division by zero when the index is 0.
    weights = (indices + 1)**(-alpha)
    
    # Normalize the weights so they sum to 1, creating a valid probability distribution.
    probabilities = weights / np.sum(weights)
    
    # Use numpy's random.choice to sample one number based on the calculated probabilities.
    return np.random.choice(indices, p=probabilities)