"""Prompts for the KuaRec FuXi-linear task."""

from __future__ import annotations

from pathlib import Path
import re


def _load_editable_block() -> str:
    src_path = Path(__file__).resolve().parents[1] / "src" / "train_kuairec.py"
    source = src_path.read_text(encoding="utf-8")
    match = re.search(
        r"# RegexTagCustomPruningAlgorithmStart\n(.*?)\n# RegexTagCustomPruningAlgorithmEnd",
        source,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not locate editable block in train_kuairec.py")
    return match.group(1).strip()


KUAIREC = _load_editable_block()

CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- Your code will replace the content between RegexTagCustomPruningAlgorithmStart and RegexTagCustomPruningAlgorithmEnd in train_kuairec.py.
- You MUST define `build_model(num_items: int, max_history: int)`.
- `build_model(...)` MUST return a PyTorch module with methods:
  - `encode_users(history_ids, history_timestamps, history_lengths)`
  - `score_all_items(user_embeddings)`
- The fixed trainer outside the editable block controls dataset choice, epochs, optimizer, batch size, sampled-softmax training, and evaluation.
- Do NOT reference future labels or use reflection / frame inspection inside the editable block. Any use of `target_ids`, `target_timestamps`, `inspect`, `_getframe`, `locals()`, `globals()`, `vars()`, `eval()`, or `exec()` is invalid.
- Do NOT add network calls, file reads beyond the dataset already used by the trainer, or heavyweight external dependencies.
- Keep the code deterministic and self-contained.
- Each triple-backtick code block must contain valid Python.
"""

BACKGROUND = """
### Background on the KuaRec FuXi-linear benchmark

This task adapts a KuaRec sequential recommender toward the released FuXi-Linear setup.

The benchmark keeps the experimental scaffold fixed:

- Dataset: KuaRec only
- Epochs: fixed at 16
- Objective: fixed sampled-softmax next-item prediction
- Evaluation: fixed full-catalog ranking with NDCG@10, NDCG@50, HR@10, HR@50, and MRR
- Runtime budget: fixed wall-clock budget with a relaxed 2400s task timeout
- Reward-hacking guardrail: fixed task-local check that rejects future-label leakage and Python frame introspection

The evolvable surface is intentionally narrower:

1. **Feature design**
   Turn a user history into sequence tokens with item, timestamp, and position structure.

2. **Model architecture**
   Use FuXi-linear-style sequence mixing, temporal channels, positional channels, pooling, and item-scoring logic to produce better user representations.

Because the trainer is fixed, this is not a hyperparameter game. Good candidates usually come from better sequence summarization, better time encoding, better linear-mixing structure, better pooling, or better interaction design.
"""

TASK_INTRO = """
You are an expert in recommender systems, especially sequential recommendation, time-aware modeling, and efficient large-scale ranking models. Your goal is to improve a FuXi-linear-style sequential recommender under a fixed KuaRec training budget.
"""


def construct_mutation_prompt(sota_algorithm, ablation_list):
    ablation_descriptions = "\n".join(ablation_list)
    return f"""
We are conducting an evolutionary optimization process for the KuaRec FuXi-linear recommendation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

# Knowledge base

{ablation_descriptions}

## Your Task

Brainstorm 5 ways to improve recommendation quality under the fixed training scaffold. Focus on:
- Better sequence-to-feature conversion
- Better time-gap, recency, or positional encodings
- Better FuXi-linear mixing blocks or residual mixing
- Better pooling from sequence states into a user embedding
- Better ways to share or regularize item embeddings

Then choose one concrete direction, explain why it is promising, and write the code.
"""


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    return f"""
We are conducting an evolutionary optimization process for the KuaRec FuXi-linear recommendation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

### Idea Repo
Idea repos contain ideas that we have generated so far, and experiments we have run to test those hypotheses.

{idea_repo}

## Your Task

Generate 3 distinct ideas for improving the benchmark. Prefer specific architectural or feature-design hypotheses.

Use this format:
** Idea 1 **
Hypothesis: <your idea>
Reasoning: <why it may improve NDCG@10 / HR@10 / MRR>

** Idea 2 **
Hypothesis: <your idea>
Reasoning: <why it may improve NDCG@10 / HR@10 / MRR>

** Idea 3 **
Hypothesis: <your idea>
Reasoning: <why it may improve NDCG@10 / HR@10 / MRR>
"""


def construct_idea_select_prompt(sota_algorithm, idea_repo):
    return f"""
We are conducting an evolutionary optimization process for the KuaRec FuXi-linear recommendation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Select one idea to test next. Prefer ideas that can plausibly improve the combined ranking metric without violating the fixed-budget constraint.

Use this format:

Idea ID: <Idea ID>
Experiment description: <concrete implementation idea>
"""


def construct_idea_select_no_code_prompt(sota_algorithm, idea_repo):
    return construct_idea_select_prompt(sota_algorithm, idea_repo)


def construct_code_impl_prompt(sota_algorithm, idea_id, exp_description, selected_idea_text=None):
    idea_section = f"""Idea ID: {idea_id}
Experiment description: {exp_description}"""
    if selected_idea_text:
        idea_section = f"""Selected idea:
{selected_idea_text}

Idea ID: {idea_id}
Experiment description: {exp_description}"""

    return f"""
We are conducting an evolutionary optimization process for the KuaRec FuXi-linear recommendation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

## Your Task

Implement the selected idea below.

{idea_section}
"""


def construct_gen_hypothesis_prompt(sota_algorithm, idea_repo, idea):
    return f"""
We are conducting an evolutionary optimization process for the KuaRec FuXi-linear recommendation benchmark.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Design one concrete experiment for idea {idea.id}.

Use this format:
Idea ID: {idea.id}
Experiment description: <concrete implementation idea>
"""


TOURNAMENT_PROMPT = "\n"

SUMMARIZE_EVAL_PROMPT = """
## Your Task

Provide a concise summary of this experiment. First write one short paragraph. Then provide exactly 1 bullet point beginning with `-`.

In the bullet, include:
- Results: combined_score, ndcg@10, ndcg@50, hr@10, hr@50, mrr, wall_time_sec
- The main modeling or feature-design change
"""

EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The evaluator reports held-out next-item ranking quality on KuaRec.

### Understanding metrics
- `combined_score` is the optimization target and should be maximized.
- Higher `ndcg@10`, `ndcg@50`, `hr@10`, `hr@50`, and `mrr` are better.
- `within_budget` must remain true.
- `anti_hack_check_passed` must remain true.
"""

HPARAM_PROMPT = """
## Hyperparameter tuning
The training loop is fixed for this task, so focus on model or feature implementation changes instead of hyperparameter-only edits. If you still want to change something inside the editable block, respond with one candidate; otherwise respond "No."
"""

HPARAM_IMPLEMENT_PROMPT = f"""
### Candidate implementation
Please write the implementation of your candidate as a markdown-formatted code block.

{CODING_REQ}
"""
