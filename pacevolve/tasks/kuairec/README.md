# KuaRec FuXi-Linear Task

PACEvolve task for evolving a KuaRec sequential recommender that is much closer to the released FuXi-Linear setup.

## Task Goal

Maximize held-out next-item ranking quality on `kuairec` while keeping:

- Dataset fixed: KuaRec only
- Training epochs fixed: 16 epochs
- Training loop fixed: sampled-softmax training and full-catalog evaluation
- Runtime budget fixed: relaxed wall-clock budget with a 2400-second eval timeout

The evolvable surface is the sequence encoder and scoring logic inside [`src/train_kuairec.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec/src/train_kuairec.py).

## Why This Task Exists

`fuxi-linear` already contains the KuaRec preprocessing path and the released paper configs. This task now tracks that family much more closely while keeping the fixed PACEvolve scaffold:

- Convert sequential KuaRec histories into item, timestamp, and position-aware token features
- Encode those tokens with a FuXi-Linear-aligned multi-channel sequence mixer
- Train against a fixed next-item objective with fixed epochs and fixed data

That makes the benchmark a clean architecture-and-feature-design problem instead of a hyperparameter search problem.

## Setup

### 0. Packages

For this task itself, the trainer and evaluator only need PyTorch plus the Python standard library.

Recommended environment:

- Python 3.9+
- PyTorch with CUDA support if you want the intended A100 runtime

Install the minimum task dependencies with:

```bash
pip install torch
```

If you want to run the original `fuxi-linear` preprocessing script unmodified, install its lightweight preprocessing dependencies too:

```bash
pip install pandas matplotlib seaborn
```

Notes:

- The task code in [`src/train_kuairec.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec/src/train_kuairec.py) does not import `pandas`, `matplotlib`, or `seaborn`.
- Those extra packages are only needed because [`fuxi-linear/preprocess_kuairec_data.py`](/Users/minghao/PACE-RL/fuxi-linear/preprocess_kuairec_data.py) imports them.
- No extra package beyond the task code itself is required because the benchmark now starts from a self-contained FuXi-linear-style baseline.

### 1. Preprocess KuaRec once

From the repo root:

```bash
mkdir -p fuxi-linear/tmp
python fuxi-linear/preprocess_kuairec_data.py
```

The evaluator expects the processed file at:

`pacevolve/tasks/kuairec/data/sasrec_format.csv`

### 2. Run the evaluator directly

Syntax-only validation:

```bash
python pacevolve/tasks/kuairec/eval/evaluate_kuairec.py \
  --candidate_path pacevolve/tasks/kuairec/src/train_kuairec.py \
  --dataset_csv pacevolve/tasks/kuairec/data/sasrec_format.csv \
  --syntax_only
```

Full training + evaluation:

```bash
python pacevolve/tasks/kuairec/eval/evaluate_kuairec.py \
  --candidate_path pacevolve/tasks/kuairec/src/train_kuairec.py \
  --dataset_csv pacevolve/tasks/kuairec/data/sasrec_format.csv
```

## Baseline Size

The current baseline model in [`src/train_kuairec.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec/src/train_kuairec.py) is now a FuXi-linear-style sequence encoder with:

- max sequence length `1024`
- embedding size `128`
- `4` FuXi-Linear-style blocks
- retention heads `4` with `dqk = 32` and `dv = 32`
- temporal heads `8`
- positional channel dimension `32`
- chunk size `128`
- dropout `0.5`
- L2-normalized dot-product scoring with temperature `0.05`

This keeps the task much closer to the paper and released `fuxi-linear/configs/kuairec/linear-4b-l1024-b64x2.gin`, while still preserving the fixed 16-epoch PACEvolve scaffold.

Using the current remapped KuaRec catalog size of roughly `9.96k` items, the baseline is about `2.46M` parameters, or about `9.4 MB` in FP32 weights.

## Runtime Note

This is no longer a 5-minute setup. The task now uses a much longer history length and a model much closer to the released FuXi-Linear KuaRec recipe, so the runtime budget has been relaxed accordingly.

- Sequence length is now `1024` instead of a short-sequence proxy.
- The model is closer to the released KuaRec configuration from `fuxi-linear`.
- Training still uses one fixed next-item target per user, so it remains cheaper than the paper's full autoregressive supervision.
- Evaluation is still full-catalog over roughly `9.96k` remapped items.
- The task caches parsed tensors to `sasrec_format.csv.kuairec_cache_v3.pt`, so repeated runs do not keep reparsing the CSV.

I have not run a full real-KuaRec timing pass in this repo session, so treat the timeout as an engineering budget rather than a benchmarked claim.

## Metrics

The candidate script prints a JSON payload with:

- `ndcg@10`
- `ndcg@50`
- `hr@10`
- `hr@50`
- `mrr`
- `combined_score`
- `wall_time_sec`
- `within_budget`
- `valid_run`
- `mean_target_tie_count`
- `max_target_tie_count`
- `frac_target_tie_gt1`
- `frac_target_tie_ge10`

`combined_score` is the optimization target and is defined as:

`(ndcg@10 + hr@10 + mrr) / 3`

Non-finite runs are treated as invalid:

- If training or evaluation produces non-finite tensors or loss, the script marks `valid_run=false`.
- PACEvolve then treats that candidate as invalid instead of allowing accidental reward through degenerate ranking behavior.

Task-local reward-hacking review is also enforced:

- The editable block must behave like a normal causal recommender and must not rely on future labels, future timestamps, hidden evaluator state, or reflection tricks.
- During PACEvolve eval, the KuaRec task runs a task-local LLM review over the candidate source before eval, and again with the eval payload after a successful run.
- KuaRec ranking metrics are also computed with tie-robust ranks, and the evaluator logs tie diagnostics such as `mean_target_tie_count` and `frac_target_tie_ge10`.
- Suspicious high-score runs with excessive target-score ties are rejected as task-local metric hacking, even if the candidate source itself looks superficially normal.
- If these task-local checks reject the run, the normal KuaRec eval-retry loop gets another repair attempt.
- These checks are local to the KuaRec task and do not change the workflow for other tasks.

### What These Metrics Mean

- `ndcg@10`: Normalized Discounted Cumulative Gain at 10. It gives credit when the true next item is ranked near the top, with more reward for rank 1 than rank 10.
- `ndcg@50`: The same ranking-quality metric over the top-50 list.
- `hr@10`: Hit Rate at 10. It is `1` if the true next item appears anywhere in the top-10 list and `0` otherwise, then averaged over users.
- `hr@50`: The same hit-rate metric over the top-50 list.
- `mrr`: Mean Reciprocal Rank. It uses `1 / rank` of the true next item, so rank 1 gives `1.0`, rank 2 gives `0.5`, rank 10 gives `0.1`, and lower ranks contribute less.

### Is The Arithmetic Mean Reasonable?

Yes, it is a reasonable simple aggregate here because all three are bounded ranking metrics on roughly the same numeric scale:

- `ndcg@10` is in `[0, 1]`
- `hr@10` is in `[0, 1]`
- `mrr` is in `(0, 1]`

They are not identical in behavior, though:

- `hr@10` is the coarsest metric and only checks whether the target entered the top-10.
- `ndcg@10` and `mrr` care more about exact position near the top.
- `mrr` is usually numerically lower than `hr@10` because it penalizes lower ranks more sharply.

So the arithmetic mean is not a theoretically perfect calibration, but it is a practical summary because it combines:

- coarse retrieval success (`hr@10`)
- top-heavy ranking quality (`ndcg@10`)
- exact-rank sensitivity (`mrr`)

### Does `metrics.json` Store The Submetrics Separately?

Yes, for PACEvolve gym runs it now does.

- The evaluator already emits every submetric in the final `Candidate: {...}` JSON payload.
- The task now implements `parse_eval_metrics(...)` in [`eval_utils.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec/eval/eval_utils.py).
- The gym recorder path now propagates the full parsed metric dictionary into each candidate entry in `metrics.json`, not just `combined_score`.

That means each recorded candidate can retain fields like `ndcg@10`, `hr@10`, `mrr`, `wall_time_sec`, `within_budget`, and the tie diagnostics alongside the aggregate score.

Candidates that exceed the wall-clock budget are treated as invalid.

## Files

- `config/config_1.yaml`: PACEvolve config
- `config/prompts.py`: mutation and idea prompts
- `src/train_kuairec.py`: fixed trainer with editable model/feature block
- `eval/evaluate_kuairec.py`: CLI evaluator
- `eval/eval_utils.py`: PACEvolve integration helpers
