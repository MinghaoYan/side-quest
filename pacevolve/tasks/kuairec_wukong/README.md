# KuaRec Wukong Task

PACEvolve task for evolving a Wukong-style sequential recommender on the KuaRec dataset from Kuaishou.

## Task Goal

Maximize held-out next-item ranking quality on `kuairec` while keeping:

- Dataset fixed: KuaRec only
- Training epochs fixed: 8 epochs
- Training loop fixed: sampled-softmax training and full-catalog evaluation
- Runtime budget fixed: intended to fit roughly within 5 minutes on 1xA100

The evolvable surface is the sequence feature construction plus the Wukong-style user encoder inside [`src/train_wukong.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec_wukong/src/train_wukong.py).

## Why This Task Exists

`fuxi-linear` already contains the KuaRec preprocessing path and a strong sequential recommendation setup. `wukong-recommendation` contains a tabular interaction backbone. This task bridges the two:

- Convert sequential KuaRec histories into sparse and dense feature fields
- Encode those fields with a Wukong-inspired interaction stack
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

- The task code in [`src/train_wukong.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec_wukong/src/train_wukong.py) does not import `pandas`, `matplotlib`, or `seaborn`.
- Those extra packages are only needed because [`fuxi-linear/preprocess_kuairec_data.py`](/Users/minghao/PACE-RL/fuxi-linear/preprocess_kuairec_data.py) imports them.
- No extra package from `wukong-recommendation` is required because the benchmark copies the relevant modeling logic into the editable block.

### 1. Preprocess KuaRec once

From the repo root:

```bash
mkdir -p fuxi-linear/tmp
python fuxi-linear/preprocess_kuairec_data.py
```

The evaluator expects the processed file at:

`pacevolve/tasks/kuairec_wukong/data/sasrec_format.csv`

### 2. Run the evaluator directly

Syntax-only validation:

```bash
python pacevolve/tasks/kuairec_wukong/eval/evaluate_kuairec_wukong.py \
  --candidate_path pacevolve/tasks/kuairec_wukong/src/train_wukong.py \
  --dataset_csv pacevolve/tasks/kuairec_wukong/data/sasrec_format.csv \
  --syntax_only
```

Full training + evaluation:

```bash
python pacevolve/tasks/kuairec_wukong/eval/evaluate_kuairec_wukong.py \
  --candidate_path pacevolve/tasks/kuairec_wukong/src/train_wukong.py \
  --dataset_csv pacevolve/tasks/kuairec_wukong/data/sasrec_format.csv
```

## Baseline Size

The current baseline model in [`src/train_wukong.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec_wukong/src/train_wukong.py) has:

- `31,013,281` trainable parameters
- about `118.3 MB` of weights in FP32

That is small for an A100. The largest parameter blocks are the shared sparse embedding table, the Wukong factorization MLP outputs, and the final user projection head.

## Why 5 Minutes Is a Reasonable Estimate

The 5-minute target is an engineering estimate from the fixed scaffold rather than a paper-verified timing number. It is reasonable because:

- The baseline is now about `31.0M` parameters.
- Training uses one fixed next-item target per user for train and eval, not full autoregressive supervision over every position.
- Training uses sampled-softmax with `127` negatives instead of full-catalog loss.
- Evaluation is full-catalog, but `kuairec` is only about `9.96k` items after the standard remapping used here.
- Batch sizes are large (`1024` train, `2048` eval), which is comfortable for a model this small on an A100.
- The task caches parsed tensors to `sasrec_format.csv.kuairec_wukong_cache.pt`, so repeated evolutionary runs do not keep reparsing the CSV.

In other words, the expensive part is mostly dense matrix math over a small model and a modest item catalog, which is exactly the kind of workload an A100 handles well.

I have only smoke-tested the task on a synthetic dataset in this repo session, not run a real A100 timing pass on full KuaRec, so this should be read as a strong estimate rather than a confirmed benchmark number.

## Metrics

The candidate script prints a JSON payload with:

- `ndcg@10`
- `hr@10`
- `mrr`
- `combined_score`
- `wall_time_sec`
- `within_budget`

`combined_score` is the optimization target and is defined as:

`(ndcg@10 + hr@10 + mrr) / 3`

### What These Metrics Mean

- `ndcg@10`: Normalized Discounted Cumulative Gain at 10. It gives credit when the true next item is ranked near the top, with more reward for rank 1 than rank 10.
- `hr@10`: Hit Rate at 10. It is `1` if the true next item appears anywhere in the top-10 list and `0` otherwise, then averaged over users.
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
- The task now implements `parse_eval_metrics(...)` in [`eval_utils.py`](/Users/minghao/PACE-RL/pacevolve/tasks/kuairec_wukong/eval/eval_utils.py).
- The gym recorder path now propagates the full parsed metric dictionary into each candidate entry in `metrics.json`, not just `combined_score`.

That means each recorded candidate can retain fields like `ndcg@10`, `hr@10`, `mrr`, `wall_time_sec`, and `within_budget` alongside the aggregate score.

Candidates that exceed the wall-clock budget are treated as invalid.

## Files

- `config/config_1.yaml`: PACEvolve config
- `config/prompts.py`: mutation and idea prompts
- `src/train_wukong.py`: fixed trainer with editable model/feature block
- `eval/evaluate_kuairec_wukong.py`: CLI evaluator
- `eval/eval_utils.py`: PACEvolve integration helpers
