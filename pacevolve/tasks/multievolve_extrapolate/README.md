# MULTI-evolve Extrapolation Benchmark

This task implements the following from the MULTI-evolve paper as a public-data benchmark:

- train on singles + doubles
- test on unseen higher-order mutants
- optimize Pearson R and top-5% precision

It is designed to be reproducible from the public benchmark data release referenced by the MULTI-evolve repo.


## Benchmark structure

The task uses a representative `lite` subset by default to keep repeated evolutionary evaluation tractable. The task also ships a `full` manifest covering all 12 benchmark assays from the MULTI-evolve summary.

It supports two split protocols:

- `paper`: train on WT + singles + doubles, test on all qualifying 3+ mutants
- `released_code`: align split thresholds to the public MULTI-evolve benchmark notebook (`max_train_muts=3`, `min_test_muts=4`)

### Default split

For each dataset:

- Train/Test depends on `benchmark_protocol`
- `paper`: train wild type, singles, and doubles; test 3+ mutants
- `released_code`: train up to triple mutants; test 4+ mutants
- Filter: keep only multimutants whose constituent mutations appear as singles in the same dataset

This follows the paper-aligned low-order-to-high-order extrapolation setup and the preprocessing logic in the public MULTI-evolve repo.

## Metrics

Per dataset:

- `pearson_r`
- `precision_top5`

Final benchmark score:

- `combined_score = 0.7 * mean_pearson_r + 0.3 * mean_precision_top5`

## Evolvable surface

Your editable code lives in [multievolve_extrapolate_1.py](/Users/minghao/PACE-RL/pacevolve/tasks/multievolve_extrapolate/src/multievolve_extrapolate_1.py).

The intended search surface includes:

- mutation featurization
- pairwise interaction modeling
- regularization and calibration
- sample weighting
- lightweight ensembling

## Public data setup

### Option 1: Download from Zenodo

```bash
python pacevolve/tasks/multievolve_extrapolate/data/download_public_data.py --benchmark-level lite
python pacevolve/tasks/multievolve_extrapolate/data/prepare_public_benchmark.py --benchmark-level lite --benchmark-protocol paper
```

### Option 2: Copy from an existing local MULTI-evolve data folder

If you already downloaded the CSVs into `MULTI-evolve/data/benchmark/datasets`:

```bash
python pacevolve/tasks/multievolve_extrapolate/data/download_public_data.py \
  --benchmark-level lite \
  --source-dir /path/to/MULTI-evolve/data/benchmark/datasets
python pacevolve/tasks/multievolve_extrapolate/data/prepare_public_benchmark.py --benchmark-level lite --benchmark-protocol paper
```

Prepared artifacts are written under:

- `pacevolve/tasks/multievolve_extrapolate/data/raw`
- `pacevolve/tasks/multievolve_extrapolate/data/prepared/lite/paper`

## Running the evaluator

```bash
python pacevolve/tasks/multievolve_extrapolate/eval/evaluate_multievolve_extrapolate.py \
  --candidate_path pacevolve/tasks/multievolve_extrapolate/src/multievolve_extrapolate_1.py \
  --data_dir pacevolve/tasks/multievolve_extrapolate/data \
  --benchmark_level lite \
  --benchmark_protocol paper
```

For syntax-only validation:

```bash
python pacevolve/tasks/multievolve_extrapolate/eval/evaluate_multievolve_extrapolate.py \
  --candidate_path pacevolve/tasks/multievolve_extrapolate/src/multievolve_extrapolate_1.py \
  --data_dir pacevolve/tasks/multievolve_extrapolate/data \
  --benchmark_level lite \
  --benchmark_protocol paper \
  --syntax_only
```

## Relationship to the original MULTI-evolve repo

This task is based on the public benchmark design and split style from the local [MULTI-evolve](../../../../MULTI-evolve/README.md) clone, but the evaluator is intentionally lighter than their full training stack so that evolutionary search can run repeatedly without WandB sweeps or the full package environment.

## Running PACEvolve

There is also a top-level launcher at [run_multievolve.sh](/Users/minghao/PACE-RL/run_multievolve.sh).

Default behavior:

- runs `multievolve_extrapolate`
- uses the `lite` benchmark
- uses the `paper` split protocol
- uses `HYBRID_PKPO_GRPO` with `HYBRID_GRPO_VARIANT=dr_grpo`
- anneals `HYBRID_ALPHA` from `0.0` to `0.8` over `200` steps
- reserves a 16-GPU node as `4 train + 8 rollout + 4 eval`

If prepared benchmark files are missing, the script will:

- copy raw CSVs from `MULTI-evolve/data/benchmark/datasets` if present
- otherwise optionally download from Zenodo when `MULTIEVOLVE_AUTO_DOWNLOAD=1`

Example:

```bash
bash /Users/minghao/PACE-RL/run_multievolve.sh
```

Useful overrides:

```bash
MULTIEVOLVE_BENCHMARK_LEVEL=full \
MULTIEVOLVE_BENCHMARK_PROTOCOL=released_code \
MULTIEVOLVE_AUTO_DOWNLOAD=1 \
WANDB_PROJECT=my_project \
bash /Users/minghao/PACE-RL/run_multievolve.sh
```
