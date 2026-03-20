"""Public-data benchmark utilities for MULTI-evolve extrapolation."""

from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


MANIFEST_FILENAME = "benchmark_manifest.json"
RAW_DIRNAME = "raw"
PREPARED_DIRNAME = "prepared"
DEFAULT_BENCHMARK_PROTOCOL = "paper"


def _task_data_dir_from_file() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _resolve_protocol_bounds(benchmark_protocol: str) -> dict[str, Any]:
    if benchmark_protocol == "paper":
        return {
            "max_train_mutations": 2,
            "min_test_mutations": 3,
            "max_test_mutations": None,
        }
    if benchmark_protocol == "released_code":
        return {
            "max_train_mutations": 3,
            "min_test_mutations": 4,
            "max_test_mutations": None,
        }
    raise ValueError(
        f"Unknown benchmark protocol '{benchmark_protocol}'. Expected one of ['paper', 'released_code']."
    )


def load_manifest(data_dir: Optional[Path] = None, benchmark_level: str = "lite") -> list[dict[str, Any]]:
    if data_dir is None:
        data_dir = _task_data_dir_from_file()
    manifest_path = Path(data_dir) / MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload[benchmark_level])


def mutation_tokens(mutation: str) -> list[str]:
    text = str(mutation).strip()
    if not text or text == "WT" or text.lower() == "nan":
        return []
    return [token for token in text.replace(":", "/").split("/") if token and token != "WT"]


def mutation_count(mutation: str) -> int:
    return len(mutation_tokens(mutation))


def _all_component_singles_exist(tokens: list[str], single_set: set[str]) -> bool:
    return all(token in single_set for token in tokens)


def _load_dataset_csv(raw_dir: Path, entry: dict[str, Any]) -> pd.DataFrame:
    path = raw_dir / entry["DMS_filename"]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw benchmark dataset: {path}")

    df = pd.read_csv(path)
    expected = {"mutant", "DMS_score"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Dataset {entry['DMS_filename']} is missing required columns {sorted(expected)}."
        )
    if "DMS_score_bin" not in df.columns:
        df["DMS_score_bin"] = np.nan

    df = df[["mutant", "DMS_score", "DMS_score_bin"]].copy()
    df["mutant"] = df["mutant"].astype(str).str.replace(":", "/", regex=False)
    df["num_mutations"] = df["mutant"].map(mutation_count)
    df["mutation_tokens"] = df["mutant"].map(mutation_tokens)
    single_set = set(df.loc[df["num_mutations"] == 1, "mutant"].tolist())
    df["components_in_single_pool"] = df["mutation_tokens"].map(
        lambda tokens: _all_component_singles_exist(tokens, single_set)
    )
    return df


def prepare_single_dataset(
    data_dir: Path,
    entry: dict[str, Any],
    prepared_dir: Path,
    force: bool = False,
    benchmark_protocol: str = DEFAULT_BENCHMARK_PROTOCOL,
) -> dict[str, Any]:
    bounds = _resolve_protocol_bounds(benchmark_protocol)
    max_train_mutations = int(bounds["max_train_mutations"])
    min_test_mutations = int(bounds["min_test_mutations"])
    max_test_mutations = bounds["max_test_mutations"]
    prepared_path = prepared_dir / f"{entry['DMS_id']}.pkl"
    if prepared_path.exists() and not force:
        payload = pd.read_pickle(prepared_path)
        return dict(payload["summary"])

    df = _load_dataset_csv(data_dir / RAW_DIRNAME, entry)
    valid = df[df["components_in_single_pool"]].copy()
    valid = valid[np.isfinite(valid["DMS_score"])].copy()

    train_df = valid[(valid["num_mutations"] <= max_train_mutations)].copy()
    if max_test_mutations is None:
        test_df = valid[(valid["num_mutations"] >= min_test_mutations)].copy()
    else:
        test_df = valid[
            (valid["num_mutations"] >= min_test_mutations)
            & (valid["num_mutations"] <= max_test_mutations)
        ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Prepared split for {entry['DMS_id']} is empty: train={len(train_df)}, test={len(test_df)}."
        )

    train_df = train_df.rename(columns={"DMS_score": "fitness"})[
        ["mutant", "fitness", "DMS_score_bin", "num_mutations"]
    ].reset_index(drop=True)
    test_df = test_df.rename(columns={"DMS_score": "fitness"})[
        ["mutant", "fitness", "DMS_score_bin", "num_mutations"]
    ].reset_index(drop=True)

    payload = {
        "metadata": {
            "DMS_id": entry["DMS_id"],
            "DMS_filename": entry["DMS_filename"],
            "target_seq": entry["target_seq"],
            "seq_len": int(entry["seq_len"]),
            "source_organism": entry["source_organism"],
            "molecule_name": entry["molecule_name"],
            "selection_assay": entry["selection_assay"],
            "coarse_selection_type": entry["coarse_selection_type"],
            "benchmark_protocol": benchmark_protocol,
            "max_train_mutations": max_train_mutations,
            "min_test_mutations": min_test_mutations,
            "max_test_mutations": max_test_mutations,
        },
        "train_df": train_df,
        "test_df": test_df,
        "summary": {
            "DMS_id": entry["DMS_id"],
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "train_single_count": int((train_df["num_mutations"] == 1).sum()),
            "train_double_count": int((train_df["num_mutations"] == 2).sum()),
            "test_top_mutation_load": int(test_df["num_mutations"].max()),
            "benchmark_protocol": benchmark_protocol,
        },
    }
    prepared_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, prepared_path)
    return dict(payload["summary"])


def prepare_public_benchmark(
    data_dir: Path,
    benchmark_level: str = "lite",
    benchmark_protocol: str = DEFAULT_BENCHMARK_PROTOCOL,
    force: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(data_dir, benchmark_level)
    prepared_dir = Path(data_dir) / PREPARED_DIRNAME / benchmark_level / benchmark_protocol
    summaries = []
    for entry in manifest:
        summaries.append(
            prepare_single_dataset(
                Path(data_dir),
                entry,
                prepared_dir,
                force=force,
                benchmark_protocol=benchmark_protocol,
            )
        )
    summary = {
        "benchmark_level": benchmark_level,
        "benchmark_protocol": benchmark_protocol,
        "prepared_dir": str(prepared_dir),
        "datasets": summaries,
    }
    with (prepared_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def load_prepared_benchmark(
    data_dir: Path,
    benchmark_level: str = "lite",
    benchmark_protocol: str = DEFAULT_BENCHMARK_PROTOCOL,
) -> list[dict[str, Any]]:
    prepared_dir = Path(data_dir) / PREPARED_DIRNAME / benchmark_level / benchmark_protocol
    manifest = load_manifest(data_dir, benchmark_level)
    datasets = []
    for entry in manifest:
        path = prepared_dir / f"{entry['DMS_id']}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prepared dataset {path}. Run download_public_data.py and prepare_public_benchmark.py first."
            )
        payload = pd.read_pickle(path)
        datasets.append(payload)
    return datasets


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _topk_precision(y_true: np.ndarray, y_pred: np.ndarray, fraction: float = 0.05) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(math.ceil(n * fraction)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return float(len(true_top.intersection(pred_top)) / k)


def validate_predictions(predictions: Any, expected_len: int) -> np.ndarray:
    array = np.asarray(predictions, dtype=float)
    if array.ndim != 1 or len(array) != expected_len:
        raise ValueError(f"Predictions must be a 1D array-like of length {expected_len}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Predictions contain NaN or infinite values.")
    return array


def evaluate_dataset_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson_r = _pearson_r(y_true, y_pred)
    precision_top5 = _topk_precision(y_true, y_pred, fraction=0.05)
    combined = 0.7 * pearson_r + 0.3 * precision_top5
    return {
        "pearson_r": float(pearson_r),
        "precision_top5": float(precision_top5),
        "combined_score": float(combined),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return str(value)


def _capture_candidate_artifact(candidate_module: Any) -> dict[str, Any] | None:
    getter = getattr(candidate_module, "get_analysis_artifact", None)
    artifact = None
    if callable(getter):
        try:
            artifact = getter()
        except Exception as exc:
            artifact = {"artifact_error": str(exc)}
    elif hasattr(candidate_module, "_PACEVOLVE_LAST_ANALYSIS_ARTIFACT"):
        artifact = getattr(candidate_module, "_PACEVOLVE_LAST_ANALYSIS_ARTIFACT")
    if artifact is None:
        return None
    safe = _json_safe(artifact)
    return safe if isinstance(safe, dict) else {"value": safe}


def _save_analysis_artifact(
    artifact_dir: str,
    metrics: dict[str, Any],
    dataset_artifacts: list[dict[str, Any]],
) -> None:
    if not artifact_dir:
        return
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "multievolve_extrapolate",
        "overall_metrics": _json_safe(metrics),
        "datasets": _json_safe(dataset_artifacts),
    }
    with (output_dir / "analysis_artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def evaluate_predictions(
    benchmark_payloads: list[dict[str, Any]],
    candidate_module: Any,
) -> dict[str, Any]:
    if not hasattr(candidate_module, "fit_and_predict"):
        raise AttributeError("Candidate is missing `fit_and_predict(train_df, test_df, dataset_context)`.")

    dataset_metrics = []
    dataset_artifacts = []
    pearsons = []
    precisions = []

    for payload in benchmark_payloads:
        train_df = payload["train_df"].copy()
        test_df = payload["test_df"].copy()
        dataset_context = dict(payload["metadata"])
        predictions = candidate_module.fit_and_predict(train_df, test_df, dataset_context)
        y_pred = validate_predictions(predictions, expected_len=len(test_df))
        y_true = test_df["fitness"].to_numpy(dtype=float)
        metrics = evaluate_dataset_predictions(y_true, y_pred)
        metrics["dataset_id"] = dataset_context["DMS_id"]
        metrics["test_size"] = int(len(test_df))
        pearsons.append(metrics["pearson_r"])
        precisions.append(metrics["precision_top5"])
        dataset_metrics.append(metrics)
        dataset_artifacts.append(
            {
                "dataset_id": dataset_context["DMS_id"],
                "metrics": dict(metrics),
                "prediction_summary": {
                    "num_predictions": int(len(y_pred)),
                    "prediction_mean": float(np.mean(y_pred)) if len(y_pred) else 0.0,
                    "prediction_std": float(np.std(y_pred)) if len(y_pred) else 0.0,
                    "prediction_min": float(np.min(y_pred)) if len(y_pred) else 0.0,
                    "prediction_max": float(np.max(y_pred)) if len(y_pred) else 0.0,
                },
                "candidate_artifact": _capture_candidate_artifact(candidate_module),
            }
        )

    mean_pearson = float(np.mean(pearsons)) if pearsons else 0.0
    mean_precision_top5 = float(np.mean(precisions)) if precisions else 0.0
    combined_score = float(0.7 * mean_pearson + 0.3 * mean_precision_top5)
    benchmark_protocol = None
    if benchmark_payloads:
        benchmark_protocol = benchmark_payloads[0].get("metadata", {}).get("benchmark_protocol")
    result = {
        "combined_score": combined_score,
        "mean_pearson_r": mean_pearson,
        "mean_precision_top5": mean_precision_top5,
        "num_datasets": int(len(dataset_metrics)),
        "benchmark_protocol": benchmark_protocol,
        "datasets": dataset_metrics,
    }
    _save_analysis_artifact(
        os.environ.get("PACEVOLVE_ARTIFACT_DIR", "").strip(),
        result,
        dataset_artifacts,
    )
    return result
