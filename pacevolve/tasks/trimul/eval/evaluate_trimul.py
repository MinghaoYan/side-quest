"""CLI evaluator for the PACE-RL TriMul task."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import dataclasses
import importlib.util
import math
import sys
import time
import traceback
from pathlib import Path

import torch

import reference


SCORE_SCALE = getattr(reference, "SCORE_SCALE", 3000.0)
BENCH_USE_CUDA_EVENTS = getattr(reference, "BENCH_USE_CUDA_EVENTS", True)
BENCH_REL_ERROR = getattr(reference, "BENCH_REL_ERROR", 0.001)
BENCH_WALL_TIMEOUT_NS = getattr(reference, "BENCH_WALL_TIMEOUT_NS", 120e9)
BENCH_NO_GRAD = getattr(reference, "BENCH_NO_GRAD", False)
BENCH_MAX_REPEATS = getattr(reference, "BENCH_MAX_REPEATS", 100)
BENCH_MAX_TIME_NS = getattr(reference, "BENCH_MAX_TIME_NS", 10e9)
BENCH_WARMUP_STYLE = getattr(reference, "BENCH_WARMUP_STYLE", "tiny_benchmark")


def _clone(data):
    if isinstance(data, tuple):
        return tuple(_clone(x) for x in data)
    if isinstance(data, list):
        return [_clone(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        fields = {f.name: _clone(getattr(data, f.name)) for f in dataclasses.fields(data)}
        return type(data)(**fields)
    if isinstance(data, torch.nn.Module):
        return copy.deepcopy(data)
    return data


def _stats(durations_ns: list[float]) -> dict[str, float]:
    runs = len(durations_ns)
    mean = sum(durations_ns) / runs
    if runs > 1:
        variance = sum((x - mean) ** 2 for x in durations_ns) / (runs - 1)
        std = math.sqrt(variance)
        err = std / math.sqrt(runs)
    else:
        std = 0.0
        err = 0.0
    return {"runs": runs, "mean": mean, "std": std, "err": err}


def _load_candidate(candidate_path: str):
    spec = importlib.util.spec_from_file_location("trimul_submission", candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["trimul_submission"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "custom_kernel"):
        raise AttributeError("Missing `custom_kernel` function")
    return module.custom_kernel


def _syntax_check(candidate_path: str) -> None:
    source = Path(candidate_path).read_text(encoding="utf-8")
    compile(source, candidate_path, "exec")
    module_ast = ast.parse(source, filename=candidate_path)
    has_custom_kernel = any(
        isinstance(node, ast.FunctionDef) and node.name == "custom_kernel"
        for node in module_ast.body
    )
    if not has_custom_kernel:
        raise AttributeError("Missing `custom_kernel` function")


def _bench_single(kernel_fn, bench_args: dict, max_time_ns: float | None = None):
    if max_time_ns is None:
        max_time_ns = BENCH_MAX_TIME_NS

    data = reference.generate_input(**bench_args)
    data_copy = _clone(data)

    ctx = torch.no_grad() if BENCH_NO_GRAD else contextlib.nullcontext()
    with ctx:
        output = kernel_fn(data)
        torch.cuda.synchronize()
        passed, message = reference.check_implementation(data_copy, output)
    if not passed:
        return None, f"Benchmark correctness: {message}"
    del output

    durations_ns: list[float] = []
    bench_start = time.perf_counter_ns()
    with ctx:
        for idx in range(BENCH_MAX_REPEATS):
            torch.cuda.synchronize()
            if BENCH_USE_CUDA_EVENTS:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = kernel_fn(data)
                end_event.record()
                torch.cuda.synchronize()
                duration_ns = start_event.elapsed_time(end_event) * 1e6
            else:
                start_ns = time.perf_counter_ns()
                output = kernel_fn(data)
                torch.cuda.synchronize()
                duration_ns = time.perf_counter_ns() - start_ns
            del output
            durations_ns.append(duration_ns)

            if idx > 1:
                stats = _stats(durations_ns)
                if stats["mean"] > 0 and stats["err"] / stats["mean"] < BENCH_REL_ERROR:
                    break
                if stats["mean"] * stats["runs"] > max_time_ns:
                    break
                if (
                    BENCH_WALL_TIMEOUT_NS is not None
                    and time.perf_counter_ns() - bench_start > BENCH_WALL_TIMEOUT_NS
                ):
                    break

    return _stats(durations_ns), None


def _warmup(kernel_fn, bench_args: dict) -> None:
    if BENCH_WARMUP_STYLE == "timed_calls":
        data = reference.generate_input(**bench_args)
        start = time.perf_counter()
        while time.perf_counter() - start < 0.2:
            kernel_fn(data)
            torch.cuda.synchronize()
        return

    _bench_single(kernel_fn, bench_args, max_time_ns=10e7)


def evaluate(candidate_path: str) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TriMul evaluation.")

    custom_kernel = _load_candidate(candidate_path)

    for idx, test_case in enumerate(reference.TEST_CASES):
        try:
            data = reference.generate_input(**test_case)
            data_copy = _clone(data)
            torch.cuda.synchronize()
            output = custom_kernel(data)
            torch.cuda.synchronize()
            passed, message = reference.check_implementation(data_copy, output)
            if not passed:
                raise RuntimeError(f"Test {idx} failed: {message}")
        except Exception as exc:
            raise RuntimeError(f"Correctness failure on test {idx}: {exc}") from exc

    _warmup(custom_kernel, reference.BENCHMARK_CASES[0])

    bench_means_ns: list[float] = []
    for idx, bench_args in enumerate(reference.BENCHMARK_CASES):
        stats, error = _bench_single(custom_kernel, bench_args)
        if error:
            raise RuntimeError(f"Benchmark {idx} failed: {error}")
        assert stats is not None
        bench_means_ns.append(stats["mean"])

    means_seconds = [ns / 1e9 for ns in bench_means_ns]
    geom_mean_s = math.pow(math.prod(means_seconds), 1.0 / len(means_seconds))
    geom_mean_us = geom_mean_s * 1e6
    return SCORE_SCALE / geom_mean_us


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a TriMul candidate.")
    parser.add_argument("--candidate_path", required=True, type=str)
    parser.add_argument(
        "--syntax_only",
        action="store_true",
        help="Only validate syntax/imports without running GPU correctness or benchmarks.",
    )
    args = parser.parse_args()

    try:
        if args.syntax_only:
            _syntax_check(args.candidate_path)
            print("Syntax check passed.")
            return 0

        score = evaluate(args.candidate_path)
        print(f"Kernel speedup: {score:.6f}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
