"""CLI evaluator for the KuaRec Wukong task."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _run_command(command: list[str]) -> int:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a KuaRec Wukong candidate.")
    parser.add_argument("--candidate_path", type=str, required=True)
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--syntax_only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.candidate_path):
        print(f"Candidate script not found: {args.candidate_path}", file=sys.stderr)
        raise SystemExit(1)

    if not args.syntax_only and not os.path.exists(args.dataset_csv):
        print(f"Dataset CSV not found: {args.dataset_csv}", file=sys.stderr)
        raise SystemExit(1)

    command = [sys.executable, args.candidate_path]
    if args.syntax_only:
        command.append("--syntax_only")
    else:
        command.extend(["--dataset_csv", args.dataset_csv])

    raise SystemExit(_run_command(command))


if __name__ == "__main__":
    main()

