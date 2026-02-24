from src.utils import read_file
from src.eval import eval_kernel_against_ref
import argparse
import sys

parser = argparse.ArgumentParser(description="Train a model with an optional compile flag for a quick test run.")

parser.add_argument(
    "--baseline_path", 
    type=str,
    required=True,
    help="Path pattern for training data files."
)

# ADDED: Argument for validation files
parser.add_argument(
    "--kernel_path", 
    type=str,
    required=True,
    help="Path pattern for validation data files."
)

parser.add_argument(
    "--build_dir", 
    type=str,
    required=True,
    help="Unique path for this kernel's compilation cache."
)

parser.add_argument(
    "--baseline_time", 
    type=float,
    required=True,
    help="Kernel baseline time."
)


args = parser.parse_args()

# eval_kernel_against_ref(original_model_src=read_file(args.baseline_path), custom_model_src=read_file(args.baseline_path), measure_performance=True)
result = eval_kernel_against_ref(original_model_src=read_file(args.baseline_path), custom_model_src=read_file(args.kernel_path), baseline_time=args.baseline_time, measure_performance=True, build_dir=args.build_dir)

if result is None:
    print("[Eval Script] Exiting with error code 1 due to lack of output.")
    sys.exit(1)
elif not result.compiled:
    print("[Eval Script] Exiting with error code 1 due to compilation or critical failure.")
    sys.exit(1)
elif not result.correctness:
    print("[Eval Script] Exiting with error code 1 due to incorrect kernel.")
    sys.exit(1)