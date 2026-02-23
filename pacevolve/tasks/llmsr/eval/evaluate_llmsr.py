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

import traceback
import numpy as np
import argparse
import importlib.util
import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import Optional, Any
import datasets
import h5py

@dataclass
class Equation:
    symbols: list
    symbol_descs: list
    symbol_properties: list
    expression: str
    desc: Optional[str] = None

    sympy_format: Optional[Any] = None
    lambda_format: Optional[callable] = None
    program_format: Optional[str] = None

@dataclass
class SearchResult:
    equation: Equation
    aux: Any

@dataclass
class SEDTask:
    name: str
    symbols: list
    symbol_descs: list
    symbol_properties: list
    samples: Any
    desc: Optional[str] = None

@dataclass
class Problem:
    dataset_identifier: str
    equation_idx: str
    gt_equation: Equation
    samples: Any

    def create_task(self) -> SEDTask:
        return SEDTask(name=self.equation_idx,
                        symbols=self.gt_equation.symbols,
                        symbol_descs=self.gt_equation.symbol_descs,
                        symbol_properties=self.gt_equation.symbol_properties,
                        samples=self.train_samples,
                        desc=self.gt_equation.desc)
    @property
    def train_samples(self):
        return self.samples['train']
    
    @property
    def test_samples(self):
        return self.samples['test']
    
    @property
    def ood_test_samples(self):
        return self.samples.get('ood_test', None) 



def load_class_from_file(path, class_name="LLMSR"):
    """Loads a specific class from a Python source file by its name."""
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {path}")
    module = importlib.util.module_from_spec(spec)
    # Use a unique name for the module in sys.modules to avoid conflicts
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        raise AttributeError(f"Class '{class_name}' not found in {path}")


def main():
    """Parses arguments, runs candidate and baseline, and prints the diff."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=False, default=None)
    parser.add_argument("--output", type=str, required=False, default="trial_results.pkl")
    parser.add_argument("--problem_idx", type=int, required=False, default=0)
    parser.add_argument("--candidate_path", type=str, required=True)

    args = parser.parse_args()
    
    # Define paths to candidate and reference implementations
    # candidate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'llmsr.py'))
    print(args.candidate_path)

    try:
        # Load classes
        CandidateClass = load_class_from_file(args.candidate_path)

        ds = datasets.load_dataset(args.data_path, data_files="llm-srbench-lsr_synth_phys_osc.arrow")
        sample_h5file_path = os.path.join(args.data_path, "lsr_bench_data.hdf5")
        # problems = []
        e = ds['train'][args.problem_idx]
        
        with h5py.File(sample_h5file_path, "r") as sample_file:
            # Iterate over the converted data (which is a list of dictionaries)
            samples = {k:v[...].astype(np.float64) for k,v in sample_file[f'/lsr_synth/phys_osc/{e["name"]}'].items()}
            problem = Problem(dataset_identifier='phys_osc',
                            equation_idx = e['name'],
                            gt_equation=Equation(
                                symbols=e['symbols'],
                                symbol_descs=e['symbol_descs'],
                                symbol_properties=e['symbol_properties'],
                                expression=e['expression'],
                            ),
                            samples=samples)
        
        X_id = problem.train_samples[:, 1:]
        y_id = problem.train_samples[:, 0] 
        test_X_id = problem.test_samples[:, 1:]
        test_y_id = problem.test_samples[:, 0] 
        ood_test_X_id = problem.ood_test_samples[:, 1:]
        ood_test_y_id = problem.ood_test_samples[:, 0] 

        candidate_value = CandidateClass().evaluate({'inputs': X_id, 'outputs': y_id}, {'inputs': test_X_id, 'outputs': test_y_id}, {'inputs': ood_test_X_id, 'outputs': ood_test_y_id})
        
        print(f"Candidate: {candidate_value}")
        # Assume the higher the better when comparing, so add a minus sign if you smaller is better.
        # if args.config_path is not None: 
        #     with open(config['paths']['results_path']+'/'+args.dataset+'.pkl', 'wb') as file:
        #         pickle.dump([candidate_value], file)
    except Exception as e:
        print(f"Evaluation script failed with error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()
