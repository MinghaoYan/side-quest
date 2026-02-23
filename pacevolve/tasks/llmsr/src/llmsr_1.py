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

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_percentage_error

class LLMSR():

    def compute_output_base_metrics(self, y_pred, y):
        nonnan_idx = np.argwhere(~np.isnan(y_pred))
        y_pred = y_pred[nonnan_idx]
        y = y[nonnan_idx]

        var = np.var(y)
        nmse = np.mean((y - y_pred)**2) / var 
        if np.sum((y - y.mean())**2) == 0:
            print(y)
        r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
        kdt = kendalltau(y, y_pred)[0]
        mape = mean_absolute_percentage_error(y, y_pred)
        log10_nmse = np.log10(nmse)

        return {
            "mse": np.mean((y - y_pred)**2),
            "nmse": nmse,
            "log10_nmse": log10_nmse,
            "r2": r2,
            "kdt": kdt,
            "mape": mape,
            # "num_valid_points": len(y_pred),
        }

    def evaluate(self, train_data: dict, test_data: dict, ood_test_data: dict) -> float:
        ''' Evaluate the equation on data observations.'''
        
        # Load data observations
        MAX_NPARAMS = 10
        params = [1.0]*MAX_NPARAMS
        inputs, outputs = train_data['inputs'], train_data['outputs']
        X = inputs
        
        # Optimize parameters based on data
        from scipy.optimize import minimize
        def loss(params):
            x_inputs = X[:, 0]
            t_inputs = X[:, 1]
            v_inputs = X[:, 2]

            # Pass the individual columns to the equation
            y_pred = self.equation(x_inputs, t_inputs, v_inputs, params)
            # y_pred = self.equation(*X, params)
            return np.mean((y_pred - outputs) ** 2)

        loss_partial = lambda params: loss(params)
        result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
        
        # Return evaluation score
        optimized_params = result.x
        inputs, outputs = test_data['inputs'], test_data['outputs']
        X = inputs
        x_inputs = X[:, 0]
        t_inputs = X[:, 1]
        v_inputs = X[:, 2]
        
        y_pred = self.equation(x_inputs, t_inputs, v_inputs, optimized_params)
        metrics = self.compute_output_base_metrics(y_pred, outputs)

        inputs, outputs = ood_test_data['inputs'], ood_test_data['outputs']
        X = inputs
        x_inputs = X[:, 0]
        t_inputs = X[:, 1]
        v_inputs = X[:, 2]
        
        y_pred = self.equation(x_inputs, t_inputs, v_inputs, optimized_params)
        ood_metrics = self.compute_output_base_metrics(y_pred, outputs)

        return {'log10_nmse': metrics['log10_nmse'], 'ood_log10_nmse': ood_metrics['log10_nmse']}

    # RegexTagCustomPruningAlgorithmStart
    def equation(self, x: np.ndarray, t: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        """ Mathematical function for Acceleration in Non-linear Harmonic Oscillator

        Args:
            x: A numpy array representing observations of Position at time t.
            t: A numpy array representing observations of Time.
            v: A numpy array representing observations of Velocity at time t.
            params: Array of numeric constants or parameters to be optimized

        Return:
            A numpy array representing Acceleration in Non-linear Harmonic Oscillator as the result of applying the mathematical function to the inputs.
        """
        restoring_force = params[0] * x + params[1] * x**3 + params[2] * x**5

        damping_force = params[3] * v + params[4] * v**3

        driving_force = params[5] * np.cos(t) + params[6] * np.sin(t)
        
        parametric_force = (params[7] * x + params[8] * x**3 + params[9] * x**5) * np.cos(2 * t)

        output = restoring_force + damping_force + driving_force + parametric_force

        return output
    # RegexTagCustomPruningAlgorithmEnd

