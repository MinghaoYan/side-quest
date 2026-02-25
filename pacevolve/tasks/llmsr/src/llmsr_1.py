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
        """
        Mathematical function for Acceleration in a Non-linear Harmonic Oscillator.

        This model improves upon a polynomial-based approach by incorporating a
        generalized power-law damping term. This provides greater flexibility in
        capturing a wide range of dissipative behaviors, where the damping
        force is proportional to a variable power of velocity. To ensure numerical
        stability, the exponent of the power law is clipped, and calculations
        handle zero-velocity conditions carefully to avoid division by zero.

        Args:
            x: A numpy array representing observations of Position at time t.
            t: A numpy array representing observations of Time. (unused)
            v: A numpy array representing observations of Velocity at time t.
            params: Array of 10 numeric constants or parameters to be optimized.

        Return:
            A numpy array representing Acceleration as the result of applying
            the mathematical function to the inputs.
        """
        
        # Polynomial restoring force captures non-linear spring-like behavior.
        # params[0]: Constant offset or external force
        # params[1]*x: Linear restoring force (Hooke's Law)
        # params[2]*x**2: Quadratic restoring force
        # params[3]*x**3: Cubic restoring force (e.g., Duffing oscillator)
        restoring_force = (params[0] +
                           params[1] * x +
                           params[2] * x**2 +
                           params[3] * x**3)
        
        # Damping and interaction terms.
        # params[4]*v: Standard linear damping
        # params[6]*x*v, params[7]*x**2*v: Position-dependent damping terms
        damping_and_interaction = (params[4] * v +
                                   params[6] * x * v +
                                   params[7] * (x**2) * v)
        
        # Generalized power-law damping term.
        # To prevent numerical instability (overflow or division by zero), we
        # constrain the exponent `params[8]` to a reasonable range.
        exponent = np.clip(params[8], -5.0, 5.0)

        # We calculate the power term safely. We initialize the term with zeros
        # and then compute the power only for non-zero velocities. This avoids
        # the case of 0 raised to a negative power, which would result in 'inf'.
        power_v = np.zeros_like(v, dtype=np.float64)
        non_zero_v_mask = (v != 0)
        
        # Only apply power to non-zero elements of v
        abs_v_non_zero = np.abs(v[non_zero_v_mask])
        power_v[non_zero_v_mask] = np.power(abs_v_non_zero, exponent)

        # The full generalized damping term. np.sign(v) correctly handles the
        # direction and ensures the force is zero when velocity is zero.
        generalized_damping = params[5] * np.sign(v) * power_v

        # A sinusoidal restoring force term is included to capture periodic
        # potential wells or other oscillating components in the restoring force.
        # params[9]: Amplitude of the sinusoidal restoring force.
        periodic_restoring_force = params[9] * np.sin(x)

        # The final acceleration is the sum of all force components.
        output = restoring_force + damping_and_interaction + generalized_damping + periodic_restoring_force

        return output
    # RegexTagCustomPruningAlgorithmEnd

