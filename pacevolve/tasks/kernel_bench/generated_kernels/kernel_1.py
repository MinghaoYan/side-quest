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

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

fused_batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For sqrtf

// Kernel to pre-compute the effective scale (gamma_eff) and bias (beta_eff)
// This refactors the batch norm calculation into a single FMA operation.
__global__ void precompute_params_kernel(
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ gamma_eff,
    float* __restrict__ beta_eff,
    int channels,
    float epsilon
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        // Use 1.0f / sqrtf for better precision to match the baseline
        float inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
        
        // Effective scale: gamma_eff = gamma / sqrt(var + eps)
        gamma_eff[c] = gamma[c] * inv_std;
        
        // Effective bias: beta_eff = beta - (mean * gamma) / sqrt(var + eps)
        // Which is beta - mean * gamma_eff
        beta_eff[c] = beta[c] - running_mean[c] * gamma_eff[c];
    }
}

// Main batch norm kernel combining FMA for computation and float4 for memory access.
__global__ void batch_norm_fma_float4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma_eff,
    const float* __restrict__ beta_eff,
    float* __restrict__ output,
    int total_elements,
    int C, // Channels
    int H, // Height
    int W  // Width
) {
    // Each thread processes a float4, i.e., 4 elements.
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_vecs = total_elements / 4;

    if (vec_idx < num_vecs) {
        // The linear index of the first element of the float4
        int element_idx = vec_idx * 4;
        
        // Calculate channel index correctly using provided dimensions
        int spatial_size = H * W;
        int c = (element_idx / spatial_size) % C;

        // Load effective parameters for the current channel
        const float scale = gamma_eff[c];
        const float shift = beta_eff[c];

        // Cast pointers to float4 for vectorized memory access
        const float4* input4_ptr = reinterpret_cast<const float4*>(input);
        float4* output4_ptr = reinterpret_cast<float4*>(output);

        // Load 4 elements at once
        float4 in_val = input4_ptr[vec_idx];

        // Apply Fused Multiply-Add on all 4 elements
        float4 out_val;
        out_val.x = fmaf(in_val.x, scale, shift);
        out_val.y = fmaf(in_val.y, scale, shift);
        out_val.z = fmaf(in_val.z, scale, shift);
        out_val.w = fmaf(in_val.w, scale, shift);

        // Store 4 elements at once
        output4_ptr[vec_idx] = out_val;
    }
}

// C++ Wrapper function to orchestrate the kernel launches
std::vector<torch::Tensor> batch_norm_fused_fma_float4_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.size(3) % 4 == 0, "Last dimension (width) must be divisible by 4 for float4 vectorization");

    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int total_elements = input.numel();
    
    auto output = torch::empty_like(input);
    
    // Allocate temporary tensors for effective gamma and beta
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kFloat32);
    auto gamma_eff = torch::empty({C}, options);
    auto beta_eff = torch::empty({C}, options);

    // --- Step 1: Launch pre-computation kernel ---
    const int precompute_threads = 256;
    const int precompute_blocks = (C + precompute_threads - 1) / precompute_threads;
    precompute_params_kernel<<<precompute_blocks, precompute_threads>>>(
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma_eff.data_ptr<float>(),
        beta_eff.data_ptr<float>(),
        C,
        epsilon
    );

    // --- Step 2: Launch main vectorized FMA kernel ---
    const int main_threads = 256;
    const int main_blocks = (total_elements / 4 + main_threads - 1) / main_threads;
    batch_norm_fma_float4_kernel<<<main_blocks, main_threads>>>(
        input.data_ptr<float>(),
        gamma_eff.data_ptr<float>(),
        beta_eff.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        C, H, W // Pass dimensions to the kernel
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return {output};
}
"""

fused_batch_norm_cpp_source = """
std::vector<torch::Tensor> batch_norm_fused_fma_float4_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta, 
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
);
"""

# Load the CUDA kernel inline
batch_norm_cuda_optimized = load_inline(
    name='batch_norm_fused_fma_float4',
    cpp_sources=fused_batch_norm_cpp_source,
    cuda_sources=fused_batch_norm_source,
    functions=['batch_norm_fused_fma_float4_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # These parameters are equivalent to BatchNorm2d's weight and bias
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # These are buffers, not parameters, and are updated during training
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.eps = 1e-5
        # Link to our custom CUDA function
        self.batch_norm = batch_norm_cuda_optimized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The model is in eval mode, so we use running_mean and running_var
        # Ensure all tensors are on the same CUDA device as the input
        return self.batch_norm.batch_norm_fused_fma_float4_cuda(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps
        )[0]