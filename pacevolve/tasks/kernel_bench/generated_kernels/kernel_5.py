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

# Set CUDA architecture for A100-SXM4-40GB, which has compute capability 8.0
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 0):
        """
        Initializes the new model. The num_features argument is for API compatibility and not used.
        """
        super(ModelNew, self).__init__()
        
        gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU approximation constants
const float GELU_SCALAR_1 = 0.7978845608028654f; // sqrt(2.0 / PI)
const float GELU_SCALAR_2 = 0.044715f;

// Device-side helper function for GELU computation using Fused Multiply-Add (FMA).
// Inlining is forced to eliminate function call overhead.
__device__ __forceinline__ float gelu_fma(const float x) {
    const float x_sq = x * x;
    // Computes: GELU_SCALAR_2 * x_sq * x + x
    const float inner = fmaf(GELU_SCALAR_2 * x_sq, x, x);
    const float v = GELU_SCALAR_1 * inner;
    const float t = tanhf(v);
    // Computes: 0.5f * t + 0.5f, which is equivalent to 0.5f * (1.0f + t)
    const float cdf = fmaf(0.5f, t, 0.5f);
    return x * cdf;
}

// Kernel using a grid-stride loop to process float4 vectors.
// This design improves scalability by decoupling the grid size from the input size.
// It also simplifies the logic by removing the need for a separate tail-handling branch
// within the main processing logic.
__global__ void gelu_grid_stride_float4_fma_kernel(const float* __restrict__ input, float* __restrict__ output, const int size) {
    // The number of float4 vectors that fit entirely in the input tensor.
    const int num_float4_elements = size / 4;
    
    // The stride for the grid-stride loop is the total number of threads in the grid.
    const int grid_stride = gridDim.x * blockDim.x;

    // Grid-stride loop for the main body (elements divisible by 4)
    // Each thread processes multiple float4 elements separated by the grid stride.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_float4_elements; i += grid_stride) {
        // Load one float4 vector.
        const float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        
        float4 out_vec;
        // Process each component of the vector.
        out_vec.x = gelu_fma(in_vec.x);
        out_vec.y = gelu_fma(in_vec.y);
        out_vec.z = gelu_fma(in_vec.z);
        out_vec.w = gelu_fma(in_vec.w);
        
        // Store the resulting float4 vector.
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // After the loop, threads collaboratively handle the remaining tail elements (size % 4).
    const int tail_start_idx = num_float4_elements * 4;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tail_idx = tail_start_idx + thread_id;

    // Each thread handles one potential tail element.
    if (tail_idx < size) {
        output[tail_idx] = gelu_fma(input[tail_idx]);
    }
}


// C++ wrapper function to launch the CUDA kernel from PyTorch.
torch::Tensor gelu_cuda(torch::Tensor input) {
    // Input validation checks.
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    const auto size = input.numel();
    auto output = torch::empty_like(input);

    if (size == 0) {
        return output;
    }
    
    // Kernel launch configuration.
    // Use a large block size, as shown effective in prior experiments.
    const int block_size = 1024;
    
    // For a grid-stride loop, we launch a grid that is large enough to saturate the
    // GPU, but not necessarily large enough to cover the entire input in one go.
    // This decouples the grid size from the input size.
    // An A100 GPU has 108 SMs. Launching 2 blocks per SM is a good heuristic.
    const int num_blocks = 108 * 2; // 216 blocks
    
    // Launch the grid-stride kernel.
    gelu_grid_stride_float4_fma_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    // Error checking for the kernel launch.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}
"""

        gelu_cpp_source = """
#include <torch/extension.h>

// Forward declaration of the C++ function that launches the CUDA kernel.
torch::Tensor gelu_cuda(torch::Tensor input);
"""

        # Use load_inline to JIT compile the CUDA C++ code.
        # A unique name is used to avoid caching issues with different experiments.
        self.gelu_cuda_module = load_inline(
            name='gelu_cuda_fixedgrid_A100',
            cpp_sources=gelu_cpp_source,
            cuda_sources=gelu_source,
            functions=['gelu_cuda'],
            verbose=True,
            extra_cuda_cflags=['--use_fast_math', '-O3']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the custom GELU activation function.
        
        Args:
            x (torch.Tensor): The input tensor. Assumed to be on the correct CUDA device.
            
        Returns:
            torch.Tensor: The output tensor after applying the custom GELU.
        """
        return self.gelu_cuda_module.gelu_cuda(x)