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
import math

# Set the CUDA architecture to be compatible with A100 GPUs (Compute Capability 8.0).
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# In-line C++/CUDA source for a tiled GELU kernel using a fast tanh approximation.
# This kernel combines several optimization strategies:
# 1. Tiling & Shared Memory: Each thread block cooperatively loads a 'tile' of data
#    (256 float4 vectors) into shared memory to improve data locality and reuse.
# 2. Vectorization: All memory operations and computations are performed on float4
#    vectors to maximize memory bandwidth and computational throughput.
# 3. Fast Tanh Approximation: Replaces the computationally intensive erff() with a
#    faster, high-performance tanh approximation using the __expf intrinsic. This
#    builds on the success of previous experiments.
# 4. Numerical Stability: The custom tanh function includes input clamping to
#    prevent numerical overflow/underflow from the __expf intrinsic.
# 5. Grid-Stride Loop: Ensures the kernel can process inputs of any size while
#    keeping all Streaming Multiprocessors (SMs) utilized.
# 6. Compiler Optimizations: Compiled with `--use_fast_math` for additional
#    performance gains from approximate math intrinsics.
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// A numerically stable, fast tanh approximation using the __expf intrinsic.
// Input is clamped to a safe range to avoid NaN/inf results from expf(2*x).
__device__ __forceinline__ float fast_tanhf(float x) {
    // Clamp input to avoid overflow in __expf. The range [-10, 10] is sufficient
    // as tanh(10) is very close to 1.0 and tanh(-10) is very close to -1.0.
    x = fmaxf(-10.0f, fminf(10.0f, x));
    float exp_val = __expf(2.0f * x);
    return (exp_val - 1.0f) / (exp_val + 1.0f);
}

// A __device__ function for the GELU activation using the fast tanh approximation.
// Constants are pre-calculated for efficiency.
__device__ __forceinline__ float gelu_tanh_approx_compute(float x) {
    const float c1 = 0.044715f;
    const float c2 = 0.79788456f; // sqrt(2.0f / M_PI)
    return 0.5f * x * (1.0f + fast_tanhf(c2 * (x + c1 * x * x * x)));
}

// A __device__ function that applies the fast GELU approximation to a float4 vector.
__device__ __forceinline__ float4 gelu_tanh_approx_compute_f4(const float4& v) {
    float4 out;
    out.x = gelu_tanh_approx_compute(v.x);
    out.y = gelu_tanh_approx_compute(v.y);
    out.z = gelu_tanh_approx_compute(v.z);
    out.w = gelu_tanh_approx_compute(v.w);
    return out;
}

// The main CUDA kernel using a tiled approach with shared memory and the fast tanh approximation.
__global__ void gelu_tiled_smem_tanh_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    // Dynamic shared memory allocated at launch time.
    extern __shared__ float4 smem[];

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x; // e.g., 256
    const int grid_stride_threads = gridDim.x * block_threads;

    const int tile_size_f4 = block_threads; // One float4 per thread
    const int vectorized_size_f4 = size / 4;

    // Grid-stride loop over tiles. Each block processes one tile per iteration.
    for (int tile_base_f4 = blockIdx.x * tile_size_f4; tile_base_f4 < vectorized_size_f4; tile_base_f4 += gridDim.x * tile_size_f4) {
        const int thread_in_tile_offset_f4 = tid;
        const int global_idx_f4 = tile_base_f4 + thread_in_tile_offset_f4;

        // Step 1: Cooperatively load a tile from global to shared memory.
        if (global_idx_f4 < vectorized_size_f4) {
            smem[thread_in_tile_offset_f4] = ((const float4*)input)[global_idx_f4];
        }
        __syncthreads(); // Ensure all loads are complete.

        // Step 2: Compute from shared memory and store back to global memory.
        if (global_idx_f4 < vectorized_size_f4) {
            float4 val = smem[thread_in_tile_offset_f4];
            val = gelu_tanh_approx_compute_f4(val);
            ((float4*)output)[global_idx_f4] = val;
        }

        // Synchronize before the next tile iteration to prevent race conditions.
        __syncthreads();
    }

    // Handle the remaining scalar elements (size % 4) using a standard grid-stride loop.
    const int cleanup_start_idx = vectorized_size_f4 * 4;
    for (int i = cleanup_start_idx + blockIdx.x * blockDim.x + threadIdx.x; i < size; i += grid_stride_threads) {
        output[i] = gelu_tanh_approx_compute(input[i]);
    }
}


// C++ function that serves as the interface between PyTorch and the CUDA kernel.
torch::Tensor gelu_cuda(torch::Tensor input) {
    // Input validation.
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be of type float32");

    const auto size = input.numel();
    if (size == 0) {
        return torch::empty_like(input);
    }

    auto output = torch::empty_like(input);

    // Configure kernel launch parameters.
    const int block_size = 256;
    // Launch enough blocks to keep the GPU busy, but not an excessive amount.
    const int num_blocks = std::min((int)((size + block_size * 4 - 1) / (block_size * 4)), 4096);

    // Calculate dynamic shared memory size for the tile.
    const size_t shared_mem_size = block_size * sizeof(float4);

    // Launch the CUDA kernel.
    gelu_tiled_smem_tanh_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    // Check for any errors during kernel launch or execution.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for defining the function signature that PyTorch will bind to.
gelu_cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor input);
"""

# Use torch.utils.cpp_extension.load_inline to compile the C++/CUDA code on-the-fly.
gelu_cuda_impl = load_inline(
    name='gelu_cuda_impl_tiled_smem_tanh',
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    extra_cuda_cflags=['--use_fast_math'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    An optimized PyTorch model that replaces the standard GELU activation
    with a custom CUDA kernel. This implementation combines the shared-memory
    tiling strategy from the SOTA with the high-performance, numerically-stable
    tanh approximation identified as the best computational approach in prior
    experiments.
    """
    def __init__(self, num_features: int = 0):
        """
        Initializes the new model. The num_features argument is ignored but
        kept for signature compatibility.
        """
        super(ModelNew, self).__init__()
        # Store the loaded CUDA function for use in the forward pass.
        self.gelu_cuda = gelu_cuda_impl.gelu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the custom GELU kernel.
        """
        return self.gelu_cuda(x)