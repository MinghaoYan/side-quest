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

# Set CUDA architecture for A100 to enable SM 8.0 features.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# The C++/CUDA source code for the custom kernel is defined in a single string.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>
#include <c10/cuda/CUDAException.h> // Header for C10_CUDA_CHECK
#include <algorithm> 

// __device__ function for a 16-thread group sum reduction using shuffle-xor instructions.
__device__ __forceinline__ float halfWarpReduceSumXOR(float val) {
    const unsigned int mask = 0xffffffff;
    val += __shfl_xor_sync(mask, val, 8);
    val += __shfl_xor_sync(mask, val, 4);
    val += __shfl_xor_sync(mask, val, 2);
    val += __shfl_xor_sync(mask, val, 1);
    return val;
}

// Unified reduction kernel with a grid-stride loop and path-specific caching hints via PTX.
// The template parameter selects between a contiguous path optimized for streaming memory
// access and a non-contiguous path optimized to reduce cache pollution.
template <bool IsContiguous>
__global__ __launch_bounds__(256, 4) void grid_stride_reduction_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    const int reduce_dim_size, const int outer_size, const int inner_size, const int total_outputs,
    const float inv_reduce_dim_size) {

    if constexpr (IsContiguous) {
        // Path 1: Contiguous data, optimized with ld.global.cs (cache-streaming) PTX hint.
        // This hint is for data that is read once, reducing L1 cache pollution.
        const unsigned int threads_per_group = 16;
        const unsigned int group_id = threadIdx.x / threads_per_group;
        const unsigned int local_lane_id = threadIdx.x % threads_per_group;
        const unsigned int groups_per_block = blockDim.x / threads_per_group;

        for (int output_idx = blockIdx.x * groups_per_block + group_id;
             output_idx < total_outputs;
             output_idx += gridDim.x * groups_per_block) {

            const float* input_ptr = input + output_idx * reduce_dim_size;
            
            const int vec_size = 2;
            const int reduce_dim_vec = reduce_dim_size / vec_size;
            
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const int unroll_factor = 4;
            const int loop_stride = threads_per_group * unroll_factor;
            
            int i = local_lane_id;
            const int unrolled_limit = reduce_dim_vec - (reduce_dim_vec % unroll_factor);
            
            for (; i < unrolled_limit; i += loop_stride) {
                float f1, f2;
                // Use ld.global.cs for streaming loads
                const float2* ptr_a = &reinterpret_cast<const float2*>(input_ptr)[i];
                asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(f1), "=f"(f2) : "l"(ptr_a));
                s0 += f1 + f2;
                
                const float2* ptr_b = &reinterpret_cast<const float2*>(input_ptr)[i + threads_per_group];
                asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(f1), "=f"(f2) : "l"(ptr_b));
                s1 += f1 + f2;
                
                const float2* ptr_c = &reinterpret_cast<const float2*>(input_ptr)[i + 2 * threads_per_group];
                asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(f1), "=f"(f2) : "l"(ptr_c));
                s2 += f1 + f2;

                const float2* ptr_d = &reinterpret_cast<const float2*>(input_ptr)[i + 3 * threads_per_group];
                asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(f1), "=f"(f2) : "l"(ptr_d));
                s3 += f1 + f2;
            }
            
            // Process remaining vectorized elements
            for (; i < reduce_dim_vec; i += threads_per_group) {
                float f1, f2;
                const float2* ptr = &reinterpret_cast<const float2*>(input_ptr)[i];
                asm("ld.global.cs.v2.f32 {%0, %1}, [%2];" : "=f"(f1), "=f"(f2) : "l"(ptr));
                s0 += f1 + f2;
            }

            float thread_sum = s0 + s1 + s2 + s3;

            // Process remainder elements that don't fit into a float2
            const int remainder_start = reduce_dim_vec * vec_size;
            for (int j = remainder_start + local_lane_id; j < reduce_dim_size; j += threads_per_group) {
                thread_sum += input_ptr[j]; // Remainder too small to benefit from PTX
            }

            float group_sum = halfWarpReduceSumXOR(thread_sum);

            if (local_lane_id == 0) {
                output[output_idx] = group_sum * inv_reduce_dim_size;
            }
        }

    } else {
        // Path 2: Non-contiguous data, optimized with ld.global.cg (cache-global) PTX hint.
        // This hint bypasses L1, reducing cache pollution for strided access patterns.
        constexpr int ROWS = 8;
        constexpr int COLS = 32;
        extern __shared__ float tile[];
        
        const int tx = threadIdx.x % COLS;
        const int ty = threadIdx.x / COLS;

        for (int base_output_idx = blockIdx.x * COLS;
             base_output_idx < total_outputs;
             base_output_idx += gridDim.x * COLS) {
            
            const unsigned int output_idx = base_output_idx + tx;
            if (output_idx >= total_outputs) continue;

            const int outer_idx = output_idx / inner_size;
            const int inner_idx = output_idx % inner_size;
            const float* input_ptr = input + outer_idx * reduce_dim_size * inner_size + inner_idx;

            float thread_sum = 0.0f;
            int i = ty;
            const int unrolled_limit = reduce_dim_size - (reduce_dim_size % 4);
            for (; i < unrolled_limit; i += ROWS * 4) {
                 float temp;
                 // Use ld.global.cg to bypass L1 cache
                 const float* ptr_a = &input_ptr[i * inner_size];
                 asm("ld.global.cg.f32 %0, [%1];" : "=f"(temp) : "l"(ptr_a));
                 thread_sum += temp;

                 const float* ptr_b = &input_ptr[(i + ROWS) * inner_size];
                 asm("ld.global.cg.f32 %0, [%1];" : "=f"(temp) : "l"(ptr_b));
                 thread_sum += temp;
                 
                 const float* ptr_c = &input_ptr[(i + 2*ROWS) * inner_size];
                 asm("ld.global.cg.f32 %0, [%1];" : "=f"(temp) : "l"(ptr_c));
                 thread_sum += temp;
                 
                 const float* ptr_d = &input_ptr[(i + 3*ROWS) * inner_size];
                 asm("ld.global.cg.f32 %0, [%1];" : "=f"(temp) : "l"(ptr_d));
                 thread_sum += temp;
            }
            for (; i < reduce_dim_size; i += ROWS) {
                float temp;
                const float* ptr = &input_ptr[i * inner_size];
                asm("ld.global.cg.f32 %0, [%1];" : "=f"(temp) : "l"(ptr));
                thread_sum += temp;
            }

            tile[threadIdx.x] = thread_sum;
            __syncthreads();

            // Manually unrolled reduction in shared memory
            if (ty == 0) {
                float final_sum = tile[tx + 0 * COLS] + tile[tx + 1 * COLS] +
                                  tile[tx + 2 * COLS] + tile[tx + 3 * COLS] +
                                  tile[tx + 4 * COLS] + tile[tx + 5 * COLS] +
                                  tile[tx + 6 * COLS] + tile[tx + 7 * COLS];
                output[output_idx] = final_sum * inv_reduce_dim_size;
            }
            __syncthreads(); 
        }
    }
}

// C++ wrapper function that dispatches to the optimal CUDA kernel path.
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto input_sizes = input.sizes();
    int ndim = input_sizes.size();
    dim = (dim < 0) ? (dim + ndim) : dim;
    TORCH_CHECK(ndim > 0 && dim >= 0 && dim < ndim, "Invalid dimension provided");

    const int reduce_dim_size = input_sizes[dim];

    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_sizes.push_back(input_sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());

    if (reduce_dim_size == 0) {
        output.fill_(std::numeric_limits<float>::quiet_NaN());
        return output;
    }
    if (reduce_dim_size == 1) {
        output.copy_(input.squeeze(dim));
        return output;
    }

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= input_sizes[i];
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= input_sizes[i];
    
    const int total_outputs = outer_size * inner_size;
    if (total_outputs == 0) return output;
    
    const float inv_reduce_dim_size = 1.0f / static_cast<float>(reduce_dim_size);

    if (inner_size == 1) {
        // Path 1: Contiguous reduction.
        const int block_dim_x = 256;
        const int threads_per_group = 16;
        const int groups_per_block = block_dim_x / threads_per_group;
        const int grid_dim_x = std::min((total_outputs + groups_per_block - 1) / groups_per_block, 512);
        
        grid_stride_reduction_kernel<true><<<grid_dim_x, block_dim_x>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            reduce_dim_size, outer_size, inner_size, total_outputs, inv_reduce_dim_size
        );
    } else {
        // Path 2: Non-contiguous reduction.
        const int block_dim_x = 256; // 8 rows x 32 cols
        const int outputs_per_block = 32;
        const int grid_dim_x = std::min((total_outputs + outputs_per_block - 1) / outputs_per_block, 512);
        const int shared_mem_size = block_dim_x * sizeof(float);
        
        grid_stride_reduction_kernel<false><<<grid_dim_x, block_dim_x, shared_mem_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            reduce_dim_size, outer_size, inner_size, total_outputs, inv_reduce_dim_size
        );
    }
    
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

# The C++ source providing the function signature for the CUDA wrapper.
cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);
"""

# Use torch's JIT compiler to build the custom CUDA extension.
mean_reduction_module = load_inline(
    name='ptx_caching_reduction',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['mean_reduction_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    This model tests the hypothesis that explicitly controlling memory caching behavior
    at the instruction level can improve performance for different memory access patterns.
    It builds upon the state-of-the-art two-path grid-stride reduction kernel.

    The key innovation is the use of PTX inline assembly to specify caching hints for
    global memory loads:
    
    1.  For the contiguous (bandwidth-bound) path, `ld.global.cs` (cache-streaming)
        is used. This advises the hardware that the data is likely to be read only once,
        reducing L1 cache pollution and potentially improving effective memory bandwidth.

    2.  For the non-contiguous (latency-bound) path, `ld.global.cg` (cache-global)
        is used. This bypasses the L1 cache, which is beneficial for strided access
        patterns that would otherwise thrash the cache. This strategy has shown
        promise in prior experiments.

    By tailoring the caching strategy to the specific access pattern of each kernel path,
    this experiment aims to eke out further performance from an already highly optimized
    kernel.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.dim = num_features
        self.mean_reduction = mean_reduction_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the model.
        The input tensor x is processed by the custom CUDA kernel, which now uses
        path-specific PTX caching hints for memory loads.
        """
        return self.mean_reduction.mean_reduction_cuda(x.cuda(), self.dim)