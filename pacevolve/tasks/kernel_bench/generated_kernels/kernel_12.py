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

# Set CUDA architecture for A100 (Compute Capability 8.0)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void mean_reduction_peeled_unrolled4x_128x16(const float4* __restrict__ input, float* __restrict__ output,
                                                         int reduce_dim_size, long outer_size, long inner_size,
                                                         float inv_reduce_dim_size) {
    // --- Kernel Configuration ---
    // This kernel aggressively optimizes for instruction-level parallelism (ILP)
    // to hide memory latency by implementing a 4x unrolled main loop. It builds
    // upon the successful peeled-loop, hybrid-reduction architecture.
    // 1. Tiling & Work-per-thread: Maintains the proven 128x16 data tile and
    //    256-thread (64x4) block to maximize arithmetic intensity.
    // 2. Loop Transformation (New Experiment for Idea 7): The main accumulation loop is
    //    unrolled by a factor of 4. Four independent float4 accumulator registers are used
    //    per thread to break data dependencies and give the instruction scheduler maximum
    //    flexibility. This is a direct attempt to replicate prior successes where
    //    aggressive unrolling provided significant speedups by effectively hiding
    //    the latency of global memory loads.
    // 3. Loop Peeling (Idea 6): A branch-free main loop processes full 512-row chunks (4 * 128).
    //    A flexible epilogue handles the remaining full tiles (0-3) and a final partial tile.
    // 4. Hybrid Reduction (Idea 5): The state-of-the-art reduction strategy is unchanged,
    //    using shared memory for inter-warp reduction and warp shuffles for intra-warp reduction.

    constexpr int DATA_TILE_H = 128;
    constexpr int THREAD_TILE_H = 64;
    constexpr int TILE_W_VEC = 4;
    constexpr int VEC_SIZE = 4;

    __shared__ float4 smem[THREAD_TILE_H][TILE_W_VEC];

    const int tx = threadIdx.x % TILE_W_VEC;
    const int ty = threadIdx.x / TILE_W_VEC;

    const long output_vec_idx = (long)blockIdx.x * TILE_W_VEC + tx;
    const long total_inner_vecs = inner_size / VEC_SIZE;
    const long num_output_vecs = outer_size * total_inner_vecs;

    if (output_vec_idx >= num_output_vecs) {
        return;
    }

    const long o_idx = output_vec_idx / total_inner_vecs;
    const long i_idx_vec = output_vec_idx % total_inner_vecs;
    const float4* input_slice_ptr = &input[o_idx * reduce_dim_size * total_inner_vecs + i_idx_vec];

    float4 thread_sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 thread_sum2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 thread_sum3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 thread_sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // --- Accumulation Stage ---

    // 1. Peeled Main Loop (4x Unrolled, Branch-Free)
    const int unroll_stride = DATA_TILE_H * 4;
    const int num_unrolled_iters = reduce_dim_size / unroll_stride;
    const int peeled_end = num_unrolled_iters * unroll_stride;

    for (int i = 0; i < peeled_end; i += unroll_stride) {
        // Unrolled iteration 1
        const float4 v1a = input_slice_ptr[(i + ty) * total_inner_vecs];
        const float4 v2a = input_slice_ptr[(i + ty + THREAD_TILE_H) * total_inner_vecs];
        thread_sum1.x += v1a.x + v2a.x; thread_sum1.y += v1a.y + v2a.y; thread_sum1.z += v1a.z + v2a.z; thread_sum1.w += v1a.w + v2a.w;

        // Unrolled iteration 2
        const float4 v1b = input_slice_ptr[(i + DATA_TILE_H + ty) * total_inner_vecs];
        const float4 v2b = input_slice_ptr[(i + DATA_TILE_H + ty + THREAD_TILE_H) * total_inner_vecs];
        thread_sum2.x += v1b.x + v2b.x; thread_sum2.y += v1b.y + v2b.y; thread_sum2.z += v1b.z + v2b.z; thread_sum2.w += v1b.w + v2b.w;
        
        // Unrolled iteration 3
        const float4 v1c = input_slice_ptr[(i + DATA_TILE_H * 2 + ty) * total_inner_vecs];
        const float4 v2c = input_slice_ptr[(i + DATA_TILE_H * 2 + ty + THREAD_TILE_H) * total_inner_vecs];
        thread_sum3.x += v1c.x + v2c.x; thread_sum3.y += v1c.y + v2c.y; thread_sum3.z += v1c.z + v2c.z; thread_sum3.w += v1c.w + v2c.w;

        // Unrolled iteration 4
        const float4 v1d = input_slice_ptr[(i + DATA_TILE_H * 3 + ty) * total_inner_vecs];
        const float4 v2d = input_slice_ptr[(i + DATA_TILE_H * 3 + ty + THREAD_TILE_H) * total_inner_vecs];
        thread_sum4.x += v1d.x + v2d.x; thread_sum4.y += v1d.y + v2d.y; thread_sum4.z += v1d.z + v2d.z; thread_sum4.w += v1d.w + v2d.w;
    }

    // Combine accumulators
    thread_sum1.x += thread_sum2.x + thread_sum3.x + thread_sum4.x;
    thread_sum1.y += thread_sum2.y + thread_sum3.y + thread_sum4.y;
    thread_sum1.z += thread_sum2.z + thread_sum3.z + thread_sum4.z;
    thread_sum1.w += thread_sum2.w + thread_sum3.w + thread_sum4.w;

    // 2. Epilogue for Remainder
    int remainder_start = peeled_end;
    // Handle up to three potential full tiles
    if (remainder_start + DATA_TILE_H <= reduce_dim_size) {
        const float4 v1 = input_slice_ptr[(remainder_start + ty) * total_inner_vecs];
        const float4 v2 = input_slice_ptr[(remainder_start + ty + THREAD_TILE_H) * total_inner_vecs];
        thread_sum1.x += v1.x + v2.x; thread_sum1.y += v1.y + v2.y; thread_sum1.z += v1.z + v2.z; thread_sum1.w += v1.w + v2.w;
        remainder_start += DATA_TILE_H;
    }
    if (remainder_start + DATA_TILE_H <= reduce_dim_size) {
        const float4 v1 = input_slice_ptr[(remainder_start + ty) * total_inner_vecs];
        const float4 v2 = input_slice_ptr[(remainder_start + ty + THREAD_TILE_H) * total_inner_vecs];
        thread_sum1.x += v1.x + v2.x; thread_sum1.y += v1.y + v2.y; thread_sum1.z += v1.z + v2.z; thread_sum1.w += v1.w + v2.w;
        remainder_start += DATA_TILE_H;
    }
    if (remainder_start + DATA_TILE_H <= reduce_dim_size) {
        const float4 v1 = input_slice_ptr[(remainder_start + ty) * total_inner_vecs];
        const float4 v2 = input_slice_ptr[(remainder_start + ty + THREAD_TILE_H) * total_inner_vecs];
        thread_sum1.x += v1.x + v2.x; thread_sum1.y += v1.y + v2.y; thread_sum1.z += v1.z + v2.z; thread_sum1.w += v1.w + v2.w;
        remainder_start += DATA_TILE_H;
    }
    
    // Handle final partial tile
    if (remainder_start < reduce_dim_size) {
        int idx1 = remainder_start + ty;
        if (idx1 < reduce_dim_size) {
            const float4 v1 = input_slice_ptr[idx1 * total_inner_vecs];
            thread_sum1.x += v1.x; thread_sum1.y += v1.y; thread_sum1.z += v1.z; thread_sum1.w += v1.w;
        }
        int idx2 = remainder_start + ty + THREAD_TILE_H;
        if (idx2 < reduce_dim_size) {
            const float4 v2 = input_slice_ptr[idx2 * total_inner_vecs];
            thread_sum1.x += v2.x; thread_sum1.y += v2.y; thread_sum1.z += v2.z; thread_sum1.w += v2.w;
        }
    }

    // --- Reduction Stage (unchanged) ---
    smem[ty][tx] = thread_sum1;
    __syncthreads();

    if (ty < 8) {
        float4 psum = smem[ty][tx];
        #pragma unroll
        for (int i = 1; i < 8; ++i) {
            float4 other = smem[ty + i * 8][tx];
            psum.x += other.x; psum.y += other.y; psum.z += other.z; psum.w += other.w;
        }
        smem[ty][tx] = psum;
    }
    __syncthreads();

    if (ty < 8) {
        float4 final_sum = smem[ty][tx];

        final_sum.x += __shfl_down_sync(0xffffffff, final_sum.x, TILE_W_VEC);
        final_sum.y += __shfl_down_sync(0xffffffff, final_sum.y, TILE_W_VEC);
        final_sum.z += __shfl_down_sync(0xffffffff, final_sum.z, TILE_W_VEC);
        final_sum.w += __shfl_down_sync(0xffffffff, final_sum.w, TILE_W_VEC);

        final_sum.x += __shfl_down_sync(0xffffffff, final_sum.x, TILE_W_VEC * 2);
        final_sum.y += __shfl_down_sync(0xffffffff, final_sum.y, TILE_W_VEC * 2);
        final_sum.z += __shfl_down_sync(0xffffffff, final_sum.z, TILE_W_VEC * 2);
        final_sum.w += __shfl_down_sync(0xffffffff, final_sum.w, TILE_W_VEC * 2);
        
        final_sum.x += __shfl_down_sync(0xffffffff, final_sum.x, TILE_W_VEC * 4);
        final_sum.y += __shfl_down_sync(0xffffffff, final_sum.y, TILE_W_VEC * 4);
        final_sum.z += __shfl_down_sync(0xffffffff, final_sum.z, TILE_W_VEC * 4);
        final_sum.w += __shfl_down_sync(0xffffffff, final_sum.w, TILE_W_VEC * 4);

        if (ty == 0) {
            if (reduce_dim_size > 0) {
                long base_output_idx = output_vec_idx * VEC_SIZE;
                long total_outputs = outer_size * inner_size;

                if (base_output_idx < total_outputs) {
                    output[base_output_idx] = final_sum.x * inv_reduce_dim_size;
                }
                if (base_output_idx + 1 < total_outputs) {
                    output[base_output_idx + 1] = final_sum.y * inv_reduce_dim_size;
                }
                if (base_output_idx + 2 < total_outputs) {
                    output[base_output_idx + 2] = final_sum.z * inv_reduce_dim_size;
                }
                if (base_output_idx + 3 < total_outputs) {
                    output[base_output_idx + 3] = final_sum.w * inv_reduce_dim_size;
                }
            }
        }
    }
}


torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto input_sizes = input.sizes();
    int ndim = input_sizes.size();
    dim = dim < 0 ? dim + ndim : dim;
    
    int reduce_dim_size = input_sizes[dim];
    
    long outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input_sizes[i];
    }
    
    long inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= input_sizes[i];
    }
    
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_sizes.push_back(input_sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    const long num_outputs = outer_size * inner_size;
    if (num_outputs == 0 || reduce_dim_size == 0) {
        if (reduce_dim_size == 0) output.fill_(NAN);
        return output;
    }

    TORCH_CHECK(inner_size % 4 == 0, "inner_size must be a multiple of 4 for vectorized kernel");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr<float>()) % 16 == 0, 
                "Input tensor data pointer must be 16-byte aligned for float4 loads");

    const float inv_reduce_dim_size = 1.0f / reduce_dim_size;
    
    constexpr int TILE_W_FLOAT = 16;
    constexpr int TILE_W_VEC = TILE_W_FLOAT / 4;
    constexpr int THREAD_TILE_H = 64;
    
    const int threads_per_block = TILE_W_VEC * THREAD_TILE_H;
    const long num_output_vecs = num_outputs / 4;
    const int blocks = (num_output_vecs + TILE_W_VEC - 1) / TILE_W_VEC;
    
    mean_reduction_peeled_unrolled4x_128x16<<<blocks, threads_per_block>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        output.data_ptr<float>(),
        reduce_dim_size,
        outer_size,
        inner_size,
        inv_reduce_dim_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return output;
}
"""

mean_reduction_cpp_source = """
#include <torch/extension.h>
#include <vector>

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);
"""

mean_reduction = load_inline(
    name='mean_reduction_peeled_unrolled4x_128x16',
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=['mean_reduction_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.dim = num_features
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)