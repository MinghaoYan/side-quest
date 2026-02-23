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

# Set CUDA architecture for A100 to enable Tensor Core operations and optimizations
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernel for fused Linear, LayerNorm, and Add-Mul.
# This version evolves the SOTA's `2x16x2` parallelism model (where each
# 16-thread sub-warp computes two output features) to a `2x16x4` model.
# Each sub-warp now computes four adjacent output features, doubling the
# arithmetic intensity again. The goal is to maximize the reuse of the
# input vector cached in shared memory, pushing the compute-to-memory
# ratio higher to better hide the latency of fetching the weight matrix
# from global memory. This is a high-risk, high-reward strategy that
# bets the performance gain from improved latency hiding will outweigh the
# significant increase in register pressure from maintaining four separate
# accumulators per thread.
fused_op_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <c10/cuda/CUDAException.h>

// Use a power of two for block size for optimal performance
constexpr int THREADS_PER_BLOCK = 256;
// WARP_SIZE is a fixed hardware constant, 32 on NVIDIA GPUs
constexpr int WARP_SIZE = 32;
// SUB_WARP_SIZE for the parallelism model
constexpr int SUB_WARP_SIZE = 16;

// __device__ function for performing a sum-reduction within a warp using shuffle-down instructions.
// This is significantly faster than using shared memory for intra-warp reductions.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// __device__ function for performing a sum-reduction within a 16-thread sub-warp.
__device__ __forceinline__ float sub_warp_reduce_sum(float val) {
    // A mask of 0xFFFFFFFF is safe for sub-warp shuffles as inactive lanes will not corrupt the result.
    for (int offset = SUB_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_linear_layernorm_add_mul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const int B,  // Batch size
    const int M, // in_features
    const int N, // out_features
    const float eps
) {
    // Shared memory layout
    extern __shared__ float smem[];
    float* s_input = smem;
    float* linear_out = &smem[M];
    // Allocate space for two reductions (sum and sum_sq) per warp
    const int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    float* reduce_smem = &smem[M + N]; 

    // --- Outer Grid-Stride Loop for Batch Processing ---
    for (int b_idx = blockIdx.x; b_idx < B; b_idx += gridDim.x) {
        const int t_idx = threadIdx.x;
        
        // --- Step 1: Vectorized Caching of Input Vector in Shared Memory ---
        const float4* input_row_vec = reinterpret_cast<const float4*>(input + b_idx * M);
        float4* s_input_vec = reinterpret_cast<float4*>(s_input);
        for (int i = t_idx; i < M / 4; i += THREADS_PER_BLOCK) {
            s_input_vec[i] = input_row_vec[i];
        }
        __syncthreads();

        // --- Step 2: Fused Matrix-Vector Multiplication (2x16x4 Model) ---
        // Each 32-thread warp computes EIGHT output features.
        // Each 16-thread sub-warp computes FOUR output features.
        const int warp_id = t_idx / WARP_SIZE;
        const int lane_id = t_idx % WARP_SIZE;
        const int sub_warp_lane_id = lane_id % SUB_WARP_SIZE;
        
        const int outputs_per_warp = 8;
        const int outputs_per_block = num_warps * outputs_per_warp;
        
        for (int j_base = warp_id * outputs_per_warp; j_base < N; j_base += outputs_per_block) {
            float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
            
            // Each sub-warp handles 4 outputs.
            const int sub_warp_output_base = (lane_id / SUB_WARP_SIZE) * 4;
            const int j1 = j_base + sub_warp_output_base;
            const int j2 = j1 + 1;
            const int j3 = j1 + 2;
            const int j4 = j1 + 3;
            
            // Since N is divisible by 4, this check is uniform across the sub-warp.
            // All threads in the sub-warp either compute all 4 outputs or none for this chunk.
            if (j4 < N) {
                const float4* weight_row_vec1 = reinterpret_cast<const float4*>(weight + j1 * M);
                const float4* weight_row_vec2 = reinterpret_cast<const float4*>(weight + j2 * M);
                const float4* weight_row_vec3 = reinterpret_cast<const float4*>(weight + j3 * M);
                const float4* weight_row_vec4 = reinterpret_cast<const float4*>(weight + j4 * M);
                
                for (int k_vec = sub_warp_lane_id; k_vec < M / 4; k_vec += SUB_WARP_SIZE) {
                    const float4 s_val = s_input_vec[k_vec];
                    const float4 w1_val = weight_row_vec1[k_vec];
                    const float4 w2_val = weight_row_vec2[k_vec];
                    const float4 w3_val = weight_row_vec3[k_vec];
                    const float4 w4_val = weight_row_vec4[k_vec];
                    sum1 += w1_val.x * s_val.x + w1_val.y * s_val.y + w1_val.z * s_val.z + w1_val.w * s_val.w;
                    sum2 += w2_val.x * s_val.x + w2_val.y * s_val.y + w2_val.z * s_val.z + w2_val.w * s_val.w;
                    sum3 += w3_val.x * s_val.x + w3_val.y * s_val.y + w3_val.z * s_val.z + w3_val.w * s_val.w;
                    sum4 += w4_val.x * s_val.x + w4_val.y * s_val.y + w4_val.z * s_val.z + w4_val.w * s_val.w;
                }
            }
            
            sum1 = sub_warp_reduce_sum(sum1);
            sum2 = sub_warp_reduce_sum(sum2);
            sum3 = sub_warp_reduce_sum(sum3);
            sum4 = sub_warp_reduce_sum(sum4);

            if (sub_warp_lane_id == 0) {
                if (j1 < N) linear_out[j1] = sum1 + bias[j1];
                if (j2 < N) linear_out[j2] = sum2 + bias[j2];
                if (j3 < N) linear_out[j3] = sum3 + bias[j3];
                if (j4 < N) linear_out[j4] = sum4 + bias[j4];
            }
        }
        __syncthreads();

        // --- Step 3: LayerNorm - Single-Pass Mean and Variance Calculation ---
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        const float4* linear_out_vec = reinterpret_cast<const float4*>(linear_out);

        for (int j = t_idx; j < N / 4; j += THREADS_PER_BLOCK) {
            const float4 val = linear_out_vec[j];
            local_sum += val.x + val.y + val.z + val.w;
            local_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        }

        float warp_total_sum = warp_reduce_sum(local_sum);
        float warp_total_sum_sq = warp_reduce_sum(local_sum_sq);

        if (lane_id == 0) {
            reduce_smem[warp_id] = warp_total_sum;
            reduce_smem[warp_id + num_warps] = warp_total_sum_sq;
        }
        __syncthreads();
        
        float final_sum = 0.0f;
        float final_sum_sq = 0.0f;
        if (warp_id == 0) {
            if (lane_id < num_warps) {
                final_sum = reduce_smem[lane_id];
                final_sum_sq = reduce_smem[lane_id + num_warps];
            }
            final_sum = warp_reduce_sum(final_sum);
            final_sum_sq = warp_reduce_sum(final_sum_sq);
        }
        
        float mean, rstd;
        if (t_idx == 0) {
            mean = final_sum / N;
            float var = (final_sum_sq / N) - (mean * mean);
            rstd = rsqrtf(var + eps);
            reduce_smem[0] = mean;
            reduce_smem[1] = rstd;
        }
        __syncthreads();
        
        mean = reduce_smem[0];
        rstd = reduce_smem[1];
        
        // --- Step 4: Apply Normalization, Fused Add-Mul, and Write to Global Memory ---
        float4* output_row_vec = reinterpret_cast<float4*>(output + b_idx * N);
        const float4* y_row_vec = reinterpret_cast<const float4*>(y + b_idx * N);
        const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
        const float4* beta_vec = reinterpret_cast<const float4*>(beta);

        for (int j = t_idx; j < N / 4; j += THREADS_PER_BLOCK) {
            const float4 linear_val = linear_out_vec[j];
            const float4 gamma_val = gamma_vec[j];
            const float4 beta_val = beta_vec[j];
            const float4 y_val = y_row_vec[j];
            
            float4 layernorm_out;
            layernorm_out.x = (linear_val.x - mean) * rstd * gamma_val.x + beta_val.x;
            layernorm_out.y = (linear_val.y - mean) * rstd * gamma_val.y + beta_val.y;
            layernorm_out.z = (linear_val.z - mean) * rstd * gamma_val.z + beta_val.z;
            layernorm_out.w = (linear_val.w - mean) * rstd * gamma_val.w + beta_val.w;
            
            float4 final_out;
            final_out.x = (layernorm_out.x + y_val.x) * y_val.x;
            final_out.y = (layernorm_out.y + y_val.y) * y_val.y;
            final_out.z = (layernorm_out.z + y_val.z) * y_val.z;
            final_out.w = (layernorm_out.w + y_val.w) * y_val.w;
            
            output_row_vec[j] = final_out;
        }
        __syncthreads();
    }
}

torch::Tensor fused_op_forward(
    torch::Tensor input,
    torch::Tensor y,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps_double
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "Y must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "Y must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    const int B = input.size(0);
    const int M = input.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(M % 4 == 0, "Input features (M) must be divisible by 4 for float4 vectorization");
    TORCH_CHECK(N % 4 == 0, "Output features (N) must be divisible by 4 for float4 vectorization");

    const float eps = static_cast<float>(eps_double);
    auto output = torch::empty({B, N}, input.options());

    // Heuristic for grid size to ensure full GPU utilization via the grid-stride loop
    const int grid_size = 256;
    const dim3 blocks(grid_size);
    const dim3 threads(THREADS_PER_BLOCK);
    const int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    const size_t smem_size = (M + N + 2 * num_warps) * sizeof(float);

    fused_linear_layernorm_add_mul_kernel<<<blocks, threads, smem_size>>>(
        input.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        M,
        N,
        eps
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_forward(torch::Tensor input, torch::Tensor y, torch::Tensor weight, torch::Tensor bias, torch::Tensor gamma, torch::Tensor beta, double eps);"

# Use JIT compilation to build the custom CUDA/C++ operator
fused_op_intensity_scaling = load_inline(
    name='fused_op_intensity_scaling',
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=['fused_op_forward'],
    verbose=True,
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    """
    This optimized model pushes the arithmetic intensity of the SOTA kernel
    even further. The previous `2x16x2` sub-warp model, where each 16-thread
    sub-warp computes two output features, is now extended to a `2x16x4`
    model. Each sub-warp calculates four adjacent output features, resulting
    in eight outputs per warp. This quadruples the work done per cached input
    vector relative to the baseline, aiming to completely hide the latency
    of weight matrix reads. This approach aggressively tests the limits of
    register file capacity and instruction-level parallelism on the A100 GPU.
    """
    def __init__(self, in_features, out_features, eps=1e-5):
        super(ModelNew, self).__init__()
        if in_features % 4 != 0 or out_features % 4 != 0:
            raise ValueError("in_features and out_features must be divisible by 4 for ModelNew.")
        
        self.linear = nn.Linear(in_features, out_features)
        self.layernorm = nn.LayerNorm(out_features, eps=eps)

    def forward(self, x, y):
        return fused_op_intensity_scaling.fused_op_forward(
            x,
            y,
            self.linear.weight,
            self.linear.bias,
            self.layernorm.weight,
            self.layernorm.bias,
            self.layernorm.eps
        )