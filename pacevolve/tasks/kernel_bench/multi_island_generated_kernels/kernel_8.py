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

# Define the custom CUDA kernel source for Fused Linear + ReLU.
# This kernel builds on the 1x4 block-tiling strategy, which maximizes input data
# reuse. A 128-thread block (4 warps) computes a 1x4 tile of the output matrix.
#
# To further enhance instruction-level parallelism (ILP), this version uses four
# independent accumulators (sum_a, sum_b, sum_c, sum_d). This breaks the dependency
# chain on registers more effectively than a dual-accumulator strategy. To manage
# the increased register pressure from the extra accumulators, the loop unroll factor
# is reduced from 8x to 4x. This experiment tests whether a wider but shorter
# ILP strategy can outperform a longer, narrower one.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// A union for type-punning between float4 and longlong2 to optimize memory bandwidth.
union float4_longlong2_union {
    float4 f4;
    longlong2 ll2;
};

__global__ void linear_relu_fma_1x4_tile_4x_unroll_4acc(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int input_size,
    int output_size)
{
    // Tiling dimensions: Each block computes a 1x4 tile in the output matrix.
    const int TILE_M = 1;
    const int TILE_N = 4;

    // Block-level indices
    const int batch_idx = blockIdx.y; // Each block processes one row from the batch
    const int block_tile_n = blockIdx.x * TILE_N;

    // Identify warp and lane
    const int warp_id = threadIdx.y; // 0..3
    const int lane_id = threadIdx.x; // 0..31

    // Map warps to the 1x4 output tile. Each warp handles one output element.
    const int output_idx = block_tile_n + warp_id;

    // Boundary check for the entire tile
    if (batch_idx >= batch_size || output_idx >= output_size) {
        return;
    }

    // Pointers for the specific input vector and weight row this warp processes
    const float* input_vec = input + batch_idx * input_size;
    const float* weight_row = weight + output_idx * input_size;

    // Use four accumulators to increase instruction-level parallelism
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    float sum_c = 0.0f;
    float sum_d = 0.0f;

    const int vectorized_limit = input_size / 4;
    // Stride for 4x unroll (32 threads * 4 unrolls)
    const int stride = blockDim.x * 4; 

    if (vectorized_limit > 0) {
        const longlong2* input_ll2 = reinterpret_cast<const longlong2*>(input_vec);
        const longlong2* weight_ll2 = reinterpret_cast<const longlong2*>(weight_row);
        
        float4_longlong2_union in_val, wt_val;

        // 4x unrolled loop with four interleaved accumulators
        for (int k = lane_id; k < vectorized_limit; k += stride) {
            // Unroll 1 -> sum_a
            in_val.ll2 = input_ll2[k];
            wt_val.ll2 = weight_ll2[k];
            sum_a += in_val.f4.x * wt_val.f4.x + in_val.f4.y * wt_val.f4.y + in_val.f4.z * wt_val.f4.z + in_val.f4.w * wt_val.f4.w;

            // Unroll 2 -> sum_b
            if (k + blockDim.x < vectorized_limit) {
                in_val.ll2 = input_ll2[k + blockDim.x];
                wt_val.ll2 = weight_ll2[k + blockDim.x];
                sum_b += in_val.f4.x * wt_val.f4.x + in_val.f4.y * wt_val.f4.y + in_val.f4.z * wt_val.f4.z + in_val.f4.w * wt_val.f4.w;
            }

            // Unroll 3 -> sum_c
            if (k + blockDim.x * 2 < vectorized_limit) {
                in_val.ll2 = input_ll2[k + blockDim.x * 2];
                wt_val.ll2 = weight_ll2[k + blockDim.x * 2];
                sum_c += in_val.f4.x * wt_val.f4.x + in_val.f4.y * wt_val.f4.y + in_val.f4.z * wt_val.f4.z + in_val.f4.w * wt_val.f4.w;
            }

            // Unroll 4 -> sum_d
            if (k + blockDim.x * 3 < vectorized_limit) {
                in_val.ll2 = input_ll2[k + blockDim.x * 3];
                wt_val.ll2 = weight_ll2[k + blockDim.x * 3];
                sum_d += in_val.f4.x * wt_val.f4.x + in_val.f4.y * wt_val.f4.y + in_val.f4.z * wt_val.f4.z + in_val.f4.w * wt_val.f4.w;
            }
        }
    }
    
    float sum = sum_a + sum_b + sum_c + sum_d;

    // Handle remainder elements if input_size is not a multiple of 4
    for (int i = vectorized_limit * 4 + lane_id; i < input_size; i += blockDim.x) {
        sum += input_vec[i] * weight_row[i];
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Thread 0 of each warp writes the final result
    if (lane_id == 0) {
        sum += bias[output_idx];
        output[batch_idx * output_size + output_idx] = fmaxf(0.0f, sum);
    }
}

// C++ wrapper to be called from Python
torch::Tensor linear_relu_cuda_1x4_4x_4acc(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = weight.size(0);

    auto output = torch::empty({batch_size, output_size}, input.options());

    // Kernel Launch Configuration
    const int TILE_N = 4;
    const int warps_per_block = 4;

    dim3 threads(32, warps_per_block); // 128 threads per block
    dim3 blocks(
        (output_size + TILE_N - 1) / TILE_N,
        batch_size
    );
    
    linear_relu_fma_1x4_tile_4x_unroll_4acc<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

cpp_source = "torch::Tensor linear_relu_cuda_1x4_4x_4acc(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Inline compilation using torch.utils.cpp_extension
fused_op_1x4_4x_4acc = load_inline(
    name='fused_op_1x4_4x_4acc',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['linear_relu_cuda_1x4_4x_4acc'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        self.layers = nn.ModuleList()
        current_input_size = input_size
        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
        self.final_linear = nn.Linear(current_input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # Use the custom 1x4 tile, 4x unroll, 4-accumulator kernel
            x = fused_op_1x4_4x_4acc.linear_relu_cuda_1x4_4x_4acc(x, layer.weight, layer.bias)
        x = self.final_linear(x)
        return x