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

# Define the custom CUDA kernel for a fused, two-pass Layer Normalization.
# This version modifies the shared memory layout in the statistics kernel
# to use a struct-of-arrays (SoA) to array-of-structs (AoS) transformation.
# By interleaving sum and sum_sq data, we aim to improve cache locality
# during the shared memory reduction phase.
layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector_types.h> // For float4

// Define a struct to hold sum and sum-of-squares pairs.
// This facilitates an Array-of-Structs (AoS) layout in shared memory.
struct Sums {
    float sum;
    float sum_sq;
};

// Define a constant for the number of blocks to launch per channel for the partial reduction.
constexpr int BLOCKS_PER_CHANNEL = 64;
// Define unroll factor for main compute loops to increase ILP.
constexpr int UNROLL_FACTOR = 16;

// Kernel 1: Fused statistics calculation using block-level reduction and atomics.
// This version uses a struct to improve shared memory locality during reduction.
__global__ void fused_atomic_stats_kernel(
    const float4* __restrict__ input,
    float* __restrict__ stats, // Output: (C, 2) -> {sum, sum_sq}, must be zero-initialized
    int N, int C, int H, int W_div_4) {

    const int total_elements_per_channel_div_4 = N * H * W_div_4;

    const int global_block_idx = blockIdx.x;
    const int feature_idx = global_block_idx / BLOCKS_PER_CHANNEL;
    if (feature_idx >= C) {
        return;
    }

    const int block_in_channel_idx = global_block_idx % BLOCKS_PER_CHANNEL;
    
    // Use an array of structs (AoS) for shared memory.
    extern __shared__ Sums sdata[];

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    
    const int HW_div_4 = H * W_div_4;
    const int CHW_div_4 = C * HW_div_4;
    const int grid_stride = BLOCKS_PER_CHANNEL * blockDim.x;
    const int unrolled_stride = grid_stride * UNROLL_FACTOR;

    int i = block_in_channel_idx * blockDim.x + threadIdx.x;

    // Unrolled main loop with factor 16
    while (i + (UNROLL_FACTOR - 1) * grid_stride < total_elements_per_channel_div_4) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int current_i = i + j * grid_stride;
            const int n = current_i / HW_div_4;
            const int hw_idx = current_i % HW_div_4;
            const int global_idx = n * CHW_div_4 + feature_idx * HW_div_4 + hw_idx;

            const float4 val4 = input[global_idx];
            
            thread_sum += val4.x + val4.y + val4.z + val4.w;
            thread_sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
        }
        i += unrolled_stride;
    }
    
    // Tail loop to handle remaining elements
    for (; i < total_elements_per_channel_div_4; i += grid_stride) {
        const int n = i / HW_div_4;
        const int hw_idx = i % HW_div_4;
        const int global_idx = n * CHW_div_4 + feature_idx * HW_div_4 + hw_idx;

        const float4 val4 = input[global_idx];
        
        thread_sum += val4.x + val4.y + val4.z + val4.w;
        thread_sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    }
    
    sdata[threadIdx.x] = {thread_sum, thread_sum_sq};
    __syncthreads();

    // Standard block-level reduction using the AoS layout.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x].sum += sdata[threadIdx.x + s].sum;
            sdata[threadIdx.x].sum_sq += sdata[threadIdx.x + s].sum_sq;
        }
        __syncthreads();
    }
    
    // One thread per block performs an atomic add to the global stats tensor
    if (threadIdx.x == 0) {
        atomicAdd(&stats[feature_idx * 2], sdata[0].sum);
        atomicAdd(&stats[feature_idx * 2 + 1], sdata[0].sum_sq);
    }
}


// Kernel 2: Apply normalization, computing mean/inv_var on-the-fly from raw stats.
// This kernel is unchanged from the SOTA.
__global__ void apply_norm_from_stats_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ stats, // Input: (C, 2) -> {sum, sum_sq}
    float4* __restrict__ output,
    int total_elements_div_4,
    int C, int HW,
    float num_elements_per_channel) {

    const int grid_stride = gridDim.x * blockDim.x;
    const int unrolled_stride = grid_stride * UNROLL_FACTOR;
    int idx_div_4 = blockIdx.x * blockDim.x + threadIdx.x;

    // Unrolled main loop with factor 16
    while (idx_div_4 + (UNROLL_FACTOR - 1) * grid_stride < total_elements_div_4) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; ++j) {
            const int current_idx = idx_div_4 + j * grid_stride;
            const int feature_idx = ((current_idx * 4) / HW) % C;
            
            const float total_sum = stats[feature_idx * 2];
            const float total_sum_sq = stats[feature_idx * 2 + 1];

            const float m = total_sum / num_elements_per_channel;
            const float var = (total_sum_sq / num_elements_per_channel) - (m * m);
            const float iv = rsqrtf(var + 1e-5f);

            const float w = weight[feature_idx];
            const float b = bias[feature_idx];

            const float4 in4 = input[current_idx];
            float4 out4;
            out4.x = (in4.x - m) * iv * w + b;
            out4.y = (in4.y - m) * iv * w + b;
            out4.z = (in4.z - m) * iv * w + b;
            out4.w = (in4.w - m) * iv * w + b;
            output[current_idx] = out4;
        }
        idx_div_4 += unrolled_stride;
    }
    
    // Tail loop to handle remaining elements
    for (; idx_div_4 < total_elements_div_4; idx_div_4 += grid_stride) {
        const int feature_idx = ((idx_div_4 * 4) / HW) % C;
        
        const float total_sum = stats[feature_idx * 2];
        const float total_sum_sq = stats[feature_idx * 2 + 1];
        
        const float m = total_sum / num_elements_per_channel;
        const float var = (total_sum_sq / num_elements_per_channel) - (m * m);
        const float iv = rsqrtf(var + 1e-5f);

        const float w = weight[feature_idx];
        const float b = bias[feature_idx];

        const float4 in4 = input[idx_div_4];
        float4 out4;
        out4.x = (in4.x - m) * iv * w + b;
        out4.y = (in4.y - m) * iv * w + b;
        out4.z = (in4.z - m) * iv * w + b;
        out4.w = (in4.w - m) * iv * w + b;
        output[idx_div_4] = out4;
    }
}


torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    input = input.contiguous();

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    TORCH_CHECK(W % 4 == 0, "Input width (W) must be divisible by 4 for float4 optimization.");
    
    const auto total_elements = input.numel();
    const auto W_div_4 = W / 4;

    TORCH_CHECK(weight.numel() == C, "Weight must have C elements");
    TORCH_CHECK(bias.numel() == C, "Bias must have C elements");

    auto output = torch::empty_like(input);
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kFloat32);
    
    // Allocate a single tensor for summed stats {sum, sum_sq} and zero it.
    auto stats = torch::zeros({C, 2}, options);

    // --- Pass 1: Fused Atomic Statistics Calculation ---
    const int stats_block_size = 256;
    const int stats_num_blocks = C * BLOCKS_PER_CHANNEL;
    // The shared memory size is the same, just the layout has changed.
    // block_size * sizeof(Sums) == block_size * 2 * sizeof(float)
    const int stats_shared_mem_size = stats_block_size * sizeof(Sums);
    
    fused_atomic_stats_kernel<<<stats_num_blocks, stats_block_size, stats_shared_mem_size>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        stats.data_ptr<float>(),
        N, C, H, W_div_4
    );

    // --- Pass 2: Apply Normalization (with on-the-fly mean/inv_var calculation) ---
    const int apply_block_size = 256;
    const int total_elements_div_4 = total_elements / 4;
    const int apply_num_blocks = (total_elements_div_4 + apply_block_size - 1) / apply_block_size;
    const float num_elements_per_channel = static_cast<float>(N * H * W);

    apply_norm_from_stats_kernel<<<apply_num_blocks, apply_block_size>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        stats.data_ptr<float>(),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        total_elements_div_4,
        C, H*W,
        num_elements_per_channel
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    return output;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
_layer_norm_module = load_inline(
    name="fused_layernorm_2pass_atomic_aos",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        
        if isinstance(num_features, (list, tuple)):
            channel_count = num_features[0]
        else:
            channel_count = num_features
        
        self.num_features = channel_count

        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        
        if _layer_norm_module is None:
            raise RuntimeError("CUDA extension for LayerNorm was not compiled successfully.")
        self.layer_norm_cuda = _layer_norm_module.layer_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), but got {x.dim()}D")
        
        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected input to have {self.num_features} features (channels), but got {x.size(1)}"
            )

        if x.size(3) % 4 != 0:
            raise ValueError(
                f"Input width (W) must be divisible by 4 for float4 optimization, but got {x.size(3)}"
            )
            
        return self.layer_norm_cuda(x, self.weight, self.bias)