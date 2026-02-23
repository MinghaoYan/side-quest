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

# Set CUDA architecture for A100 (sm_80) to ensure compilation for the target GPU
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernel for fused Linear, InstanceNorm, and add/mul operations.
# This version modifies the GEMV stage to reduce register pressure. Instead of using 20
# independent float accumulators for the 20x unrolled loop, it uses only 4 accumulators
# and cycles through them. The hypothesis is that the reduction in register pressure
# may lead to higher occupancy or better instruction scheduling, outweighing the
# potential reduction in instruction-level parallelism from having fewer independent
# accumulation chains.
fused_linear_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- Device Helper Functions ---

__device__ inline float rsqrt_fast(float x) {
    return rsqrtf(x);
}

// Reduces a single float value across a warp using shuffle-down instructions.
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --- Fused Kernel ---
// Fuses Linear (GEMV), InstanceNorm, and element-wise add/mul.
// It uses a "one-output-per-thread" approach for GEMV.
// The GEMV loop is 20x unrolled but uses only 4 accumulators to reduce register pressure.
__global__ void __launch_bounds__(128, 4) fused_linear_norm_add_mul_kernel(
    const float* __restrict__ x,          // Input tensor (Batch, InFeatures)
    const float* __restrict__ y,          // Second input tensor (Batch, OutFeatures)
    const float* __restrict__ linear_w,   // Linear layer weights (OutFeatures, InFeatures)
    const float* __restrict__ linear_b,   // Linear layer bias (OutFeatures)
    const float* __restrict__ norm_w,     // InstanceNorm weights (OutFeatures)
    const float* __restrict__ norm_b,     // InstanceNorm bias (OutFeatures)
    float* __restrict__ out,              // Output tensor (Batch, OutFeatures)
    const int batch_size,
    const int in_features,
    const int out_features,
    const float eps
) {
    // --- Setup ---
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    // Shared memory for intermediate linear output and reduction sums.
    extern __shared__ float s_mem[];
    float* s_linear_out = s_mem; // Size: out_features
    float* s_reduce_mem = &s_mem[out_features]; // Size for reduction: num_warps * 2

    // --- Part 1: Fused Linear Layer (GEMV) - Reduced Register Pressure ---
    for (int c_out_idx = threadIdx.x; c_out_idx < out_features; c_out_idx += blockDim.x) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        const int in_features_vec = in_features / 4;
        const float4* x_vec = reinterpret_cast<const float4*>(&x[batch_idx * in_features]);
        const float4* linear_w_vec = reinterpret_cast<const float4*>(&linear_w[c_out_idx * in_features]);

        const int unroll_factor = 20;
        int k = 0;
        if (in_features_vec >= unroll_factor) {
            for (; k <= in_features_vec - unroll_factor; k += unroll_factor) {
                #pragma unroll
                for(int i=0; i < unroll_factor; ++i) {
                    float4 x_v = x_vec[k + i];
                    float4 w_v = linear_w_vec[k + i];
                    // Cycle through 4 accumulators. The compiler resolves `i % 4` at compile time.
                    acc[i % 4] += x_v.x * w_v.x + x_v.y * w_v.y + x_v.z * w_v.z + x_v.w * w_v.w;
                }
            }
        }
        
        float remainder_acc = 0.0f;
        for (; k < in_features_vec; ++k) {
            float4 x_v = x_vec[k];
            float4 w_v = linear_w_vec[k];
            remainder_acc += x_v.x * w_v.x + x_v.y * w_v.y + x_v.z * w_v.z + x_v.w * w_v.w;
        }

        float total_acc = remainder_acc + acc[0] + acc[1] + acc[2] + acc[3];

        s_linear_out[c_out_idx] = total_acc + linear_b[c_out_idx];
    }
    __syncthreads();

    // --- Part 2: Instance Normalization (Mean and Variance Calculation) ---
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    const int out_features_vec = out_features / 4;
    const float4* s_linear_out_vec = reinterpret_cast<const float4*>(s_linear_out);

    for (int i = threadIdx.x; i < out_features_vec; i += blockDim.x) {
        float4 val4 = s_linear_out_vec[i];
        local_sum += val4.x + val4.y + val4.z + val4.w;
        local_sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    }
    
    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sum_sq = warp_reduce_sum(local_sum_sq);

    if (lane_id == 0) {
        s_reduce_mem[warp_id] = warp_sum;
        s_reduce_mem[warp_id + blockDim.x / 32] = warp_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < blockDim.x / 32) ? s_reduce_mem[lane_id] : 0.0f;
        local_sum_sq = (lane_id < blockDim.x / 32) ? s_reduce_mem[lane_id + blockDim.x / 32] : 0.0f;
        warp_sum = warp_reduce_sum(local_sum);
        warp_sum_sq = warp_reduce_sum(local_sum_sq);
    }
    
    if (threadIdx.x == 0) {
        float mean = warp_sum / out_features;
        float var = (warp_sum_sq / out_features) - (mean * mean);
        s_reduce_mem[0] = mean;
        s_reduce_mem[1] = rsqrt_fast(var + eps);
    }
    __syncthreads();
    
    const float mean = s_reduce_mem[0];
    const float inv_std = s_reduce_mem[1];

    // --- Part 3: Vectorized Normalization, Affine Transform, and Fused Add/Mul ---
    for (int i = threadIdx.x; i < out_features_vec; i += blockDim.x) {
        const int offset = i * 4;
        
        const float4 val4 = *reinterpret_cast<const float4*>(&s_linear_out[offset]);
        const float4 norm_w4 = *reinterpret_cast<const float4*>(&norm_w[offset]);
        const float4 norm_b4 = *reinterpret_cast<const float4*>(&norm_b[offset]);
        const float4 y4 = *reinterpret_cast<const float4*>(&y[batch_idx * out_features + offset]);
        
        const float4 mean4 = {mean, mean, mean, mean};
        const float4 inv_std4 = {inv_std, inv_std, inv_std, inv_std};
        
        const float4 normalized_val4 = {
            (val4.x - mean4.x) * inv_std4.x, (val4.y - mean4.y) * inv_std4.y,
            (val4.z - mean4.z) * inv_std4.z, (val4.w - mean4.w) * inv_std4.w
        };
        
        const float4 transformed_val4 = {
            normalized_val4.x * norm_w4.x + norm_b4.x, normalized_val4.y * norm_w4.y + norm_b4.y,
            normalized_val4.z * norm_w4.z + norm_b4.z, normalized_val4.w * norm_w4.w + norm_b4.w
        };
        
        const float4 temp_val4 = {
            transformed_val4.x + y4.x, transformed_val4.y + y4.y,
            transformed_val4.z + y4.z, transformed_val4.w + y4.w
        };
        
        const float4 out4 = {
            temp_val4.x * y4.x, temp_val4.y * y4.y,
            temp_val4.z * y4.z, temp_val4.w * y4.w
        };
        
        *reinterpret_cast<float4*>(&out[batch_idx * out_features + offset]) = out4;
    }
}


torch::Tensor fused_linear_norm_add_mul(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor linear_w,
    torch::Tensor linear_b,
    torch::Tensor norm_w,
    torch::Tensor norm_b,
    double eps
) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && linear_w.is_cuda() && linear_b.is_cuda() && norm_w.is_cuda() && norm_b.is_cuda(), "All input tensors must be on a CUDA device");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && linear_w.is_contiguous(), "Inputs x, y, and linear_w must be contiguous");
    
    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = y.size(1);
    
    TORCH_CHECK(in_features % 4 == 0, "in_features must be divisible by 4 for float4 vectorization");
    TORCH_CHECK(out_features % 4 == 0, "out_features must be divisible by 4 for float4 vectorization in reduction");
    TORCH_CHECK(out_features > 0, "out_features must be positive");

    auto out = torch::empty_like(y);

    if (batch_size == 0) {
        return out;
    }

    int threads_per_block = 128;
    const int num_blocks = batch_size;
    const int num_warps = threads_per_block / 32;

    const int shared_mem_size = (out_features + num_warps * 2) * sizeof(float);
    
    fused_linear_norm_add_mul_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        linear_w.data_ptr<float>(),
        linear_b.data_ptr<float>(),
        norm_w.data_ptr<float>(),
        norm_b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        static_cast<float>(eps)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed in fused_linear_norm_add_mul: ", cudaGetErrorString(err));
    }

    return out;
}
"""

fused_linear_norm_cpp_source = """
torch::Tensor fused_linear_norm_add_mul(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor linear_w,
    torch::Tensor linear_b,
    torch::Tensor norm_w,
    torch::Tensor norm_b,
    double eps
);
"""

# JIT compile the CUDA/C++ code
fused_op_reg_pressure_opt = load_inline(
    name='fused_op_reg_pressure_opt',
    cpp_sources=fused_linear_norm_cpp_source,
    cuda_sources=fused_linear_norm_source,
    functions=['fused_linear_norm_add_mul'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ModelNew(nn.Module):
    """
    An optimized model that fuses Linear, InstanceNorm, and element-wise operations.
    
    This version modifies the GEMV stage of the fused kernel to reduce register pressure.
    Instead of using 20 independent accumulators for the 20x unrolled loop, it uses only
    4 accumulators and cycles through them. The hypothesis is that lower register pressure
    may lead to better performance by enabling higher occupancy or more efficient
    instruction scheduling by the compiler.
    """
    def __init__(self, in_features: int, out_features: int, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if out_features % 4 != 0:
            raise ValueError(f"out_features must be divisible by 4 for this optimized kernel, but got {out_features}")

        self.linear = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm1d(out_features, eps=eps, momentum=momentum, affine=True)
        self.eps = eps
        self.fused_op = fused_op_reg_pressure_opt

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(f"Input x has wrong dimensions. Expected (batch, {self.in_features}), got {x.shape}")
        if y.dim() != 2 or y.size(1) != self.out_features:
            raise ValueError(f"Input y has wrong dimensions. Expected (batch, {self.out_features}), got {y.shape}")

        return self.fused_op.fused_linear_norm_add_mul(
            x,
            y,
            self.linear.weight,
            self.linear.bias,
            self.instance_norm.weight,
            self.instance_norm.bias,
            self.eps
        )