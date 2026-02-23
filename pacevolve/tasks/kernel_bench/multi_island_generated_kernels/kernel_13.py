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

# A batch_size variable is assumed to be in the global scope for the
# model constructor, as per the structure of the provided state-of-the-art code.
# We define a default here for standalone execution.
try:
    batch_size
except NameError:
    batch_size = 64

rnn_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// This experiment builds on the successful feature-parallel strategy from Idea 5,
// which uses two separate kernels for the i2h and h2o stages. The current state-of-the-art
// combines this macro-architecture with the best micro-kernel from Idea 2: a symmetric
// 4-way instruction-level parallel (ILP) warp-reduction engine.
//
// The hypothesis here (related to Idea 2) is that the optimal micro-kernel shape may not be
// symmetric across the two distinct kernels. The h2o kernel has a smaller input vector
// (hidden_size vs. input_size + hidden_size) and simpler arithmetic (no tanh activation).
// This reduced complexity might lower register pressure, creating an opportunity for
// more aggressive latency hiding. This experiment tests this by increasing the ILP of the
// h2o kernel from 4 to 8, while retaining the proven 4-way ILP for the more complex i2h kernel.

#define I2H_TILE_H 128 // Work per block for the i2h kernel (hidden features)
#define H2O_TILE_O 128 // Work per block for the h2o kernel (output features)
#define THREADS_PER_BLOCK 1024

__device__ __forceinline__ float fast_tanhf(float x) {
    // Use the PTX instruction for tanh approximation available on SM 8.0+ (A100)
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1) i2h_feature_parallel_kernel(
    const float* __restrict__ input,
    const float* __restrict__ hidden,
    const float* __restrict__ i2h_weight,
    const float* __restrict__ i2h_bias,
    float* __restrict__ new_hidden_out,
    int batch_size,
    int input_size,
    int hidden_size
) {
    // Grid is (ceil(hidden_size/I2H_TILE_H), batch_size)
    const int batch_idx = blockIdx.y;
    const int h_tile_start = blockIdx.x * I2H_TILE_H;
    
    extern __shared__ float s_mem[];
    const int combined_size = input_size + hidden_size;
    float* s_combined_input = s_mem;

    // --- Cooperative, Vectorized Load of [input, hidden] into Shared Memory ---
    const float* current_input = input + batch_idx * input_size;
    const float* current_hidden = hidden + batch_idx * hidden_size;
    
    // Load input part
    for (int i = threadIdx.x; i < input_size / 2; i += blockDim.x) {
        reinterpret_cast<float2*>(s_combined_input)[i] = reinterpret_cast<const float2*>(current_input)[i];
    }
    if ((input_size % 2 != 0) && (threadIdx.x == 0)) {
       s_combined_input[input_size - 1] = current_input[input_size - 1];
    }

    // Load hidden part
    for (int i = threadIdx.x; i < hidden_size / 2; i += blockDim.x) {
        reinterpret_cast<float2*>(s_combined_input + input_size)[i] = reinterpret_cast<const float2*>(current_hidden)[i];
    }
    if ((hidden_size % 2 != 0) && (threadIdx.x == 0)) {
       s_combined_input[input_size + hidden_size - 1] = current_hidden[hidden_size - 1];
    }
    __syncthreads();


    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp processes up to 4 output hidden features (4x ILP)
    const int h_idx = h_tile_start + warp_id * 4;
    
    if (h_idx >= hidden_size) return;

    // --- Dot Product with Boundary Checks ---
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    const bool process2 = (h_idx + 1 < hidden_size);
    const bool process3 = (h_idx + 2 < hidden_size);
    const bool process4 = (h_idx + 3 < hidden_size);

    const float2* s_combined_input_v2 = reinterpret_cast<const float2*>(s_combined_input);
    const int combined_size_v2 = combined_size / 2;

    const float2* weight_row1_v2 = reinterpret_cast<const float2*>(i2h_weight + h_idx * combined_size);
    const float2* weight_row2_v2 = process2 ? reinterpret_cast<const float2*>(i2h_weight + (h_idx + 1) * combined_size) : nullptr;
    const float2* weight_row3_v2 = process3 ? reinterpret_cast<const float2*>(i2h_weight + (h_idx + 2) * combined_size) : nullptr;
    const float2* weight_row4_v2 = process4 ? reinterpret_cast<const float2*>(i2h_weight + (h_idx + 3) * combined_size) : nullptr;

    for (int i = lane_id; i < combined_size_v2; i += 32) {
        const float2 s_val = s_combined_input_v2[i];
        sum1 += s_val.x * weight_row1_v2[i].x + s_val.y * weight_row1_v2[i].y;
        if (process2) sum2 += s_val.x * weight_row2_v2[i].x + s_val.y * weight_row2_v2[i].y;
        if (process3) sum3 += s_val.x * weight_row3_v2[i].x + s_val.y * weight_row3_v2[i].y;
        if (process4) sum4 += s_val.x * weight_row4_v2[i].x + s_val.y * weight_row4_v2[i].y;
    }

    if (combined_size % 2 != 0) {
        if (lane_id == 0) {
            const float s_val_last = s_combined_input[combined_size - 1];
            sum1 += s_val_last * i2h_weight[h_idx * combined_size + combined_size - 1];
            if (process2) sum2 += s_val_last * i2h_weight[(h_idx + 1) * combined_size + combined_size - 1];
            if (process3) sum3 += s_val_last * i2h_weight[(h_idx + 2) * combined_size + combined_size - 1];
            if (process4) sum4 += s_val_last * i2h_weight[(h_idx + 3) * combined_size + combined_size - 1];
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffff, sum3, offset);
        sum4 += __shfl_down_sync(0xffffffff, sum4, offset);
    }

    if (lane_id == 0) {
        new_hidden_out[batch_idx * hidden_size + h_idx] = fast_tanhf(sum1 + i2h_bias[h_idx]);
        if (process2) new_hidden_out[batch_idx * hidden_size + h_idx + 1] = fast_tanhf(sum2 + i2h_bias[h_idx + 1]);
        if (process3) new_hidden_out[batch_idx * hidden_size + h_idx + 2] = fast_tanhf(sum3 + i2h_bias[h_idx + 2]);
        if (process4) new_hidden_out[batch_idx * hidden_size + h_idx + 3] = fast_tanhf(sum4 + i2h_bias[h_idx + 3]);
    }
}


__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1) h2o_feature_parallel_kernel(
    const float* __restrict__ new_hidden,
    const float* __restrict__ h2o_weight,
    const float* __restrict__ h2o_bias,
    float* __restrict__ output,
    int batch_size,
    int hidden_size,
    int output_size
) {
    // Grid is (ceil(output_size/H2O_TILE_O), batch_size)
    const int batch_idx = blockIdx.y;
    const int o_tile_start = blockIdx.x * H2O_TILE_O;

    extern __shared__ float s_mem[];
    float* s_new_hidden = s_mem;

    // Cooperative, vectorized load of new_hidden into Shared Memory
    const float* current_hidden = new_hidden + batch_idx * hidden_size;
    for (int i = threadIdx.x; i < hidden_size / 2; i += blockDim.x) {
        reinterpret_cast<float2*>(s_new_hidden)[i] = reinterpret_cast<const float2*>(current_hidden)[i];
    }
    if ((hidden_size % 2 != 0) && (threadIdx.x == 0)) {
       s_new_hidden[hidden_size - 1] = current_hidden[hidden_size - 1];
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp processes up to 8 output features (8x ILP)
    const int o_idx = o_tile_start + warp_id * 8;

    if (o_idx >= output_size) return;

    // --- Dot Product with 8x ILP and Boundary Checks ---
    float sum1=0, sum2=0, sum3=0, sum4=0, sum5=0, sum6=0, sum7=0, sum8=0;
    const bool p2 = (o_idx + 1 < output_size);
    const bool p3 = (o_idx + 2 < output_size);
    const bool p4 = (o_idx + 3 < output_size);
    const bool p5 = (o_idx + 4 < output_size);
    const bool p6 = (o_idx + 5 < output_size);
    const bool p7 = (o_idx + 6 < output_size);
    const bool p8 = (o_idx + 7 < output_size);

    const float2* s_new_hidden_v2 = reinterpret_cast<const float2*>(s_new_hidden);
    const int hidden_size_v2 = hidden_size / 2;

    const float2* w1 = reinterpret_cast<const float2*>(h2o_weight + o_idx * hidden_size);
    const float2* w2 = p2 ? reinterpret_cast<const float2*>(h2o_weight + (o_idx + 1) * hidden_size) : nullptr;
    const float2* w3 = p3 ? reinterpret_cast<const float2*>(h2o_weight + (o_idx + 2) * hidden_size) : nullptr;
    const float2* w4 = p4 ? reinterpret_cast<const float2*>(h2o_weight + (o_idx + 3) * hidden_size) : nullptr;
    const float2* w5 = p5 ? reinterpret_cast<const float2*>(h2o_weight + (o_idx + 4) * hidden_size) : nullptr;
    const float2* w6 = p6 ? reinterpret_cast<const float2*>(h2o_weight + (o_idx + 5) * hidden_size) : nullptr;
    const float2* w7 = p7 ? reinterpret_cast<const float2*>(h2o_weight + (o_idx + 6) * hidden_size) : nullptr;
    const float2* w8 = p8 ? reinterpret_cast<const float2*>(h2o_weight + (o_idx + 7) * hidden_size) : nullptr;

    for (int i = lane_id; i < hidden_size_v2; i += 32) {
        const float2 s_val = s_new_hidden_v2[i];
        sum1 += s_val.x * w1[i].x + s_val.y * w1[i].y;
        if (p2) sum2 += s_val.x * w2[i].x + s_val.y * w2[i].y;
        if (p3) sum3 += s_val.x * w3[i].x + s_val.y * w3[i].y;
        if (p4) sum4 += s_val.x * w4[i].x + s_val.y * w4[i].y;
        if (p5) sum5 += s_val.x * w5[i].x + s_val.y * w5[i].y;
        if (p6) sum6 += s_val.x * w6[i].x + s_val.y * w6[i].y;
        if (p7) sum7 += s_val.x * w7[i].x + s_val.y * w7[i].y;
        if (p8) sum8 += s_val.x * w8[i].x + s_val.y * w8[i].y;
    }

    if (hidden_size % 2 != 0) {
        if (lane_id == 0) {
            const float s_val_last = s_new_hidden[hidden_size - 1];
            sum1 += s_val_last * h2o_weight[o_idx * hidden_size + hidden_size - 1];
            if(p2) sum2 += s_val_last * h2o_weight[(o_idx + 1) * hidden_size + hidden_size - 1];
            if(p3) sum3 += s_val_last * h2o_weight[(o_idx + 2) * hidden_size + hidden_size - 1];
            if(p4) sum4 += s_val_last * h2o_weight[(o_idx + 3) * hidden_size + hidden_size - 1];
            if(p5) sum5 += s_val_last * h2o_weight[(o_idx + 4) * hidden_size + hidden_size - 1];
            if(p6) sum6 += s_val_last * h2o_weight[(o_idx + 5) * hidden_size + hidden_size - 1];
            if(p7) sum7 += s_val_last * h2o_weight[(o_idx + 6) * hidden_size + hidden_size - 1];
            if(p8) sum8 += s_val_last * h2o_weight[(o_idx + 7) * hidden_size + hidden_size - 1];
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffff, sum3, offset);
        sum4 += __shfl_down_sync(0xffffffff, sum4, offset);
        sum5 += __shfl_down_sync(0xffffffff, sum5, offset);
        sum6 += __shfl_down_sync(0xffffffff, sum6, offset);
        sum7 += __shfl_down_sync(0xffffffff, sum7, offset);
        sum8 += __shfl_down_sync(0xffffffff, sum8, offset);
    }
    
    if (lane_id == 0) {
        output[batch_idx * output_size + o_idx] = sum1 + h2o_bias[o_idx];
        if (p2) output[batch_idx * output_size + o_idx + 1] = sum2 + h2o_bias[o_idx + 1];
        if (p3) output[batch_idx * output_size + o_idx + 2] = sum3 + h2o_bias[o_idx + 2];
        if (p4) output[batch_idx * output_size + o_idx + 3] = sum4 + h2o_bias[o_idx + 3];
        if (p5) output[batch_idx * output_size + o_idx + 4] = sum5 + h2o_bias[o_idx + 4];
        if (p6) output[batch_idx * output_size + o_idx + 5] = sum6 + h2o_bias[o_idx + 5];
        if (p7) output[batch_idx * output_size + o_idx + 6] = sum7 + h2o_bias[o_idx + 6];
        if (p8) output[batch_idx * output_size + o_idx + 7] = sum8 + h2o_bias[o_idx + 7];
    }
}

std::vector<torch::Tensor> rnn_forward_feature_parallel_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = hidden.size(1);
    const int output_size = h2o_weight.size(0);

    auto new_hidden = torch::empty({batch_size, hidden_size}, input.options());
    auto output = torch::empty({batch_size, output_size}, input.options());

    const dim3 threads(THREADS_PER_BLOCK);

    // Launch i2h kernel
    const dim3 i2h_blocks((hidden_size + I2H_TILE_H - 1) / I2H_TILE_H, batch_size);
    const size_t i2h_shared_mem_size = (input_size + hidden_size) * sizeof(float);
    i2h_feature_parallel_kernel<<<i2h_blocks, threads, i2h_shared_mem_size>>>(
        input.data_ptr<float>(),
        hidden.data_ptr<float>(),
        i2h_weight.data_ptr<float>(),
        i2h_bias.data_ptr<float>(),
        new_hidden.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    // Launch h2o kernel
    const dim3 h2o_blocks((output_size + H2O_TILE_O - 1) / H2O_TILE_O, batch_size);
    const size_t h2o_shared_mem_size = hidden_size * sizeof(float);
    h2o_feature_parallel_kernel<<<h2o_blocks, threads, h2o_shared_mem_size>>>(
        new_hidden.data_ptr<float>(),
        h2o_weight.data_ptr<float>(),
        h2o_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        output_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return {output, new_hidden};
}
"""

rnn_cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declaration of the function that will be defined in the CUDA source
std::vector<torch::Tensor> rnn_forward_feature_parallel_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias
);
"""

# JIT compilation of the CUDA kernel
rnn_cuda_feature_parallel = load_inline(
    name='rnn_cuda_feature_parallel',
    cpp_sources=rnn_cpp_source,
    cuda_sources=rnn_cuda_source,
    functions=['rnn_forward_feature_parallel_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize hidden state. It will be moved to the correct device
        # and resized based on batch size during the first forward pass.
        self.hidden = torch.randn((batch_size, self.hidden_size))

        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        # Store the JIT compiled function
        self.rnn_cuda = rnn_cuda_feature_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure hidden state is on the same device as the input
        if self.hidden.device != x.device:
            self.hidden = self.hidden.to(x.device)
        
        # Check if batch size has changed and re-initialize hidden state if necessary.
        # This makes the model more flexible to varying batch sizes at inference.
        if self.hidden.shape[0] != x.shape[0]:
            self.hidden = torch.randn((x.shape[0], self.hidden_size), device=x.device, dtype=x.dtype)

        # Call the custom CUDA kernel for the fused RNN cell operation
        output, self.hidden = self.rnn_cuda.rnn_forward_feature_parallel_cuda(
            x,
            self.hidden,
            self.i2h.weight,
            self.i2h.bias,
            self.h2o.weight,
            self.h2o.bias
        )
        return output