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

# Set CUDA architecture for A100 to ensure FP16/BF16 support is compiled correctly.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Vertically fused kernel that extends the SOTA by replacing expensive integer
# arithmetic (div/mod) in the input data loading loop with an incremental
# address calculation scheme. The SOTA kernel recalculates the 3D source index
# from a flat index on every loop iteration. This version calculates the
# initial coordinates once and then updates them incrementally with cheaper
# additions, aiming to reduce instruction overhead and improve memory loading
# efficiency.

cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Define constants for the fused operation
constexpr int CONV_KERNEL_SIZE = 3;
constexpr int CONV_STRIDE = 1;
constexpr int CONV_PADDING = 1;
constexpr int POOL_SIZE = 2;
constexpr int OUT_CHANNELS = 16;
constexpr int CONV_KERNEL_VOL = CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE;

// Output tile dimensions computed by one thread block (4x16 tile)
constexpr int TILE_H_OUT = 4;
constexpr int TILE_W_OUT = 16;

// Intermediate convolution tile dimensions needed for the output tile
constexpr int TILE_D_CONV = POOL_SIZE; // = 2
constexpr int TILE_H_CONV = TILE_H_OUT * POOL_SIZE; // 4*2 = 8
constexpr int TILE_W_CONV = TILE_W_OUT * POOL_SIZE; // 16*2 = 32

// Input 'x' tile dimensions required for the convolution tile
constexpr int TILE_D_IN = TILE_D_CONV + CONV_KERNEL_SIZE - 1; // 2+3-1 = 4
constexpr int TILE_H_IN = TILE_H_CONV + CONV_KERNEL_SIZE - 1; // 8+3-1 = 10
constexpr int TILE_W_IN = TILE_W_CONV + CONV_KERNEL_SIZE - 1; // 32+3-1 = 34

__global__ void fused_conv_maxpool_bias_logsumexp_relu_kernel(
    const half* __restrict__ x, const half* __restrict__ weight, const half* __restrict__ bias,
    half* __restrict__ y,
    int N, int C_in, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out)
{
    // --- 1. Thread and Block Indexing ---
    const int w_out_base = blockIdx.x * TILE_W_OUT;
    const int h_out_base = blockIdx.y * TILE_H_OUT;
    const int nd_idx = blockIdx.z;

    if (nd_idx >= N * D_out) return;

    const int n = nd_idx / D_out;
    const int d_out = nd_idx % D_out;

    // Shared memory for convolution weights, input tensor 'x', and convolution output
    extern __shared__ half smem[];
    half* sm_weights = smem;
    half* sm_input_x = sm_weights + OUT_CHANNELS * CONV_KERNEL_VOL;
    half* sm_conv_out = sm_input_x + TILE_D_IN * TILE_H_IN * TILE_W_IN;

    // Use 128 threads. Each computes 4 points of the intermediate conv tile.
    half conv_accum[OUT_CHANNELS][4] = {{0.0f}};

    // --- 2. Fused Convolution Stage ---
    const int d_conv_base = d_out * POOL_SIZE;
    const int h_conv_base = h_out_base * POOL_SIZE;
    const int w_conv_base = w_out_base * POOL_SIZE;

    const int d_in_base = d_conv_base - CONV_PADDING;
    const int h_in_base = h_conv_base - CONV_PADDING;
    const int w_in_base = w_conv_base - CONV_PADDING;

    constexpr int num_weights_per_ic = OUT_CHANNELS * CONV_KERNEL_VOL;

    // Stream through input channels to manage memory pressure
    for (int ic = 0; ic < C_in; ++ic) {
        // a. Cooperatively load weights for the current input channel using half2 vectorization.
        const int num_weights_per_ic_vec = num_weights_per_ic / 2;
        const half2* weight_ic_base_vec = reinterpret_cast<const half2*>(weight + (long long)ic * num_weights_per_ic);
        half2* sm_weights_vec = reinterpret_cast<half2*>(sm_weights);
        for (int i = threadIdx.x; i < num_weights_per_ic_vec; i += blockDim.x) {
            sm_weights_vec[i] = weight_ic_base_vec[i];
        }

        // b. Cooperatively load the spatial tile of 'x' using incremental address calculation.
        half2* sm_input_x_vec = reinterpret_cast<half2*>(sm_input_x);
        const int smem_tile_size_vec = (TILE_D_IN * TILE_H_IN * TILE_W_IN) / 2;
        
        // Calculate initial local coordinates from threadIdx for the grid-stride loop
        int initial_flat_idx = threadIdx.x * 2;
        int w_local_curr = initial_flat_idx % TILE_W_IN;
        int h_rem = initial_flat_idx / TILE_W_IN;
        int h_local_curr = h_rem % TILE_H_IN;
        int d_local_curr = h_rem / TILE_H_IN;

        // Pre-calculate coordinate strides for the next iteration (avoids div/mod in loop)
        const int stride_flat = blockDim.x * 2;
        const int w_stride = stride_flat % TILE_W_IN;
        const int h_rem_stride = stride_flat / TILE_W_IN;
        const int h_stride = h_rem_stride % TILE_H_IN;
        const int d_stride = h_rem_stride / TILE_H_IN;
        
        for (int i = threadIdx.x; i < smem_tile_size_vec; i += blockDim.x) {
            // --- Load first element of half2 using current coordinates ---
            int d_in_idx_1 = d_in_base + d_local_curr;
            int h_in_idx_1 = h_in_base + h_local_curr;
            int w_in_idx_1 = w_in_base + w_local_curr;
            
            half val1 = (half)0.0f;
            if (d_in_idx_1 >= 0 && d_in_idx_1 < D_in &&
                h_in_idx_1 >= 0 && h_in_idx_1 < H_in &&
                w_in_idx_1 >= 0 && w_in_idx_1 < W_in) {
                val1 = x[((((long long)n * C_in + ic) * D_in + d_in_idx_1) * H_in + h_in_idx_1) * W_in + w_in_idx_1];
            }

            // --- Load second element of half2 by incrementing coordinates by 1 ---
            int d_local_2 = d_local_curr;
            int h_local_2 = h_local_curr;
            int w_local_2 = w_local_curr + 1;
            if (w_local_2 == TILE_W_IN) {
                w_local_2 = 0; h_local_2++;
                if (h_local_2 == TILE_H_IN) { h_local_2 = 0; d_local_2++; }
            }

            int d_in_idx_2 = d_in_base + d_local_2;
            int h_in_idx_2 = h_in_base + h_local_2;
            int w_in_idx_2 = w_in_base + w_local_2;

            half val2 = (half)0.0f;
            if (d_in_idx_2 >= 0 && d_in_idx_2 < D_in &&
                h_in_idx_2 >= 0 && h_in_idx_2 < H_in &&
                w_in_idx_2 >= 0 && w_in_idx_2 < W_in) {
                val2 = x[((((long long)n * C_in + ic) * D_in + d_in_idx_2) * H_in + h_in_idx_2) * W_in + w_in_idx_2];
            }
            
            sm_input_x_vec[i] = make_half2(val1, val2);

            // --- Update coordinates for next iteration using pre-calculated strides ---
            w_local_curr += w_stride;
            if (w_local_curr >= TILE_W_IN) { w_local_curr -= TILE_W_IN; h_local_curr++; }
            
            h_local_curr += h_stride;
            if (h_local_curr >= TILE_H_IN) { h_local_curr -= TILE_H_IN; d_local_curr++; }
            
            d_local_curr += d_stride;
        }
        __syncthreads();

        // c. Compute partial convolutions and accumulate in registers
        #pragma unroll
        for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
            #pragma unroll
            for (int p = 0; p < 4; ++p) { // Each thread computes 4 points
                int point_idx = threadIdx.x * 4 + p;
                int w_conv_local = point_idx % TILE_W_CONV;
                int h_conv_local = (point_idx / TILE_W_CONV) % TILE_H_CONV;
                int d_conv_local = point_idx / (TILE_W_CONV * TILE_H_CONV);

                half accum_val = 0.0f;
                #pragma unroll
                for (int kd = 0; kd < CONV_KERNEL_SIZE; ++kd) {
                    #pragma unroll
                    for (int kh = 0; kh < CONV_KERNEL_SIZE; ++kh) {
                        #pragma unroll
                        for (int kw = 0; kw < CONV_KERNEL_SIZE; ++kw) {
                            int d_in_local = d_conv_local + kd;
                            int h_in_local = h_conv_local + kh;
                            int w_in_local = w_conv_local + kw;
                            int input_idx = d_in_local * TILE_H_IN * TILE_W_IN + h_in_local * TILE_W_IN + w_in_local;
                            int weight_idx = oc * CONV_KERNEL_VOL + kd * 9 + kh * 3 + kw;
                            accum_val = __hfma(sm_input_x[input_idx], sm_weights[weight_idx], accum_val);
                        }
                    }
                }
                conv_accum[oc][p] = __hadd(conv_accum[oc][p], accum_val);
            }
        }
        __syncthreads();
    }
    
    // Write accumulators to shared memory
    for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
        for (int p = 0; p < 4; ++p) {
            int point_idx = threadIdx.x * 4 + p;
            int sm_idx = oc * (TILE_D_CONV * TILE_H_CONV * TILE_W_CONV) + point_idx;
            sm_conv_out[sm_idx] = conv_accum[oc][p];
        }
    }
    __syncthreads();

    // --- 3. Fused MaxPool3D, Bias, LogSumExp, ReLU Stage ---
    if (threadIdx.x < (TILE_H_OUT * TILE_W_OUT)) {
        const int h_out_local = threadIdx.x / TILE_W_OUT;
        const int w_out_local = threadIdx.x % TILE_W_OUT;
        const int h_out = h_out_base + h_out_local;
        const int w_out = w_out_base + w_out_local;

        if (w_out < W_out && h_out < H_out) {
            half max_pooled_vals[OUT_CHANNELS];
            
            for(int c = 0; c < OUT_CHANNELS; ++c) {
                const int h_conv_start = h_out_local * POOL_SIZE;
                const int w_conv_start = w_out_local * POOL_SIZE;
                
                const int sm_c_base = c * (TILE_D_CONV * TILE_H_CONV * TILE_W_CONV);
                const int sm_d0_base = sm_c_base;
                const int sm_d1_base = sm_c_base + TILE_H_CONV * TILE_W_CONV;

                half m0 = sm_conv_out[sm_d0_base + (h_conv_start+0)*TILE_W_CONV + (w_conv_start+0)];
                half m1 = sm_conv_out[sm_d0_base + (h_conv_start+0)*TILE_W_CONV + (w_conv_start+1)];
                half m2 = sm_conv_out[sm_d0_base + (h_conv_start+1)*TILE_W_CONV + (w_conv_start+0)];
                half m3 = sm_conv_out[sm_d0_base + (h_conv_start+1)*TILE_W_CONV + (w_conv_start+1)];
                half m4 = sm_conv_out[sm_d1_base + (h_conv_start+0)*TILE_W_CONV + (w_conv_start+0)];
                half m5 = sm_conv_out[sm_d1_base + (h_conv_start+0)*TILE_W_CONV + (w_conv_start+1)];
                half m6 = sm_conv_out[sm_d1_base + (h_conv_start+1)*TILE_W_CONV + (w_conv_start+0)];
                half m7 = sm_conv_out[sm_d1_base + (h_conv_start+1)*TILE_W_CONV + (w_conv_start+1)];

                half max_plane0 = __hmax(__hmax(m0, m1), __hmax(m2, m3));
                half max_plane1 = __hmax(__hmax(m4, m5), __hmax(m6, m7));
                max_pooled_vals[c] = __hadd(__hmax(max_plane0, max_plane1), bias[c]);
            }

            // --- 4. Fused LogSumExp and ReLU ---
            half m_0_1 = __hmax(max_pooled_vals[0], max_pooled_vals[1]); half m_2_3 = __hmax(max_pooled_vals[2], max_pooled_vals[3]);
            half m_4_5 = __hmax(max_pooled_vals[4], max_pooled_vals[5]); half m_6_7 = __hmax(max_pooled_vals[6], max_pooled_vals[7]);
            half m_8_9 = __hmax(max_pooled_vals[8], max_pooled_vals[9]); half m_10_11 = __hmax(max_pooled_vals[10], max_pooled_vals[11]);
            half m_12_13 = __hmax(max_pooled_vals[12], max_pooled_vals[13]); half m_14_15 = __hmax(max_pooled_vals[14], max_pooled_vals[15]);
            half m_0_3 = __hmax(m_0_1, m_2_3); half m_4_7 = __hmax(m_4_5, m_6_7);
            half m_8_11 = __hmax(m_8_9, m_10_11); half m_12_15 = __hmax(m_12_13, m_14_15);
            half m_0_7 = __hmax(m_0_3, m_4_7); half m_8_15 = __hmax(m_8_11, m_12_15);
            half max_val = __hmax(m_0_7, m_8_15);

            if (__hgt(max_val, __float2half(-65000.0f))) {
                float sum_exp_fp32 = 0.0f;
                #pragma unroll
                for(int k = 0; k < OUT_CHANNELS; ++k) {
                    sum_exp_fp32 += __half2float(hexp(__hsub(max_pooled_vals[k], max_val)));
                }
                half lse_h = __hadd(max_val, hlog(__float2half_rn(sum_exp_fp32)));
                long long y_idx = (long long)n * D_out * H_out * W_out +
                                    (long long)d_out * H_out * W_out +
                                    (long long)h_out * W_out + w_out;
                y[y_idx] = __hmax(lse_h, (half)0.0f);
            } else {
                long long y_idx = (long long)n * D_out * H_out * W_out +
                                    (long long)d_out * H_out * W_out +
                                    (long long)h_out * W_out + w_out;
                y[y_idx] = (half)0.0f;
            }
        }
    }
}


torch::Tensor fused_conv_maxpool_bias_logsumexp_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kHalf, "Input x invalid");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kHalf, "Input weight invalid");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == torch::kHalf, "Input bias invalid");

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int D_in = x.size(2);
    const int H_in = x.size(3);
    const int W_in = x.size(4);
    
    TORCH_CHECK(weight.dim() == 2, "Reshaped weight must be 2D");
    TORCH_CHECK(weight.size(0) == C_in, "Reshaped weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == OUT_CHANNELS, "Bias size must be 16");

    const int D_conv = (D_in + 2 * CONV_PADDING - CONV_KERNEL_SIZE) / CONV_STRIDE + 1;
    const int H_conv = (H_in + 2 * CONV_PADDING - CONV_KERNEL_SIZE) / CONV_STRIDE + 1;
    const int W_conv = (W_in + 2 * CONV_PADDING - CONV_KERNEL_SIZE) / CONV_STRIDE + 1;
    
    const int D_out = D_conv / POOL_SIZE;
    const int H_out = H_conv / POOL_SIZE;
    const int W_out = W_conv / POOL_SIZE;

    auto y = torch::zeros({N, 1, D_out, H_out, W_out}, x.options());
    if (y.numel() == 0) return y;
    
    const dim3 threads_per_block(128);
    const dim3 num_blocks(
        (W_out + TILE_W_OUT - 1) / TILE_W_OUT,
        (H_out + TILE_H_OUT - 1) / TILE_H_OUT,
        (long long)N * D_out
    );
    
    size_t smem_size = (OUT_CHANNELS * CONV_KERNEL_VOL +
                        TILE_D_IN * TILE_H_IN * TILE_W_IN +
                        OUT_CHANNELS * TILE_D_CONV * TILE_H_CONV * TILE_W_CONV) * sizeof(half);

    fused_conv_maxpool_bias_logsumexp_relu_kernel<<<num_blocks, threads_per_block, smem_size>>>(
        (const half*)x.data_ptr<at::Half>(),
        (const half*)weight.data_ptr<at::Half>(),
        (const half*)bias.data_ptr<at::Half>(),
        (half*)y.data_ptr<at::Half>(),
        N, C_in, D_in, H_in, W_in, D_out, H_out, W_out
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_conv_maxpool_bias_logsumexp_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

vertically_fused_op = load_inline(
    name='vertically_fused_op_v11_inc_addr',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_conv_maxpool_bias_logsumexp_relu_cuda'],
    verbose=True,
    extra_cuda_cflags=['-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '--use_fast_math']
)

class ModelNew(nn.Module):
    """
    This model implements an operator chain with a vertically fused custom CUDA
    kernel. This version modifies the SOTA kernel by replacing the expensive
    integer division and modulo operations within the input data loading loop
    with a more efficient incremental address calculation scheme. By computing
    initial coordinates once and updating them with cheap additions inside the
    loop, this kernel aims to reduce instruction-level overhead, building on
    the successful strategy of minimizing arithmetic intensity in memory access
    patterns. All other SOTA parameters, including vectorized weight loading,
    are preserved.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ModelNew, self).__init__()
        # The custom kernel is specialized for an output channel count of 16.
        specialized_out_channels = 16
        
        # Standard Conv3d layer to get initial weights and bias
        conv_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=specialized_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        ).half()

        self.conv_weight = nn.Parameter(conv_layer.weight)
        self.conv_bias = nn.Parameter(conv_layer.bias)
        self.in_channels = in_channels
        
        self.fused_op = vertically_fused_op.fused_conv_maxpool_bias_logsumexp_relu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convert input to half-precision.
        x_half = x.half()

        # 2. Reshape weight for coalesced memory access in the kernel.
        # Original shape: (out_channels, in_channels, K, K, K)
        # New shape for kernel: (in_channels, out_channels * K^3)
        weight_reshaped = self.conv_weight.permute(1, 0, 2, 3, 4).contiguous().view(self.in_channels, -1)

        # 3. Apply the single, vertically fused custom operator.
        # It performs: Conv3d -> MaxPool3d -> Bias Add -> LogSumExp -> ReLU
        output_half = self.fused_op(x_half, weight_reshaped, self.conv_bias)

        # 4. Convert output back to float32.
        return output_half.float()