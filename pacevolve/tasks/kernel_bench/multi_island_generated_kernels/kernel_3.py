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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os
import math

# Set CUDA architecture for A100-SXM4-40GB (Compute Capability 8.0)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# This kernel modifies the state-of-the-art 8-output-per-thread design.
# The key change is the use of the `__ldcs` (load-cache-streaming) intrinsic
# for the 128-bit vectorized memory loads in the fast path. The baseline kernel
# uses standard pointer-based loads, relying on the default cache policy (.ca),
# while a previous experiment showed that bypassing L1 with `__ldcg` hurts
# performance. `__ldcs` provides a hint to the memory system that the data is
# "streaming" with low temporal locality. This is intended to prevent the
# streaming input data from polluting the L1/L2 caches, potentially preserving
# cache resources for other data and improving overall memory system efficiency.
# This experiment tests if this explicit cache hinting strategy is more
# effective than the default policy for this memory-bound workload.
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

__global__ void fused_bias_maxpool_lse_relu_kernel_fp16_vec8_ldcs(
    const at::Half* __restrict__ x, const at::Half* __restrict__ bias, float* __restrict__ y,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out)
{
    const long long H_in_stride = W_in;
    const long long D_in_stride = H_in * W_in;
    const long long C_stride = D_in * H_in * W_in;
    const long long N_stride = C * C_stride;

    const long long num_outputs = (long long)N * D_out * H_out * W_out;

    // Each thread now computes 8 outputs.
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
         i < num_outputs;
         i += ((long long)blockDim.x * gridDim.x) * 8)
    {
        long long temp = i;
        int w_out0 = temp % W_out;
        temp /= W_out;
        int h_out = temp % H_out;
        temp /= H_out;
        int d_out = temp % D_out;
        int n = temp / D_out;

        const bool process_second = (i + 1 < num_outputs) && (w_out0 < W_out - 1);
        const bool process_third  = (i + 2 < num_outputs) && (w_out0 < W_out - 2);
        const bool process_fourth = (i + 3 < num_outputs) && (w_out0 < W_out - 3);
        const bool process_fifth  = (i + 4 < num_outputs) && (w_out0 < W_out - 4);
        const bool process_sixth  = (i + 5 < num_outputs) && (w_out0 < W_out - 5);
        const bool process_seventh= (i + 6 < num_outputs) && (w_out0 < W_out - 6);
        const bool process_eighth = (i + 7 < num_outputs) && (w_out0 < W_out - 7);

        const int d_in = d_out * 2;
        const int h_in = h_out * 2;
        const int w_in = w_out0 * 2;

        float max_val0 = -FLT_MAX; float sum_exp0 = 0.0f;
        float max_val1 = -FLT_MAX; float sum_exp1 = 0.0f;
        float max_val2 = -FLT_MAX; float sum_exp2 = 0.0f;
        float max_val3 = -FLT_MAX; float sum_exp3 = 0.0f;
        float max_val4 = -FLT_MAX; float sum_exp4 = 0.0f;
        float max_val5 = -FLT_MAX; float sum_exp5 = 0.0f;
        float max_val6 = -FLT_MAX; float sum_exp6 = 0.0f;
        float max_val7 = -FLT_MAX; float sum_exp7 = 0.0f;

        const bool fast_path = process_eighth &&
                               (d_in + 1 < D_in) &&
                               (h_in + 1 < H_in) &&
                               (w_in + 15 < W_in);

        if (fast_path) {
            // FAST PATH: 2x 128-bit vectorized loads using __ldcs intrinsic
            for (int c = 0; c < C; ++c) {
                long long base_idx = n * N_stride + c * C_stride;
                float p0 = -FLT_MAX, p1 = -FLT_MAX, p2 = -FLT_MAX, p3 = -FLT_MAX;
                float p4 = -FLT_MAX, p5 = -FLT_MAX, p6 = -FLT_MAX, p7 = -FLT_MAX;

                #pragma unroll
                for(int kd=0; kd<2; ++kd) {
                    #pragma unroll
                    for(int kh=0; kh<2; ++kh) {
                        const at::Half* x_ptr_base = x + base_idx + (d_in + kd) * D_in_stride + (h_in + kh) * H_in_stride + w_in;
                        const int4* x_ptr_vec = reinterpret_cast<const int4*>(x_ptr_base);
                        
                        // Use __ldcs for streaming loads
                        const int4 r0 = __ldcs(x_ptr_vec);
                        const int4 r1 = __ldcs(x_ptr_vec + 1);

                        const __half2* r0_h2 = reinterpret_cast<const __half2*>(&r0);
                        const __half2* r1_h2 = reinterpret_cast<const __half2*>(&r1);

                        const float2 f0 = __half22float2(r0_h2[0]); const float2 f1 = __half22float2(r0_h2[1]);
                        const float2 f2 = __half22float2(r0_h2[2]); const float2 f3 = __half22float2(r0_h2[3]);
                        const float2 f4 = __half22float2(r1_h2[0]); const float2 f5 = __half22float2(r1_h2[1]);
                        const float2 f6 = __half22float2(r1_h2[2]); const float2 f7 = __half22float2(r1_h2[3]);

                        p0 = fmaxf(p0, fmaxf(f0.x, f0.y)); p1 = fmaxf(p1, fmaxf(f1.x, f1.y));
                        p2 = fmaxf(p2, fmaxf(f2.x, f2.y)); p3 = fmaxf(p3, fmaxf(f3.x, f3.y));
                        p4 = fmaxf(p4, fmaxf(f4.x, f4.y)); p5 = fmaxf(p5, fmaxf(f5.x, f5.y));
                        p6 = fmaxf(p6, fmaxf(f6.x, f6.y)); p7 = fmaxf(p7, fmaxf(f7.x, f7.y));
                    }
                }
                const float bias_c = static_cast<float>(bias[c]);
                const float biased_val0 = p0 + bias_c; const float new_max0 = fmaxf(max_val0, biased_val0); sum_exp0 = sum_exp0 * __expf(max_val0 - new_max0) + __expf(biased_val0 - new_max0); max_val0 = new_max0;
                const float biased_val1 = p1 + bias_c; const float new_max1 = fmaxf(max_val1, biased_val1); sum_exp1 = sum_exp1 * __expf(max_val1 - new_max1) + __expf(biased_val1 - new_max1); max_val1 = new_max1;
                const float biased_val2 = p2 + bias_c; const float new_max2 = fmaxf(max_val2, biased_val2); sum_exp2 = sum_exp2 * __expf(max_val2 - new_max2) + __expf(biased_val2 - new_max2); max_val2 = new_max2;
                const float biased_val3 = p3 + bias_c; const float new_max3 = fmaxf(max_val3, biased_val3); sum_exp3 = sum_exp3 * __expf(max_val3 - new_max3) + __expf(biased_val3 - new_max3); max_val3 = new_max3;
                const float biased_val4 = p4 + bias_c; const float new_max4 = fmaxf(max_val4, biased_val4); sum_exp4 = sum_exp4 * __expf(max_val4 - new_max4) + __expf(biased_val4 - new_max4); max_val4 = new_max4;
                const float biased_val5 = p5 + bias_c; const float new_max5 = fmaxf(max_val5, biased_val5); sum_exp5 = sum_exp5 * __expf(max_val5 - new_max5) + __expf(biased_val5 - new_max5); max_val5 = new_max5;
                const float biased_val6 = p6 + bias_c; const float new_max6 = fmaxf(max_val6, biased_val6); sum_exp6 = sum_exp6 * __expf(max_val6 - new_max6) + __expf(biased_val6 - new_max6); max_val6 = new_max6;
                const float biased_val7 = p7 + bias_c; const float new_max7 = fmaxf(max_val7, biased_val7); sum_exp7 = sum_exp7 * __expf(max_val7 - new_max7) + __expf(biased_val7 - new_max7); max_val7 = new_max7;
            }
        } else {
            // SLOW PATH: Optimized with __half2 loads for boundaries
            for (int c = 0; c < C; ++c) {
                float p[8] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
                long long base_idx = n * N_stride + c * C_stride;
                #pragma unroll
                for (int kd = 0; kd < 2; ++kd) {
                    if (d_in + kd >= D_in) continue;
                    #pragma unroll
                    for (int kh = 0; kh < 2; ++kh) {
                        if (h_in + kh >= H_in) continue;
                        const at::Half* x_ptr_row = x + base_idx + (d_in + kd) * D_in_stride + (h_in + kh) * H_in_stride;
                        const __half2* x_ptr_row_h2 = reinterpret_cast<const __half2*>(x_ptr_row);
                        
                        if (w_in + 1 < W_in) { float2 f = __half22float2(x_ptr_row_h2[w_in/2]); p[0] = fmaxf(p[0], fmaxf(f.x, f.y)); }
                        else if (w_in < W_in) { p[0] = fmaxf(p[0], static_cast<float>(x_ptr_row[w_in])); }
                        
                        if (process_second) { if (w_in + 3 < W_in) { float2 f = __half22float2(x_ptr_row_h2[(w_in + 2)/2]); p[1] = fmaxf(p[1], fmaxf(f.x, f.y)); }
                        else if (w_in + 2 < W_in) { p[1] = fmaxf(p[1], static_cast<float>(x_ptr_row[w_in + 2])); } }
                        
                        if (process_third) { if (w_in + 5 < W_in) { float2 f = __half22float2(x_ptr_row_h2[(w_in + 4)/2]); p[2] = fmaxf(p[2], fmaxf(f.x, f.y)); }
                        else if (w_in + 4 < W_in) { p[2] = fmaxf(p[2], static_cast<float>(x_ptr_row[w_in + 4])); } }
                        
                        if (process_fourth) { if (w_in + 7 < W_in) { float2 f = __half22float2(x_ptr_row_h2[(w_in + 6)/2]); p[3] = fmaxf(p[3], fmaxf(f.x, f.y)); }
                        else if (w_in + 6 < W_in) { p[3] = fmaxf(p[3], static_cast<float>(x_ptr_row[w_in + 6])); } }
                        
                        if (process_fifth) { if (w_in + 9 < W_in) { float2 f = __half22float2(x_ptr_row_h2[(w_in + 8)/2]); p[4] = fmaxf(p[4], fmaxf(f.x, f.y)); }
                        else if (w_in + 8 < W_in) { p[4] = fmaxf(p[4], static_cast<float>(x_ptr_row[w_in + 8])); } }
                        
                        if (process_sixth) { if (w_in + 11 < W_in) { float2 f = __half22float2(x_ptr_row_h2[(w_in + 10)/2]); p[5] = fmaxf(p[5], fmaxf(f.x, f.y)); }
                        else if (w_in + 10 < W_in) { p[5] = fmaxf(p[5], static_cast<float>(x_ptr_row[w_in + 10])); } }
                        
                        if (process_seventh){ if (w_in + 13 < W_in) { float2 f = __half22float2(x_ptr_row_h2[(w_in + 12)/2]); p[6] = fmaxf(p[6], fmaxf(f.x, f.y)); }
                        else if (w_in + 12 < W_in) { p[6] = fmaxf(p[6], static_cast<float>(x_ptr_row[w_in + 12])); } }
                        
                        if (process_eighth) { if (w_in + 15 < W_in) { float2 f = __half22float2(x_ptr_row_h2[(w_in + 14)/2]); p[7] = fmaxf(p[7], fmaxf(f.x, f.y)); }
                        else if (w_in + 14 < W_in) { p[7] = fmaxf(p[7], static_cast<float>(x_ptr_row[w_in + 14])); } }
                    }
                }
                const float bias_c = static_cast<float>(bias[c]);
                const float biased_val0 = p[0] + bias_c; const float new_max0 = fmaxf(max_val0, biased_val0); sum_exp0 = sum_exp0 * __expf(max_val0 - new_max0) + __expf(biased_val0 - new_max0); max_val0 = new_max0;
                if (process_second) { const float biased_val1 = p[1] + bias_c; const float new_max1 = fmaxf(max_val1, biased_val1); sum_exp1 = sum_exp1 * __expf(max_val1 - new_max1) + __expf(biased_val1 - new_max1); max_val1 = new_max1; }
                if (process_third)  { const float biased_val2 = p[2] + bias_c; const float new_max2 = fmaxf(max_val2, biased_val2); sum_exp2 = sum_exp2 * __expf(max_val2 - new_max2) + __expf(biased_val2 - new_max2); max_val2 = new_max2; }
                if (process_fourth) { const float biased_val3 = p[3] + bias_c; const float new_max3 = fmaxf(max_val3, biased_val3); sum_exp3 = sum_exp3 * __expf(max_val3 - new_max3) + __expf(biased_val3 - new_max3); max_val3 = new_max3; }
                if (process_fifth)  { const float biased_val4 = p[4] + bias_c; const float new_max4 = fmaxf(max_val4, biased_val4); sum_exp4 = sum_exp4 * __expf(max_val4 - new_max4) + __expf(biased_val4 - new_max4); max_val4 = new_max4; }
                if (process_sixth)  { const float biased_val5 = p[5] + bias_c; const float new_max5 = fmaxf(max_val5, biased_val5); sum_exp5 = sum_exp5 * __expf(max_val5 - new_max5) + __expf(biased_val5 - new_max5); max_val5 = new_max5; }
                if (process_seventh){ const float biased_val6 = p[6] + bias_c; const float new_max6 = fmaxf(max_val6, biased_val6); sum_exp6 = sum_exp6 * __expf(max_val6 - new_max6) + __expf(biased_val6 - new_max6); max_val6 = new_max6; }
                if (process_eighth) { const float biased_val7 = p[7] + bias_c; const float new_max7 = fmaxf(max_val7, biased_val7); sum_exp7 = sum_exp7 * __expf(max_val7 - new_max7) + __expf(biased_val7 - new_max7); max_val7 = new_max7; }
            }
        }
        y[i] = fmaxf(max_val0 + __logf(sum_exp0), 0.0f);
        if (process_second) y[i+1] = fmaxf(max_val1 + __logf(sum_exp1), 0.0f);
        if (process_third)  y[i+2] = fmaxf(max_val2 + __logf(sum_exp2), 0.0f);
        if (process_fourth) y[i+3] = fmaxf(max_val3 + __logf(sum_exp3), 0.0f);
        if (process_fifth)  y[i+4] = fmaxf(max_val4 + __logf(sum_exp4), 0.0f);
        if (process_sixth)  y[i+5] = fmaxf(max_val5 + __logf(sum_exp5), 0.0f);
        if (process_seventh)y[i+6] = fmaxf(max_val6 + __logf(sum_exp6), 0.0f);
        if (process_eighth) y[i+7] = fmaxf(max_val7 + __logf(sum_exp7), 0.0f);
    }
}

torch::Tensor fused_op_fp16_cuda_vec8_ldcs(torch::Tensor x, torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor 'x' must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "Input 'x' must be a half tensor");
    TORCH_CHECK(x.dim() == 5, "Input tensor 'x' must be 5D");

    TORCH_CHECK(bias.is_cuda(), "Input tensor 'bias' must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "Input tensor 'bias' must be contiguous");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "Input 'bias' must be a half tensor");
    TORCH_CHECK(bias.dim() == 1, "Input tensor 'bias' must be 1D");
    TORCH_CHECK(bias.size(0) == x.size(1), "Bias size must match channel dimension of x");
    TORCH_CHECK(bias.size(0) <= 16, "This kernel assumes C <= 16");

    const int N = x.size(0);
    const int C = x.size(1);
    const int D_in = x.size(2);
    const int H_in = x.size(3);
    const int W_in = x.size(4);

    const int D_out = (D_in + 1) / 2;
    const int H_out = (H_in + 1) / 2;
    const int W_out = (W_in + 1) / 2;

    auto y_options = torch::TensorOptions().device(x.device()).dtype(torch::kFloat32);
    auto y = torch::zeros({N, 1, D_out, H_out, W_out}, y_options);

    const long long num_outputs = (long long)N * D_out * H_out * W_out;
    if (num_outputs == 0) {
        return y;
    }

    const int block_size = 512;
    const int num_blocks = std::min((int)((num_outputs + (block_size * 8) - 1) / (block_size * 8)), 4096);

    fused_bias_maxpool_lse_relu_kernel_fp16_vec8_ldcs<<<num_blocks, block_size>>>(
        x.data_ptr<at::Half>(),
        bias.data_ptr<at::Half>(),
        y.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return y;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_op_fp16_cuda_vec8_ldcs(torch::Tensor x, torch::Tensor bias);
"""

# JIT compile the CUDA and C++ code
fused_op_module = load_inline(
    name='fused_op_fp16_vec8_ldcs',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_op_fp16_cuda_vec8_ldcs'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ModelNew(nn.Module):
    """
    This model leverages a custom CUDA kernel that fuses bias-add, MaxPool3d,
    logsumexp, and ReLU. This version builds on the high-performance
    8-output-per-thread, vectorized-load design by replacing the standard global
    memory loads in the fast path with the `__ldcs` (load-cache-streaming)
    intrinsic. This provides a hint to the GPU that the data being loaded has
    low temporal locality, aiming to prevent cache pollution and improve overall
    memory system efficiency for this memory-bandwidth-bound workload.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ModelNew, self).__init__()
        # The custom kernel is optimized for out_channels=16.
        if out_channels != 16:
            raise ValueError("ModelNew is compiled for out_channels=16")

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True # Bias is handled by the custom kernel
        )
        # Convert the convolution layer to half precision
        self.conv.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is half-precision for compatibility with conv layer
        x_half = x.half()

        # Step 1: Apply convolution using the weight ONLY.
        conv_out = F.conv3d(
            x_half,
            self.conv.weight,
            None, # Bias is applied inside the custom kernel
            self.conv.stride,
            self.conv.padding
        )

        # Step 2: Apply the custom fused operation with half-precision inputs.
        # The kernel returns a float32 tensor.
        output = fused_op_module.fused_op_fp16_cuda_vec8_ldcs(conv_out, self.conv.bias)

        return output