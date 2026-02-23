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
import math
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100 to enable Tensor Core operations
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# C++ and CUDA source code for the hybrid cuBLAS + fusion kernel approach using FP16.
# This version includes the corrected fused ReLU + MaxPool2d kernel.
fp16_fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <iostream>

// --- Error Checking Macros ---
#define CUDA_CHECK(call)                                \\
  do {                                                  \\
    cudaError_t e = call;                               \\
    if (e != cudaSuccess) {                             \\
      throw std::runtime_error(cudaGetErrorString(e));  \\
    }                                                   \\
  } while (0)

// Helper to convert cublasStatus_t to string
const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "Unknown cuBLAS error";
}

#define CUBLAS_CHECK(call)                                        \\
  do {                                                            \\
    cublasStatus_t s = call;                                      \\
    if (s != CUBLAS_STATUS_SUCCESS) {                             \\
        throw std::runtime_error(cublasGetErrorString(s));        \\
    }                                                             \\
  } while (0)


// --- Fused ReLU + MaxPool2d Kernel (NHWC, FP16) ---
__global__ void fused_relu_maxpool2d_nhwc_fp16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int H_in, int W_in, int C,
    int H_out, int W_out,
    int kernel_size, int stride)
{
    const int total_output_elements = N * H_out * W_out * C;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_output_elements; i += grid_stride) {
        // Decompose linear output index to 4D coordinates
        int c = i % C;
        int w_out = (i / C) % W_out;
        int h_out = (i / (C * W_out)) % H_out;
        int n = i / (C * W_out * H_out);

        int h_start = h_out * stride;
        int w_start = w_out * stride;

        half max_val = __float2half(-65504.0f); // Smallest fp16 value

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in_idx = h_start + kh;
                int w_in_idx = w_start + kw;

                if (h_in_idx < H_in && w_in_idx < W_in) {
                    long long input_idx = (long long)n * H_in * W_in * C +
                                          (long long)h_in_idx * W_in * C +
                                          (long long)w_in_idx * C + c;
                    
                    half val = input[input_idx];
                    // Apply ReLU before comparing
                    val = __hmax(val, __float2half(0.0f)); 
                    max_val = __hmax(max_val, val);
                }
            }
        }
        output[i] = max_val;
    }
}


// --- Fused Post-GEMM Kernels (FP16, Vectorized, Unrolled, 512 Threads) ---

__device__ unsigned int xorshift32_dev(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float random_uniform_dev(unsigned int& state) {
    return float(xorshift32_dev(state)) / 4294967295.0f;
}

__global__ void __launch_bounds__(512) add_bias_relu_dropout_kernel_fp16(
    const __half* __restrict__ gemm_output,
    const __half* __restrict__ bias,
    __half* __restrict__ final_output,
    int batch_size,
    int out_features,
    float p,
    unsigned long long seed)
{
    const int total_half2_elements = (batch_size * out_features) / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    unsigned int prng_state = idx + (unsigned int)seed;
    if (prng_state == 0) prng_state = 1;

    const float inv_p = (p > 0.0f) ? (1.0f / (1.0f - p)) : 0.0f;
    const half2 zero_h2 = __float2half2_rn(0.0f);
    
    const int UNROLL_FACTOR = 4;
    const int unrolled_stride = stride * UNROLL_FACTOR;

    // Unrolled loop
    for (; idx + stride * (UNROLL_FACTOR - 1) < total_half2_elements; idx += unrolled_stride) {
        #pragma unroll
        for (int i=0; i < UNROLL_FACTOR; ++i) {
            int current_idx = idx + i * stride;
            const int col_base = (current_idx * 2) % out_features;
            half2 gemm_val = ((const half2*)gemm_output)[current_idx];
            half2 bias_val = ((const half2*)bias)[col_base / 2];
            half2 biased_val = __hadd2(gemm_val, bias_val);
            half2 relu_val = __hmax2(biased_val, zero_h2);
            if (p > 0.0f) {
                float v1 = __half2float(relu_val.x) * (float)(random_uniform_dev(prng_state) > p) * inv_p;
                float v2 = __half2float(relu_val.y) * (float)(random_uniform_dev(prng_state) > p) * inv_p;
                relu_val = __floats2half2_rn(v1, v2);
            }
            ((half2*)final_output)[current_idx] = relu_val;
        }
    }
    
    // Cleanup loop
    for (; idx < total_half2_elements; idx += stride) {
        const int col_base = (idx * 2) % out_features;
        half2 gemm_val = ((const half2*)gemm_output)[idx];
        half2 bias_val = ((const half2*)bias)[col_base / 2];
        half2 biased_val = __hadd2(gemm_val, bias_val);
        half2 relu_val = __hmax2(biased_val, zero_h2);
        if (p > 0.0f) {
            float v1 = __half2float(relu_val.x) * (float)(random_uniform_dev(prng_state) > p) * inv_p;
            float v2 = __half2float(relu_val.y) * (float)(random_uniform_dev(prng_state) > p) * inv_p;
            relu_val = __floats2half2_rn(v1, v2);
        }
        ((half2*)final_output)[idx] = relu_val;
    }
}


__global__ void __launch_bounds__(512) add_bias_and_cast_to_fp32_kernel_fp16(
    const __half* __restrict__ gemm_output,
    const __half* __restrict__ bias,
    float* __restrict__ final_output, // Output is FP32
    int total_elements,
    int out_features)
{
    const int total_half2_elements = total_elements / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int UNROLL_FACTOR = 4;
    const int unrolled_stride = stride * UNROLL_FACTOR;

    // Unrolled loop
    for (; idx + stride * (UNROLL_FACTOR - 1) < total_half2_elements; idx += unrolled_stride) {
        #pragma unroll
        for(int i=0; i<UNROLL_FACTOR; ++i) {
            int current_idx = idx + i * stride;
            half2 gemm_val = ((const half2*)gemm_output)[current_idx];
            half2 bias_val = ((const half2*)bias)[((current_idx * 2) % out_features) / 2];
            float2 out_f32 = __half22float2(__hadd2(gemm_val, bias_val));
            final_output[current_idx*2] = out_f32.x;
            final_output[current_idx*2 + 1] = out_f32.y;
        }
    }

    // Cleanup loop
    for (; idx < total_half2_elements; idx += stride) {
        const int col_base = (idx * 2) % out_features;
        half2 gemm_val = ((const half2*)gemm_output)[idx];
        half2 bias_val = ((const half2*)bias)[col_base / 2];
        half2 biased_val = __hadd2(gemm_val, bias_val);
        float2 out_f32 = __half22float2(biased_val);
        final_output[idx*2]     = out_f32.x;
        final_output[idx*2 + 1] = out_f32.y;
    }
}


// --- cuBLAS Handle Management ---

static cublasHandle_t cublas_handle;
static bool cublas_handle_initialized = false;

void initialize_cublas() {
    if (!cublas_handle_initialized) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
        cublas_handle_initialized = true;
    }
}

// --- Main Wrapper Functions ---
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void gemm_fp16(cublasHandle_t handle, torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int m = B.size(0);
    const int k = B.size(1);
    const int n = A.size(1);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k,
                              &alpha,
                              A.data_ptr(), CUDA_R_16F, n,
                              B.data_ptr(), CUDA_R_16F, k,
                              &beta,
                              C.data_ptr(), CUDA_R_16F, n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

torch::Tensor cublas_fused_linear_relu_dropout_fp16(
    torch::Tensor input, torch::Tensor weight_t, torch::Tensor bias, float p, unsigned long long seed)
{
    CHECK_INPUT(input); CHECK_INPUT(weight_t); CHECK_INPUT(bias);
    const int batch_size = input.size(0);
    const int out_features = bias.size(0);
    
    initialize_cublas();
    auto gemm_output = torch::empty({batch_size, out_features}, input.options());
    gemm_fp16(cublas_handle, weight_t, input, gemm_output);

    auto final_output = torch::empty({batch_size, out_features}, input.options());
    const int threads_per_block = 512;
    const int total_half2_elements = (batch_size * out_features) / 2;
    const int num_blocks = (total_half2_elements + threads_per_block - 1) / threads_per_block;

    add_bias_relu_dropout_kernel_fp16<<<num_blocks, threads_per_block>>>(
        (const __half*)gemm_output.data_ptr(), (const __half*)bias.data_ptr(),
        (__half*)final_output.data_ptr(), batch_size, out_features, p, seed);
    CUDA_CHECK(cudaGetLastError());

    return final_output;
}

torch::Tensor cublas_fused_flatten_linear_relu_dropout_fp16(
    torch::Tensor input, torch::Tensor weight_t, torch::Tensor bias, float p, unsigned long long seed) {
    
    const int batch_size = input.size(0);
    const int in_features = input.numel() / batch_size;
    auto reshaped_input = input.reshape({batch_size, in_features});
    return cublas_fused_linear_relu_dropout_fp16(reshaped_input, weight_t, bias, p, seed);
}

torch::Tensor cublas_fused_linear_fp16_to_fp32(
    torch::Tensor input, torch::Tensor weight_t, torch::Tensor bias)
{
    CHECK_INPUT(input); CHECK_INPUT(weight_t); CHECK_INPUT(bias);
    const int batch_size = input.size(0);
    const int out_features = bias.size(0);

    initialize_cublas();
    auto gemm_output = torch::empty({batch_size, out_features}, input.options());
    gemm_fp16(cublas_handle, weight_t, input, gemm_output);
    
    auto final_output = torch::empty({batch_size, out_features}, input.options().dtype(torch::kFloat));
    const int total_elements = batch_size * out_features;
    const int threads_per_block = 512;
    const int total_half2_elements = total_elements / 2;
    const int num_blocks = (total_half2_elements + threads_per_block - 1) / threads_per_block;
    
    add_bias_and_cast_to_fp32_kernel_fp16<<<num_blocks, threads_per_block>>>(
        (const __half*)gemm_output.data_ptr(), (const __half*)bias.data_ptr(),
        (float*)final_output.data_ptr(), total_elements, out_features);
    CUDA_CHECK(cudaGetLastError());
    
    return final_output;
}

torch::Tensor fused_relu_maxpool2d_fp16(
    torch::Tensor input, int kernel_size, int stride)
{
    TORCH_CHECK(input.suggest_memory_format() == torch::MemoryFormat::ChannelsLast, "Input tensor must be in channels_last format");
    CHECK_CUDA(input);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int H_out = (H_in - kernel_size) / stride + 1;
    const int W_out = (W_in - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options().memory_format(torch::MemoryFormat::ChannelsLast));
    
    const int total_output_elements = N * H_out * W_out * C;
    if (total_output_elements == 0) return output;

    const int threads_per_block = 512;
    const int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;

    fused_relu_maxpool2d_nhwc_fp16_kernel<<<num_blocks, threads_per_block>>>(
        (const __half*)input.data_ptr(),
        (__half*)output.data_ptr(),
        N, H_in, W_in, C, H_out, W_out,
        kernel_size, stride);
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
"""

fp16_fused_ops_cpp_source = """
#include <torch/extension.h>

// Classifier functions
torch::Tensor cublas_fused_linear_relu_dropout_fp16(torch::Tensor input, torch::Tensor weight_t, torch::Tensor bias, float p, unsigned long long seed);
torch::Tensor cublas_fused_flatten_linear_relu_dropout_fp16(torch::Tensor input, torch::Tensor weight_t, torch::Tensor bias, float p, unsigned long long seed);
torch::Tensor cublas_fused_linear_fp16_to_fp32(torch::Tensor input, torch::Tensor weight_t, torch::Tensor bias);

// Feature extractor function
torch::Tensor fused_relu_maxpool2d_fp16(torch::Tensor input, int kernel_size, int stride);
"""

# Compile the inline CUDA code
fp16_fused_ops = load_inline(
    name='cublas_fused_ops_fp16_relu_maxpool_fixed',
    cpp_sources=fp16_fused_ops_cpp_source,
    cuda_sources=fp16_fused_ops_cuda_source,
    functions=['cublas_fused_linear_relu_dropout_fp16', 'cublas_fused_flatten_linear_relu_dropout_fp16', 'cublas_fused_linear_fp16_to_fp32', 'fused_relu_maxpool2d_fp16'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_ldflags=['-lcublas'],
    verbose=False
)

class FusedReLUMaxPool2dFP16(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        return fp16_fused_ops.fused_relu_maxpool2d_fp16(input, self.kernel_size, self.stride)

class FusedLinearReLUDropoutFP16(nn.Module):
    def __init__(self, in_features, out_features, p=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).half())
        self.bias = nn.Parameter(torch.Tensor(out_features).half())
        self.reset_parameters()

    def reset_parameters(self):
        temp_weight = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(temp_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        temp_bias = torch.empty(self.out_features).uniform_(-bound, bound)
        with torch.no_grad():
            self.weight.copy_(temp_weight.T.half())
            self.bias.copy_(temp_bias.half())

    def forward(self, input):
        dropout_p = self.p if self.training else 0.0
        seed = torch.randint(2**63 - 1, (1,)).item()
        return fp16_fused_ops.cublas_fused_linear_relu_dropout_fp16(input, self.weight, self.bias, dropout_p, seed)


class FusedFlattenLinearReLUDropoutFP16(nn.Module):
    def __init__(self, in_channels, h, w, out_features, p=0.0):
        super().__init__()
        self.in_features = in_channels * h * w
        self.out_features = out_features
        self.p = p
        self.weight = nn.Parameter(torch.Tensor(self.in_features, out_features).half())
        self.bias = nn.Parameter(torch.Tensor(out_features).half())
        self.reset_parameters()
    
    def reset_parameters(self):
        temp_weight = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(temp_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        temp_bias = torch.empty(self.out_features).uniform_(-bound, bound)
        with torch.no_grad():
            self.weight.copy_(temp_weight.T.half())
            self.bias.copy_(temp_bias.half())

    def forward(self, input):
        input_contiguous = input.contiguous()
        dropout_p = self.p if self.training else 0.0
        seed = torch.randint(2**63 - 1, (1,)).item()
        return fp16_fused_ops.cublas_fused_flatten_linear_relu_dropout_fp16(input_contiguous, self.weight, self.bias, dropout_p, seed)

class FusedLinearFP16ToFP32(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).half())
        self.bias = nn.Parameter(torch.Tensor(out_features).half())
        self.reset_parameters()

    def reset_parameters(self):
        temp_weight = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(temp_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        temp_bias = torch.empty(self.out_features).uniform_(-bound, bound)
        with torch.no_grad():
            self.weight.copy_(temp_weight.T.half())
            self.bias.copy_(temp_bias.half())

    def forward(self, input):
        return fp16_fused_ops.cublas_fused_linear_fp16_to_fp32(input, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 1000):
        super(ModelNew, self).__init__()
        if num_features % 2 != 0:
            num_features += 1
        
        num_classes = num_features
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            FusedReLUMaxPool2dFP16(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            FusedReLUMaxPool2dFP16(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            FusedReLUMaxPool2dFP16(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            FusedFlattenLinearReLUDropoutFP16(in_channels=256, h=6, w=6, out_features=4096, p=0.0),
            FusedLinearReLUDropoutFP16(in_features=4096, out_features=4096, p=0.0),
            FusedLinearFP16ToFP32(in_features=4096, out_features=num_classes),
        )

        self.features.half().to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half().to(memory_format=torch.channels_last)
        x = self.features(x)
        x = self.classifier(x)
        return x
