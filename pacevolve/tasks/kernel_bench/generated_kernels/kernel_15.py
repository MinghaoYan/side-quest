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

# Set CUDA architecture for A100-SXM4-40GB, which has compute capability 8.0
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# --- KERNEL 1: Fused Bias + ReLU for Linear Layers (FP16) - UNCHANGED ---
fused_bias_relu_fp16_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector_types.h> // For float4

// Each thread computes 8 half-precision elements (using float4). Block size is 256.
__global__ __launch_bounds__(256, 2) void fused_bias_relu_2d_fp16_kernel(
    const __half* __restrict__ input,      // [batch_size, out_features]
    const __half* __restrict__ bias,       // [out_features]
    __half* __restrict__ output,           // [batch_size, out_features]
    const int out_features_h8) {         // out_features / 8

    const int col_h8 = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (col_h8 < out_features_h8) {
        const int linear_idx_h8 = row * out_features_h8 + col_h8;
        
        // Load 8 halfs as a float4 for 128-bit memory transaction
        const float4 input_f4 = ((const float4*)input)[linear_idx_h8];
        const float4 bias_f4  = ((const float4*)bias)[col_h8];

        // Process as 4 __half2 vectors
        const __half2* input_h2 = reinterpret_cast<const __half2*>(&input_f4);
        const __half2* bias_h2  = reinterpret_cast<const __half2*>(&bias_f4);
        
        __half2 result_h2[4];
        const __half2 zero_vec = __float2half2_rn(0.0f);

        #pragma unroll
        for (int i=0; i<4; ++i) {
            result_h2[i] = __hmax2(__hadd2(input_h2[i], bias_h2[i]), zero_vec);
        }

        // Store the result back as a float4
        ((float4*)output)[linear_idx_h8] = *reinterpret_cast<float4*>(result_h2);
    }
}

torch::Tensor fused_bias_relu_fp16_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kHalf && bias.scalar_type() == torch::kHalf, "Tensors must be FP16");
    TORCH_CHECK(input.dim() == 2 && bias.dim() == 1, "Tensor dimension mismatch");
    TORCH_CHECK(input.size(1) == bias.size(0), "Dimension mismatch");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(input.size(1) % 8 == 0, "out_features must be divisible by 8 for float4 vectorization");

    const int batch_size = input.size(0);
    const int out_features = input.size(1);
    const int out_features_h8 = out_features / 8;
    auto output = torch::empty_like(input);

    const dim3 blockSize(256, 1);
    const dim3 gridSize( (out_features_h8 + blockSize.x - 1) / blockSize.x, batch_size );

    fused_bias_relu_2d_fp16_kernel<<<gridSize, blockSize>>>(
        (const __half*)input.data_ptr(), (const __half*)bias.data_ptr(), (__half*)output.data_ptr(), out_features_h8 );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}
"""

# --- KERNEL 2: Fused Bias + ReLU for 4D Tensors (FP16) - UNCHANGED ---
fused_bias_relu_4d_fp16_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector_types.h>

__global__ void fused_bias_relu_4d_fp16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int num_elements,
    const int C,
    const int plane_size) {

    // Vectorized main loop using a grid-stride loop for float4 (8 halfs)
    const int num_vec_elements = num_elements / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vec_elements;
         i += gridDim.x * blockDim.x) {
        
        const int linear_idx_base = i * 8;
        
        // Load 8 halfs as a float4
        const float4 in_f4 = ((const float4*)input)[i];
        
        // Unpack to process element-wise, ensuring correct bias is used
        const __half* in_h = reinterpret_cast<const __half*>(&in_f4);
        
        float4 out_f4;
        __half* out_h = reinterpret_cast<__half*>(&out_f4);

        const __half zero = __float2half_rn(0.0f);

        // Unroll the loop for the 8 elements within the float4
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int current_idx = linear_idx_base + k;
            const int c_idx = (current_idx / plane_size) % C; 
            const __half bias_val = bias[c_idx];
            out_h[k] = __hmax(__hadd(in_h[k], bias_val), zero);
        }
        
        // Store 8 halfs as a float4
        ((float4*)output)[i] = out_f4;
    }

    // Scalar cleanup loop for remaining elements (if num_elements is not divisible by 8)
    const int start_cleanup_idx = num_vec_elements * 8;
    for (int idx = start_cleanup_idx + blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_elements;
         idx += gridDim.x * blockDim.x) {
        
        const int c_idx = (idx / plane_size) % C;
        const __half bias_val = bias[c_idx];
        const __half input_val = input[idx];
        output[idx] = __hmax(__hadd(input_val, bias_val), __float2half_rn(0.0f));
    }
}

torch::Tensor fused_bias_relu_4d_fp16_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kHalf && bias.scalar_type() == torch::kHalf, "Tensors must be FP16");
    TORCH_CHECK(input.dim() == 4 && bias.dim() == 1, "Dimension mismatch");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input channels must match bias size");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "Inputs must be contiguous");

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int num_elements = B * C * H * W;
    const int plane_size = H * W;

    auto output = torch::empty_like(input);

    if (num_elements == 0) {
        return output;
    }

    const int block_size = 512;
    // Launch enough blocks to cover all elements, can be tuned.
    const int grid_size = std::min(1024, (num_elements + block_size - 1) / block_size);

    fused_bias_relu_4d_fp16_kernel<<<grid_size, block_size>>>(
        (const __half*)input.data_ptr(),
        (const __half*)bias.data_ptr(),
        (__half*)output.data_ptr(),
        num_elements, C, plane_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return output;
}
"""

# --- KERNEL 3: Fused Bias + ReLU + MaxPool2d (k=3, s=2) (FP16) - Interleaved ILP version ---
fused_bias_relu_maxpool_k3s2_fp16_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat> 
#include <stdexcept>

// Each block processes 4 channels for a given output tile (out_y, out_x)
#define CHANNELS_PER_BLOCK 4

__global__ void fused_bias_relu_maxpool_k3s2_fp16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int C,
    const int H_in, const int W_in,
    const int H_out, const int W_out) {

    // Each thread calculates one output pixel (out_y, out_x) for a group of 4 channels
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_y >= H_out || out_x >= W_out) {
        return;
    }

    const int C_groups = C / CHANNELS_PER_BLOCK;
    const int group_idx = blockIdx.z;
    const int b_idx = group_idx / C_groups;
    const int start_c = (group_idx % C_groups) * CHANNELS_PER_BLOCK;

    // Base pointers for this batch
    const int B_stride_in = C * H_in * W_in;
    const int B_stride_out = C * H_out * W_out;
    const __half* input_batch = input + b_idx * B_stride_in;
    __half* output_batch = output + b_idx * B_stride_out;

    const __half NEG_INF = __float2half_rn(-FLT_MAX);
    const __half zero_h = __float2half_rn(0.0f);
    
    // --- Interleaved version ---
    // Initialize max values and pointers for all 4 channels
    __half max_vals[CHANNELS_PER_BLOCK] = {NEG_INF, NEG_INF, NEG_INF, NEG_INF};
    
    const int C_stride_in = H_in * W_in;
    const __half* input_plane0 = input_batch + (start_c + 0) * C_stride_in;
    const __half* input_plane1 = input_batch + (start_c + 1) * C_stride_in;
    const __half* input_plane2 = input_batch + (start_c + 2) * C_stride_in;
    const __half* input_plane3 = input_batch + (start_c + 3) * C_stride_in;

    const __half bias0 = bias[start_c + 0];
    const __half bias1 = bias[start_c + 1];
    const __half bias2 = bias[start_c + 2];
    const __half bias3 = bias[start_c + 3];

    const int in_start_y = out_y * 2;
    const int in_start_x = out_x * 2;

    // Loop over the 3x3 stencil once, computing for all 4 channels inside
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int in_y = in_start_y + r;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int in_x = in_start_x + c;
            if (in_y < H_in && in_x < W_in) {
                const int idx = in_y * W_in + in_x;
                __half val;

                // Channel 0
                val = __hmax(__hadd(input_plane0[idx], bias0), zero_h);
                max_vals[0] = __hmax(max_vals[0], val);
                
                // Channel 1
                val = __hmax(__hadd(input_plane1[idx], bias1), zero_h);
                max_vals[1] = __hmax(max_vals[1], val);

                // Channel 2
                val = __hmax(__hadd(input_plane2[idx], bias2), zero_h);
                max_vals[2] = __hmax(max_vals[2], val);

                // Channel 3
                val = __hmax(__hadd(input_plane3[idx], bias3), zero_h);
                max_vals[3] = __hmax(max_vals[3], val);
            }
        }
    }
    
    // Write the results to the output tensor
    const int C_stride_out = H_out * W_out;
    const int base_out_idx = out_y * W_out + out_x;
    output_batch[(start_c + 0) * C_stride_out + base_out_idx] = max_vals[0];
    output_batch[(start_c + 1) * C_stride_out + base_out_idx] = max_vals[1];
    output_batch[(start_c + 2) * C_stride_out + base_out_idx] = max_vals[2];
    output_batch[(start_c + 3) * C_stride_out + base_out_idx] = max_vals[3];
}

torch::Tensor fused_bias_relu_maxpool_k3s2_fp16_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kHalf && bias.scalar_type() == torch::kHalf, "Tensors must be FP16");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (NCHW)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input channels must match bias size");
    TORCH_CHECK(input.size(1) % 4 == 0, "Input channels must be divisible by 4 for multi-channel kernel");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "Inputs must be contiguous");

    const int B = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    
    const int H_out = (H_in - 3) / 2 + 1;
    const int W_out = (W_in - 3) / 2 + 1;

    auto output = torch::empty({B, C, H_out, W_out}, input.options());
    
    const int C_groups = C / 4;

    const dim3 block_size(16, 16); 
    const dim3 grid_size(
        (W_out + block_size.x - 1) / block_size.x,
        (H_out + block_size.y - 1) / block_size.y,
        B * C_groups
    );

    fused_bias_relu_maxpool_k3s2_fp16_kernel<<<grid_size, block_size>>>(
        (const __half*)input.data_ptr(), (const __half*)bias.data_ptr(), (__half*)output.data_ptr(),
        C, H_in, W_in, H_out, W_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# --- KERNEL 4: Fused Bias + ReLU + MaxPool2d (k=3, s=2) + Flatten (FP16) - Interleaved ILP version ---
fused_bias_relu_maxpool_flatten_k3s2_fp16_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat> 
#include <stdexcept>

// Each block processes 4 channels for a given output tile (out_y, out_x)
#define CHANNELS_PER_BLOCK 4

__global__ void fused_bias_relu_maxpool_k3s2_flatten_fp16_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int C,
    const int H_in, const int W_in,
    const int H_out, const int W_out) {

    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_y >= H_out || out_x >= W_out) {
        return;
    }

    const int C_groups = C / CHANNELS_PER_BLOCK;
    const int group_idx = blockIdx.z;
    const int b_idx = group_idx / C_groups;
    const int start_c = (group_idx % C_groups) * CHANNELS_PER_BLOCK;

    const int B_stride_in = C * H_in * W_in;
    const __half* input_batch = input + b_idx * B_stride_in;

    const int out_plane_size = C * H_out * W_out;
    __half* output_batch = output + b_idx * out_plane_size;

    const __half NEG_INF = __float2half_rn(-FLT_MAX);
    const __half zero_h = __float2half_rn(0.0f);

    // --- Interleaved version ---
    __half max_vals[CHANNELS_PER_BLOCK] = {NEG_INF, NEG_INF, NEG_INF, NEG_INF};
    
    const int C_stride_in = H_in * W_in;
    const __half* input_plane0 = input_batch + (start_c + 0) * C_stride_in;
    const __half* input_plane1 = input_batch + (start_c + 1) * C_stride_in;
    const __half* input_plane2 = input_batch + (start_c + 2) * C_stride_in;
    const __half* input_plane3 = input_batch + (start_c + 3) * C_stride_in;

    const __half bias0 = bias[start_c + 0];
    const __half bias1 = bias[start_c + 1];
    const __half bias2 = bias[start_c + 2];
    const __half bias3 = bias[start_c + 3];

    const int in_start_y = out_y * 2;
    const int in_start_x = out_x * 2;

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int in_y = in_start_y + r;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int in_x = in_start_x + c;
            if (in_y < H_in && in_x < W_in) {
                const int idx = in_y * W_in + in_x;
                __half val;

                val = __hmax(__hadd(input_plane0[idx], bias0), zero_h);
                max_vals[0] = __hmax(max_vals[0], val);
                
                val = __hmax(__hadd(input_plane1[idx], bias1), zero_h);
                max_vals[1] = __hmax(max_vals[1], val);

                val = __hmax(__hadd(input_plane2[idx], bias2), zero_h);
                max_vals[2] = __hmax(max_vals[2], val);

                val = __hmax(__hadd(input_plane3[idx], bias3), zero_h);
                max_vals[3] = __hmax(max_vals[3], val);
            }
        }
    }
        
    // Calculate flattened output index
    const int in_channel_plane_size = H_out * W_out;
    const int base_out_idx = out_y * W_out + out_x;
    output_batch[(start_c + 0) * in_channel_plane_size + base_out_idx] = max_vals[0];
    output_batch[(start_c + 1) * in_channel_plane_size + base_out_idx] = max_vals[1];
    output_batch[(start_c + 2) * in_channel_plane_size + base_out_idx] = max_vals[2];
    output_batch[(start_c + 3) * in_channel_plane_size + base_out_idx] = max_vals[3];
}

torch::Tensor fused_bias_relu_maxpool_k3s2_flatten_fp16_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kHalf && bias.scalar_type() == torch::kHalf, "Tensors must be FP16");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (NCHW)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input channels must match bias size");
    TORCH_CHECK(input.size(1) % 4 == 0, "Input channels must be divisible by 4 for multi-channel kernel");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "Inputs must be contiguous");

    const int B = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    
    const int H_out = (H_in - 3) / 2 + 1;
    const int W_out = (W_in - 3) / 2 + 1;

    auto output = torch::empty({B, C * H_out * W_out}, input.options());

    const int C_groups = C / 4;

    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (W_out + block_size.x - 1) / block_size.x,
        (H_out + block_size.y - 1) / block_size.y,
        B * C_groups
    );

    fused_bias_relu_maxpool_k3s2_flatten_fp16_kernel<<<grid_size, block_size>>>(
        (const __half*)input.data_ptr(), (const __half*)bias.data_ptr(), (__half*)output.data_ptr(),
        C, H_in, W_in, H_out, W_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# --- C++ Wrappers ---
combined_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_bias_relu_fp16_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor fused_bias_relu_4d_fp16_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor fused_bias_relu_maxpool_k3s2_fp16_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor fused_bias_relu_maxpool_k3s2_flatten_fp16_cuda(torch::Tensor input, torch::Tensor bias);
"""

# --- Inline Compilation ---
fp16_ops_ilp_maxpool = load_inline(
    name='fp16_ops_ilp_maxpool',
    cpp_sources=combined_cpp_source,
    cuda_sources=[
        fused_bias_relu_fp16_source, 
        fused_bias_relu_4d_fp16_source, 
        fused_bias_relu_maxpool_k3s2_fp16_source,
        fused_bias_relu_maxpool_flatten_k3s2_fp16_source
    ],
    functions=[
        'fused_bias_relu_fp16_cuda', 
        'fused_bias_relu_4d_fp16_cuda', 
        'fused_bias_relu_maxpool_k3s2_fp16_cuda',
        'fused_bias_relu_maxpool_k3s2_flatten_fp16_cuda'
    ],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

# --- PyTorch Modules ---
class FusedLinearReLUDropoutFP16Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output_matmul = torch.matmul(input, weight.t())
        output = fp16_ops_ilp_maxpool.fused_bias_relu_fp16_cuda(output_matmul, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented")

class FusedLinearReLUDropoutFP16(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return FusedLinearReLUDropoutFP16Function.apply(input, self.weight, self.bias)

class FusedBiasAddReLU_4D_FP16(nn.Module):
    def forward(self, input, bias):
        return fp16_ops_ilp_maxpool.fused_bias_relu_4d_fp16_cuda(input.contiguous(), bias.contiguous())

class FusedBiasAddReLU_CustomMaxPool2d_FP16(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        if kernel_size != 3 or stride != 2:
            raise ValueError("This custom op is hardcoded for kernel_size=3 and stride=2.")

    def forward(self, input, bias):
        return fp16_ops_ilp_maxpool.fused_bias_relu_maxpool_k3s2_fp16_cuda(input.contiguous(), bias.contiguous())

class FusedBiasAddReLU_CustomMaxPool2d_Flatten_FP16(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        if kernel_size != 3 or stride != 2:
            raise ValueError("This custom op is hardcoded for kernel_size=3 and stride=2.")

    def forward(self, input, bias):
        return fp16_ops_ilp_maxpool.fused_bias_relu_maxpool_k3s2_flatten_fp16_cuda(input.contiguous(), bias.contiguous())

# --- Main Model ---
class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        # Conv layers must have output channels divisible by 4 for the multi-channel kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        
        # Standard fused op for first two conv layers
        self.bias_relu_maxpool_fp16 = FusedBiasAddReLU_CustomMaxPool2d_FP16(kernel_size=3, stride=2)
        # Fused op without pooling for middle two conv layers
        self.bias_relu_fp16 = FusedBiasAddReLU_4D_FP16()
        # New fused op with flatten for the final conv layer
        self.bias_relu_maxpool_flatten_fp16 = FusedBiasAddReLU_CustomMaxPool2d_Flatten_FP16(kernel_size=3, stride=2)
        
        self.fc1_fused = FusedLinearReLUDropoutFP16(in_features=9216, out_features=4096)
        self.fc2_fused = FusedLinearReLUDropoutFP16(in_features=4096, out_features=4096)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_features)
        
        self.half()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        
        conv1_out = F.conv2d(x, self.conv1.weight, None, self.conv1.stride, self.conv1.padding)
        x = self.bias_relu_maxpool_fp16(conv1_out, self.conv1.bias)
        
        conv2_out = F.conv2d(x, self.conv2.weight, None, self.conv2.stride, self.conv2.padding)
        x = self.bias_relu_maxpool_fp16(conv2_out, self.conv2.bias)
        
        conv3_out = F.conv2d(x, self.conv3.weight, None, self.conv3.stride, self.conv3.padding)
        x = self.bias_relu_fp16(conv3_out, self.conv3.bias)
        
        conv4_out = F.conv2d(x, self.conv4.weight, None, self.conv4.stride, self.conv4.padding)
        x = self.bias_relu_fp16(conv4_out, self.conv4.bias)
        
        conv5_out = F.conv2d(x, self.conv5.weight, None, self.conv5.stride, self.conv5.padding)
        # Apply the new fused kernel that includes flatten
        x = self.bias_relu_maxpool_flatten_fp16(conv5_out, self.conv5.bias)
        
        # The flatten operation is no longer needed here
        # x = torch.flatten(x, 1)
        
        x = self.fc1_fused(x)
        x = self.fc2_fused(x)
        
        x = self.fc3(x)
        
        return x.float()