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

# Set CUDA architecture for A100
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Combine CUDA sources for the general-purpose kernel and the new specialized kernel.
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// --- 1. General-Purpose Kernel (from SOTA for fallback) ---
// This kernel handles any valid max-pooling parameters and ensures correctness for non-specialized cases.
__global__ void max_pool2d_cuda_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    // 3D Grid: (output_width / 4, output_height, batch * channels)
    // 2D Block: (threads_x, threads_y)
    // Each thread computes a 1x4 tile of outputs
    int w_out_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc_idx = blockIdx.z;

    if (h_out >= output_height || w_out_base >= output_width) {
        return;
    }

    int c = nc_idx % channels;
    int n = nc_idx / channels;

    const float* input_ptr = input + n * channels * input_height * input_width + c * input_height * input_width;
    float* output_ptr = output + n * channels * output_height * output_width + c * output_height * output_width;

    float max_vals[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

    for (int i = 0; i < kernel_size; ++i) {
        int h_in = h_out * stride - padding + i * dilation;
        if (h_in < 0 || h_in >= input_height) {
            continue;
        }
        for (int j = 0; j < kernel_size; ++j) {
            int w_in_0 = w_out_base * stride - padding + j * dilation;
            int w_in_1 = (w_out_base + 1) * stride - padding + j * dilation;
            int w_in_2 = (w_out_base + 2) * stride - padding + j * dilation;
            int w_in_3 = (w_out_base + 3) * stride - padding + j * dilation;
            int input_idx_base = h_in * input_width;

            if (w_in_0 >= 0 && w_in_0 < input_width) max_vals[0] = fmaxf(max_vals[0], input_ptr[input_idx_base + w_in_0]);
            if (w_in_1 >= 0 && w_in_1 < input_width) max_vals[1] = fmaxf(max_vals[1], input_ptr[input_idx_base + w_in_1]);
            if (w_in_2 >= 0 && w_in_2 < input_width) max_vals[2] = fmaxf(max_vals[2], input_ptr[input_idx_base + w_in_2]);
            if (w_in_3 >= 0 && w_in_3 < input_width) max_vals[3] = fmaxf(max_vals[3], input_ptr[input_idx_base + w_in_3]);
        }
    }

    int output_idx_base = h_out * output_width;
    if (w_out_base < output_width) output_ptr[output_idx_base + w_out_base] = max_vals[0];
    if (w_out_base + 1 < output_width) output_ptr[output_idx_base + w_out_base + 1] = max_vals[1];
    if (w_out_base + 2 < output_width) output_ptr[output_idx_base + w_out_base + 2] = max_vals[2];
    if (w_out_base + 3 < output_width) output_ptr[output_idx_base + w_out_base + 3] = max_vals[3];
}

torch::Tensor max_pool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation) {
     int batch_size = x.size(0);
     int channels = x.size(1);
     int input_height = x.size(2);
     int input_width = x.size(3);
     int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
     int output_width  = (input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

     auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
     auto output = torch::empty({batch_size, channels, output_height, output_width}, options);
     if (output.numel() == 0) return output;

     dim3 threads_per_block(16, 16);
     int output_width_tiles = (output_width + 3) / 4;
     dim3 blocks(
         (output_width_tiles + threads_per_block.x - 1) / threads_per_block.x,
         (output_height + threads_per_block.y - 1) / threads_per_block.y,
         batch_size * channels
     );

     max_pool2d_cuda_kernel_vectorized<<<blocks, threads_per_block>>> (
         x.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels,
         input_height, input_width, output_height, output_width,
         kernel_size, stride, padding, dilation
     );
     
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
     return output;
}

// --- 2. NEW Specialized Kernel for 2x2, s=2 using float2 vectorization ---
// This kernel avoids shared memory and uses float2 vector loads to read the 2x2 input window
// in two transactions, aiming to maximize memory bandwidth and instruction-level parallelism.
__global__ void max_pool2d_2x2_s2_vectorized_float2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width)
{
    // Each thread computes one output pixel
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc_idx = blockIdx.z;

    if (w_out >= output_width || h_out >= output_height) {
        return;
    }

    int c = nc_idx % channels;
    int n = nc_idx / channels;

    // Pointer to the top-left of the current input feature map
    const float* input_ptr = input + (n * channels + c) * input_height * input_width;
    
    // Cast to float2 pointer for vectorized loads. Assumes input_width is even.
    const float2* input_f2 = reinterpret_cast<const float2*>(input_ptr);
    int input_width_f2 = input_width / 2;

    // Top-left corner of the 2x2 input window
    int h_in = h_out * 2;
    
    // Load the two rows of the 2x2 window using two float2 loads
    float2 top_row    = input_f2[h_in * input_width_f2 + w_out];
    float2 bottom_row = input_f2[(h_in + 1) * input_width_f2 + w_out];

    // Compute the max value
    float max_val = fmaxf(fmaxf(top_row.x, top_row.y), fmaxf(bottom_row.x, bottom_row.y));

    // Write the result
    int output_idx = (n * channels + c) * output_height * output_width + h_out * output_width + w_out;
    output[output_idx] = max_val;
}

torch::Tensor max_pool2d_2x2_s2_vectorized_cuda(torch::Tensor x) {
     int batch_size = x.size(0);
     int channels = x.size(1);
     int input_height = x.size(2);
     int input_width = x.size(3);
     
     // For k=2, s=2, p=0, d=1, output dimensions are simply halved.
     int output_height = input_height / 2;
     int output_width  = input_width / 2;
     
     // This kernel requires even input dimensions for safe float2 casting.
     // Fallback could be implemented here, but we assume valid inputs for this specialized kernel.
     AT_ASSERTM(input_width % 2 == 0, "Input width must be even for the float2 vectorized kernel.");

     auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
     auto output = torch::empty({batch_size, channels, output_height, output_width}, options);
     if (output.numel() == 0) return output;

     dim3 threads_per_block(16, 16);
     dim3 blocks(
         (output_width + threads_per_block.x - 1) / threads_per_block.x,
         (output_height + threads_per_block.y - 1) / threads_per_block.y,
         batch_size * channels
     );

     max_pool2d_2x2_s2_vectorized_float2_kernel<<<blocks, threads_per_block>>> (
         x.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels,
         input_height, input_width, output_height, output_width
     );
     
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
     return output;
}
"""

cpp_source = """
torch::Tensor max_pool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);
torch::Tensor max_pool2d_2x2_s2_vectorized_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for both kernels
max_pool_module = load_inline(
    name='max_pool_hybrid_vectorized_float2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['max_pool2d_cuda', 'max_pool2d_2x2_s2_vectorized_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Load both functions from the compiled module
        self.max_pool2d_general_cuda = max_pool_module.max_pool2d_cuda
        self.max_pool2d_specialized_cuda = max_pool_module.max_pool2d_2x2_s2_vectorized_cuda
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensor is on the correct device and is contiguous
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
             x = x.contiguous()
            
        # Dispatch to the appropriate kernel based on parameters
        if (self.kernel_size == 2 and 
            self.stride == 2 and 
            self.padding == 0 and 
            self.dilation == 1):
            # Use the faster, specialized float2 vectorized kernel
            return self.max_pool2d_specialized_cuda(x)
        else:
            # Fallback to the general-purpose kernel for all other cases
            return self.max_pool2d_general_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)