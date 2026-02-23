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

# Set CUDA architecture for A100-SXM4-40GB.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernels.
# The changes are:
# 1. Replaced `long long` with `int` for indexing and sizing variables in 4D NHWC kernels.
#    This may reduce register pressure and use more efficient 32-bit instructions.
# 2. All related variables in the kernels and C++ dispatcher functions are also changed to `int`.
cublas_fused_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused bias add + ReLU kernel using 128-bit (float4) vectorization for 2D tensors.
// This is used by the linear layers, which are layout-agnostic after the flatten operation.
__global__ void add_bias_relu_kernel_float4(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int M,
    const int N)
{
    const int N_vec = N / 8;
    const int total_vecs = M * N_vec;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vecs; i += blockDim.x * gridDim.x) {
        const int col_vec_idx = i % N_vec;
        const float4 input_f4 = reinterpret_cast<const float4*>(input)[i];
        const float4 bias_f4 = reinterpret_cast<const float4*>(bias)[col_vec_idx];

        const __half2* input_h2_vecs = reinterpret_cast<const __half2*>(&input_f4);
        const __half2* bias_h2_vecs = reinterpret_cast<const __half2*>(&bias_f4);

        float4 result_f4;
        __half2* result_h2_vecs = reinterpret_cast<__half2*>(&result_f4);

        const __half zero_h = __float2half(0.0f);
        const __half2 zero_vec = __halves2half2(zero_h, zero_h);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            result_h2_vecs[j] = __hmax2(__hadd2(input_h2_vecs[j], bias_h2_vecs[j]), zero_vec);
        }
        reinterpret_cast<float4*>(output)[i] = result_f4;
    }
}

// Fused bias add kernel using 128-bit (float4) vectorization for 2D tensors.
__global__ void add_bias_kernel_float4(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int M,
    const int N)
{
    const int N_vec = N / 8;
    const int total_vecs = M * N_vec;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vecs; i += blockDim.x * gridDim.x) {
        const int col_vec_idx = i % N_vec;
        const float4 input_f4 = reinterpret_cast<const float4*>(input)[i];
        const float4 bias_f4 = reinterpret_cast<const float4*>(bias)[col_vec_idx];

        const __half2* input_h2_vecs = reinterpret_cast<const __half2*>(&input_f4);
        const __half2* bias_h2_vecs = reinterpret_cast<const __half2*>(&bias_f4);

        float4 result_f4;
        __half2* result_h2_vecs = reinterpret_cast<__half2*>(&result_f4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            result_h2_vecs[j] = __hadd2(input_h2_vecs[j], bias_h2_vecs[j]);
        }
        reinterpret_cast<float4*>(output)[i] = result_f4;
    }
}

// Fused bias add + ReLU kernel for 4D NHWC tensors
__global__ void add_bias_relu_4d_nhwc_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int C,
    const int total_vectors) // N*H*W*C / 8. Changed from long long to int
{
    const __half zero_h = __float2half(0.0f);
    const __half2 zero_vec = __halves2half2(zero_h, zero_h);
    const int C_vec = C / 8;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vectors; i += blockDim.x * gridDim.x) { // loop variable changed to int
        const int bias_vec_idx = i % C_vec; 
        
        const float4 input_f4 = reinterpret_cast<const float4*>(input)[i];
        const float4 bias_f4 = reinterpret_cast<const float4*>(bias)[bias_vec_idx];
        
        const __half2* input_h2_vecs = reinterpret_cast<const __half2*>(&input_f4);
        const __half2* bias_h2_vecs = reinterpret_cast<const __half2*>(&bias_f4);
        
        float4 result_f4;
        __half2* result_h2_vecs = reinterpret_cast<__half2*>(&result_f4);
        
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            result_h2_vecs[j] = __hmax2(__hadd2(input_h2_vecs[j], bias_h2_vecs[j]), zero_vec);
        }
        
        reinterpret_cast<float4*>(output)[i] = result_f4;
    }
}


// Fused Bias Add + ReLU + MaxPool 2x2 stride 2 for 4D NHWC Tensors
// Changed all long long types to int for potential performance improvement
__global__ void add_bias_relu_maxpool_2x2_s2_nhwc_kernel_float4(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int C_vec,
    const int W_out,
    const int H_out,
    const int W_in_C_vec,
    const int H_in_W_in_C_vec,
    const int total_output_vectors
) {
    const __half zero_h = __float2half(0.0f);
    const __half2 zero_vec = __halves2half2(zero_h, zero_h);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_output_vectors; i += blockDim.x * gridDim.x) {
        // Decompose linear index i to (n, h_out, w_out, c_vec)
        const int c_vec_idx = i % C_vec;
        const int spatial_out_idx = i / C_vec;
        const int w_out_idx = spatial_out_idx % W_out;
        const int nh_out_idx = spatial_out_idx / W_out;
        const int h_out_idx = nh_out_idx % H_out;
        const int n_idx = nh_out_idx / H_out;

        // Load bias vector
        const float4 bias_f4 = reinterpret_cast<const float4*>(bias)[c_vec_idx];
        const __half2* bias_h2_vecs = reinterpret_cast<const __half2*>(&bias_f4);

        // Calculate input indices for the 2x2 window
        const int h_in_base = h_out_idx * 2;
        const int w_in_base = w_out_idx * 2;
        
        const int base_idx_00 = n_idx * H_in_W_in_C_vec + h_in_base * W_in_C_vec + w_in_base * C_vec + c_vec_idx;
        const int base_idx_01 = base_idx_00 + C_vec;
        const int base_idx_10 = base_idx_00 + W_in_C_vec;
        const int base_idx_11 = base_idx_10 + C_vec;

        // Load 4 input vectors for the 2x2 window
        const float4 in_f4_00 = reinterpret_cast<const float4*>(input)[base_idx_00];
        const float4 in_f4_01 = reinterpret_cast<const float4*>(input)[base_idx_01];
        const float4 in_f4_10 = reinterpret_cast<const float4*>(input)[base_idx_10];
        const float4 in_f4_11 = reinterpret_cast<const float4*>(input)[base_idx_11];

        const __half2* in_h2_vecs_00 = reinterpret_cast<const __half2*>(&in_f4_00);
        const __half2* in_h2_vecs_01 = reinterpret_cast<const __half2*>(&in_f4_01);
        const __half2* in_h2_vecs_10 = reinterpret_cast<const __half2*>(&in_f4_10);
        const __half2* in_h2_vecs_11 = reinterpret_cast<const __half2*>(&in_f4_11);

        float4 result_f4;
        __half2* result_h2_vecs = reinterpret_cast<__half2*>(&result_f4);

        // Process 4 `__half2` vectors (8 channels)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            // Add bias and apply ReLU
            __half2 v00 = __hmax2(__hadd2(in_h2_vecs_00[j], bias_h2_vecs[j]), zero_vec);
            __half2 v01 = __hmax2(__hadd2(in_h2_vecs_01[j], bias_h2_vecs[j]), zero_vec);
            __half2 v10 = __hmax2(__hadd2(in_h2_vecs_10[j], bias_h2_vecs[j]), zero_vec);
            __half2 v11 = __hmax2(__hadd2(in_h2_vecs_11[j], bias_h2_vecs[j]), zero_vec);
            // Max pooling
            result_h2_vecs[j] = __hmax2(__hmax2(v00, v01), __hmax2(v10, v11));
        }
        
        reinterpret_cast<float4*>(output)[i] = result_f4;
    }
}

// --- Dispatcher functions ---
torch::Tensor add_bias_fused_dispatch(torch::Tensor input, torch::Tensor bias, bool apply_relu) {
    const int M = input.size(0);
    const int N = input.size(1);
    auto output = torch::empty_like(input);
    if (M * N == 0) return output;

    const int block_size = 256;
    const int num_blocks = (M * N / 8 + block_size - 1) / block_size;

    if (apply_relu) {
        add_bias_relu_kernel_float4<<<num_blocks, block_size>>>((const __half*)input.data_ptr<at::Half>(), (const __half*)bias.data_ptr<at::Half>(), (__half*)output.data_ptr<at::Half>(), M, N);
    } else {
        add_bias_kernel_float4<<<num_blocks, block_size>>>((const __half*)input.data_ptr<at::Half>(), (const __half*)bias.data_ptr<at::Half>(), (__half*)output.data_ptr<at::Half>(), M, N);
    }
    return output;
}

torch::Tensor add_bias_relu_forward_float4(torch::Tensor input, torch::Tensor bias) { return add_bias_fused_dispatch(input, bias, true); }
torch::Tensor add_bias_forward_float4(torch::Tensor input, torch::Tensor bias) { return add_bias_fused_dispatch(input, bias, false); }

torch::Tensor add_bias_relu_4d_nhwc_forward(torch::Tensor input, torch::Tensor bias) {
    const int C = input.size(1); 
    auto output = torch::empty_like(input, torch::MemoryFormat::ChannelsLast);
    if (input.numel() == 0) return output;
    
    const int total_vectors = input.numel() / 8; // Changed from long long
    const int block_size = 256;
    const int num_blocks = (total_vectors + block_size - 1) / block_size;

    add_bias_relu_4d_nhwc_kernel<<<num_blocks, block_size>>>(
        (const __half*)input.data_ptr<at::Half>(), 
        (const __half*)bias.data_ptr<at::Half>(), 
        (__half*)output.data_ptr<at::Half>(), 
        C,
        total_vectors);
    return output;
}

void launch_add_bias_relu_maxpool_4d_nhwc(torch::Tensor input, torch::Tensor bias, torch::Tensor output) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    
    // Pre-calculate strides and dimensions for the kernel
    const int C_vec = C / 8;
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;
    const int W_in_C_vec = W_in * C_vec;
    const int H_in_W_in_C_vec = H_in * W_in_C_vec;

    const int total_output_vectors = N * H_out * W_out * C_vec;
    const int block_size = 256;
    const int num_blocks = (total_output_vectors + block_size - 1) / block_size;

    add_bias_relu_maxpool_2x2_s2_nhwc_kernel_float4<<<num_blocks, block_size>>>(
        (const __half*)input.data_ptr<at::Half>(), 
        (const __half*)bias.data_ptr<at::Half>(), 
        (__half*)output.data_ptr<at::Half>(), 
        C_vec, W_out, H_out, W_in_C_vec, H_in_W_in_C_vec,
        total_output_vectors);
}

torch::Tensor forward_add_bias_relu_maxpool_4d_nhwc(torch::Tensor input, torch::Tensor bias) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;
    
    auto output = torch::empty({N, C, H_out, W_out}, input.options(), torch::MemoryFormat::ChannelsLast);
    if (input.numel() == 0) return output;

    launch_add_bias_relu_maxpool_4d_nhwc(input, bias, output);
    return output;
}

torch::Tensor forward_add_bias_relu_maxpool_flatten_4d_nhwc(torch::Tensor input, torch::Tensor bias) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;
    
    const int flattened_dim = C * H_out * W_out; // Changed from long long
    auto output = torch::empty({N, flattened_dim}, input.options());
    if (input.numel() == 0) return output;
    
    launch_add_bias_relu_maxpool_4d_nhwc(input, bias, output);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_relu", &add_bias_relu_forward_float4, "Fused Add Bias and ReLU forward (CUDA)");
  m.def("forward_bias_add", &add_bias_forward_float4, "Fused Add Bias forward (CUDA)");
  m.def("forward_add_bias_relu_4d_nhwc", &add_bias_relu_4d_nhwc_forward, "Fused Add Bias and ReLU for 4D NHWC Tensors (CUDA)");
  m.def("forward_add_bias_relu_maxpool_4d_nhwc", &forward_add_bias_relu_maxpool_4d_nhwc, "Fused Add Bias, ReLU, and MaxPool for 4D NHWC Tensors (CUDA)");
  m.def("forward_add_bias_relu_maxpool_flatten_4d_nhwc", &forward_add_bias_relu_maxpool_flatten_4d_nhwc, "Fused Add Bias, ReLU, MaxPool, and Flatten for 4D NHWC Tensors (CUDA)");
}
"""

cublas_fused_module = load_inline(
    name='cublas_fused_ops_vgg_int_indexing',
    cpp_sources=[],
    cuda_sources=[cublas_fused_source],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class FusedLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, has_relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_relu = has_relu
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
        matmul_result = F.linear(input, self.weight)
        if self.has_relu:
            return cublas_fused_module.forward_relu(matmul_result, self.bias)
        else:
            return cublas_fused_module.forward_bias_add(matmul_result, self.bias)

class FusedConv2dReLU_NHWC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        conv_output = F.conv2d(input, self.weight, bias=None, stride=1, padding=self.padding)
        return cublas_fused_module.forward_add_bias_relu_4d_nhwc(conv_output, self.bias)
        
class FusedConv2dReLUMaxPool_NHWC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        conv_output = F.conv2d(input, self.weight, bias=None, stride=1, padding=self.padding)
        return cublas_fused_module.forward_add_bias_relu_maxpool_4d_nhwc(conv_output, self.bias)

class FusedConv2dReLUMaxPoolFlatten_NHWC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        conv_output = F.conv2d(input, self.weight, bias=None, stride=1, padding=self.padding)
        return cublas_fused_module.forward_add_bias_relu_maxpool_flatten_4d_nhwc(conv_output, self.bias)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        
        classifier_in_features = 512 * 7 * 7
        classifier_hidden_features = 4096
        # The float4 vectorized kernels require the output features to be a multiple of 8.
        assert num_features % 8 == 0, "num_features must be a multiple of 8 for float4 vectorization"
        assert classifier_hidden_features % 8 == 0, "classifier_hidden_features must be a multiple of 8"
        
        # All output channels must be a multiple of 8 for the NHWC kernel
        self.features = nn.Sequential(
            FusedConv2dReLU_NHWC(3, 64, kernel_size=3, padding=1),
            FusedConv2dReLUMaxPool_NHWC(64, 64, kernel_size=3, padding=1),
            
            FusedConv2dReLU_NHWC(64, 128, kernel_size=3, padding=1),
            FusedConv2dReLUMaxPool_NHWC(128, 128, kernel_size=3, padding=1),
            
            FusedConv2dReLU_NHWC(128, 256, kernel_size=3, padding=1),
            FusedConv2dReLU_NHWC(256, 256, kernel_size=3, padding=1),
            FusedConv2dReLUMaxPool_NHWC(256, 256, kernel_size=3, padding=1),
            
            FusedConv2dReLU_NHWC(256, 512, kernel_size=3, padding=1),
            FusedConv2dReLU_NHWC(512, 512, kernel_size=3, padding=1),
            FusedConv2dReLUMaxPool_NHWC(512, 512, kernel_size=3, padding=1),
            
            FusedConv2dReLU_NHWC(512, 512, kernel_size=3, padding=1),
            FusedConv2dReLU_NHWC(512, 512, kernel_size=3, padding=1),
            # Replace the last pooling layer with the new fused flatten layer
            FusedConv2dReLUMaxPoolFlatten_NHWC(512, 512, kernel_size=3, padding=1),
        )

        self.classifier = nn.Sequential(
            FusedLinearLayer(classifier_in_features, classifier_hidden_features, has_relu=True),
            nn.Dropout(p=0.0),
            FusedLinearLayer(classifier_hidden_features, classifier_hidden_features, has_relu=True),
            nn.Dropout(p=0.0),
            FusedLinearLayer(classifier_hidden_features, num_features, has_relu=False)
        )
        
        # Convert model to half precision and channels_last memory format
        self.half().cuda().to(memory_format=torch.channels_last)
        self.eval()

        # Set up CUDA graph with channels_last input
        self.batch_size = 128
        self.static_input = torch.randn(self.batch_size, 3, 224, 224, device='cuda', dtype=torch.half)
        self.static_input = self.static_input.to(memory_format=torch.channels_last)
        self.graph = torch.cuda.CUDAGraph()
        self.static_output = None
        self._capture_graph()

    def _forward_internal(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # The flatten operation is now fused into the last layer of features
        x = self.classifier(x)
        return x

    def _capture_graph(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # Warmup runs
            for _ in range(3):
                self._forward_internal(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        with torch.cuda.graph(self.graph):
            self.static_output = self._forward_internal(self.static_input)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.batch_size:
            x_nhwc = x.to(dtype=torch.half, device='cuda', memory_format=torch.channels_last)
            return self._forward_internal(x_nhwc).float().contiguous()

        self.static_input.copy_(x.to(memory_format=torch.channels_last))
        self.graph.replay()
        return self.static_output.clone().float().contiguous()