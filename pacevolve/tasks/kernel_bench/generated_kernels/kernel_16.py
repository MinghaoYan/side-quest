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

# Set CUDA architecture for A100-SXM4-40GB.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# CUDA source code for the fused depthwise convolution, batch normalization, and ReLU6 kernel.
# This version introduces stride specialization via C++ templates.
# The stride=2 path uses a register-based input patch to minimize shared memory access.
fused_dw_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_DIM 16
#define BLOCK_ROWS 16

// CUDA device function for ReLU6 activation.
__device__ inline float relu6(float x) {
    return fminf(fmaxf(0.0f, x), 6.0f);
}

template<int stride>
__global__ void fused_depthwise_conv_bn_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out) {

    // Dynamically allocated shared memory for the input tile.
    extern __shared__ float s_data[];

    // Decode batch and channel index from blockIdx.z
    const int z = blockIdx.z;
    const int n = z / C;
    const int c = z % C;

    // Output tile coordinates.
    const int out_y_base = blockIdx.y * TILE_DIM;
    const int out_x_base = blockIdx.x * TILE_DIM;

    // Thread's coordinate within the output tile.
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Corresponding input tile coordinates (top-left corner), considering padding.
    const int in_y_base = out_y_base * stride - 1;
    const int in_x_base = out_x_base * stride - 1;

    // Calculate base pointers for the current batch item and channel.
    const float* input_ptr = input + n * C * H * W + c * H * W;
    float* output_ptr = output + n * C * H_out * W_out + c * H_out * W_out;

    // Load 3x3 filter weights for the current channel into registers.
    float r_weight[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        r_weight[i] = weight[c * 9 + i];
    }

    // Determine the size of the input tile needed for the output tile.
    const int S_TILE_H = TILE_DIM * stride + 2;
    const int S_TILE_W = TILE_DIM * stride + 2;
    // Pad shared memory width to be a multiple of 4 to avoid bank conflicts.
    const int S_TILE_W_PAD = (S_TILE_W + 3) & ~3;

    // Cooperatively load input tile from global to shared memory.
    // First, zero out the shared memory to handle padding implicitly.
    const int threads_per_block = TILE_DIM * BLOCK_ROWS;
    for (int i = ty * TILE_DIM + tx; i < S_TILE_H * S_TILE_W_PAD; i += threads_per_block) {
        s_data[i] = 0.0f;
    }
    __syncthreads();

    // Load the valid data from global memory.
    for (int y_s = ty; y_s < S_TILE_H; y_s += BLOCK_ROWS) {
        int y_g = in_y_base + y_s;
        if (y_g >= 0 && y_g < H) {
            for (int x_s = tx; x_s < S_TILE_W; x_s += TILE_DIM) {
                int x_g = in_x_base + x_s;
                if (x_g >= 0 && x_g < W) {
                    s_data[y_s * S_TILE_W_PAD + x_s] = input_ptr[y_g * W + x_g];
                }
            }
        }
    }
    __syncthreads();

    // Compute convolution for the output pixel this thread is responsible for.
    const int out_y = out_y_base + ty;
    const int out_x = out_x_base + tx;

    if (out_y < H_out && out_x < W_out) {
        float acc = 0.0f;
        const int s_y_start = ty * stride;
        const int s_x_start = tx * stride;
        
        if constexpr (stride == 1) {
            // Standard path for stride=1: direct access to shared memory.
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    acc += s_data[(s_y_start + kh) * S_TILE_W_PAD + (s_x_start + kw)] * r_weight[kh * 3 + kw];
                }
            }
        } else if constexpr (stride == 2) {
            // Specialized path for stride=2: pre-load 3x3 patch into registers.
            float r_patch[9];
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    r_patch[kh*3 + kw] = s_data[(s_y_start + kh) * S_TILE_W_PAD + (s_x_start + kw)];
                }
            }
            
            #pragma unroll
            for (int i=0; i<9; ++i) {
                acc += r_patch[i] * r_weight[i];
            }
        }
        
        // Apply fused Batch Normalization and ReLU6 activation.
        float bn_val = acc * scale[c] + shift[c];
        float output_val = relu6(bn_val);
        
        output_ptr[out_y * W_out + out_x] = output_val;
    }
}

// CUDA Kernel Launcher
torch::Tensor fused_dw_conv_bn_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor shift,
    int stride) {

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto K = 3; // Kernel size
    const auto P = 1; // Padding
    
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    TORCH_CHECK(stride == 1 || stride == 2, "This custom kernel only supports stride 1 or 2.");

    const int H_out = (H + 2 * P - K) / stride + 1;
    const int W_out = (W + 2 * P - K) / stride + 1;

    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    dim3 threads_per_block(TILE_DIM, BLOCK_ROWS);
    dim3 num_blocks(
        (W_out + TILE_DIM - 1) / TILE_DIM,
        (H_out + BLOCK_ROWS - 1) / BLOCK_ROWS,
        C * N);
    
    if (stride == 1) {
        const int S_TILE_H = TILE_DIM * 1 + 2;
        const int S_TILE_W = TILE_DIM * 1 + 2;
        const int S_TILE_W_PAD = (S_TILE_W + 3) & ~3;
        const int shared_mem_size = S_TILE_H * S_TILE_W_PAD * sizeof(float);
        fused_depthwise_conv_bn_relu_kernel<1><<<num_blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
            output.data_ptr<float>(), N, C, H, W, H_out, W_out);
    } else { // stride == 2
        const int S_TILE_H = TILE_DIM * 2 + 2;
        const int S_TILE_W = TILE_DIM * 2 + 2;
        const int S_TILE_W_PAD = (S_TILE_W + 3) & ~3;
        const int shared_mem_size = S_TILE_H * S_TILE_W_PAD * sizeof(float);
        fused_depthwise_conv_bn_relu_kernel<2><<<num_blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(),
            output.data_ptr<float>(), N, C, H, W, H_out, W_out);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ source for PyTorch binding.
fused_dw_conv_cpp_source = "torch::Tensor fused_dw_conv_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor scale, torch::Tensor shift, int stride);"

# JIT compile the CUDA and C++ code.
fused_op = load_inline(
    name='fused_op_stride_specialized', # Give a new name to avoid conflicts
    cpp_sources=fused_dw_conv_cpp_source,
    cuda_sources=fused_dw_conv_source,
    functions=['fused_dw_conv_bn_relu_cuda'],
    verbose=True
)

class _FusedDWConvBNReLU(nn.Module):
    """
    Custom module for Fused Depthwise Conv -> BatchNorm -> ReLU6.
    """
    def __init__(self, conv_module, bn_module):
        super().__init__()
        self.conv = conv_module
        self.bn = bn_module

        # Pre-compute BN scale and shift for inference.
        scale = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        shift = self.bn.bias - self.bn.running_mean * scale
        self.register_buffer('bn_scale', scale.contiguous())
        self.register_buffer('bn_shift', shift.contiguous())

    def forward(self, x):
        if not self.training:
            # Use custom CUDA kernel for inference.
            return fused_op.fused_dw_conv_bn_relu_cuda(
                x.contiguous(), 
                self.conv.weight.contiguous(), 
                self.bn_scale, 
                self.bn_shift, 
                self.conv.stride[0]
            )
        else:
            # Fallback to standard PyTorch operations for training.
            return F.relu6(self.bn(self.conv(x)), inplace=True)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(round(inp * expand_ratio))
            layers = []
            if expand_ratio != 1:
                # Pointwise expansion
                layers.extend([
                    nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ])
            # Depthwise convolution
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise projection
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            return nn.Sequential(*layers)

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                original_block = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                
                # Fusion logic: replace DWConv->BN->ReLU6 sequence
                new_layers = []
                layer_iter = iter(original_block.children())
                try:
                    while True:
                        layer = next(layer_iter)
                        # Identify the depthwise convolution layer
                        if isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels and layer.in_channels > 1:
                            dw_conv = layer
                            bn_layer = next(layer_iter)
                            relu_layer = next(layer_iter)
                            
                            assert isinstance(bn_layer, nn.BatchNorm2d)
                            assert isinstance(relu_layer, nn.ReLU6)
                            
                            new_layers.append(_FusedDWConvBNReLU(dw_conv, bn_layer))
                        else:
                            new_layers.append(layer)
                except StopIteration:
                    pass
                
                features.append(nn.Sequential(*new_layers))
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_features),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x