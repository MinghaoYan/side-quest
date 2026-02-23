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

# Set CUDA architecture for A100-SXM4-40GB.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernel and C++ wrapper
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_DIM: The side length of the square tiles used for matrix multiplication.
// BLOCK_ROWS/COLS: The dimensions of the thread block.
// Each thread computes a (WPT_ROWS x WPT_COLS) sub-matrix of the output tile.
#define TILE_DIM 32
#define BLOCK_ROWS 16
#define BLOCK_COLS 16
#define WPT_ROWS (TILE_DIM / BLOCK_ROWS) // Work per thread Rows = 2
#define WPT_COLS (TILE_DIM / BLOCK_COLS) // Work per thread Cols = 2

// Fused kernel for Linear (GEMM) + Bias + ReLU using 2D shared memory tiling
// and float4 vectorized loads for high memory bandwidth.
__global__ void tiled_gemm_relu_kernel(
    const float* __restrict__ A, // Input matrix (M, K)
    const float* __restrict__ B, // Transposed Weight matrix (K, N)
    const float* __restrict__ bias, // Bias vector (N)
    float* __restrict__ C, // Output matrix (M, N)
    const int M, const int K, const int N) {

    // Thread and block identification
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int tid = ty * BLOCK_COLS + tx; // 0..255

    // Block's starting position in the output matrix C
    const int block_row = blockIdx.y * TILE_DIM;
    const int block_col = blockIdx.x * TILE_DIM;

    // Shared memory for tiles of A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    // Accumulators for the output sub-matrix computed by this thread
    float acc[WPT_ROWS][WPT_COLS] = {{0.0f}};

    const int num_k_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Loop over tiles in the K dimension
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // --- Vectorized Global Memory Load into Shared Memory ---
        // Each of the 256 threads loads one float4 (4 floats) into shared memory.
        // 256 threads * 4 floats/thread = 1024 floats = 32x32 tile.

        // Load tile from A (M, K)
        const int gmem_A_start_col = k_tile * TILE_DIM;
        const int tile_load_row = tid / (TILE_DIM / 4);
        const int tile_load_col = (tid % (TILE_DIM / 4)) * 4;

        const int gmem_A_row = block_row + tile_load_row;
        const int gmem_A_col = gmem_A_start_col + tile_load_col;

        if (gmem_A_row < M && gmem_A_col < K) {
            *((float4*)&sA[tile_load_row][tile_load_col]) = *((const float4*)&A[gmem_A_row * K + gmem_A_col]);
        } else {
            *((float4*)&sA[tile_load_row][tile_load_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // Load tile from B (K, N)
        const int gmem_B_start_row = k_tile * TILE_DIM;
        const int gmem_B_row = gmem_B_start_row + tile_load_row;
        const int gmem_B_col = block_col + tile_load_col;

        if (gmem_B_row < K && gmem_B_col < N) {
            *((float4*)&sB[tile_load_row][tile_load_col]) = *((const float4*)&B[gmem_B_row * N + gmem_B_col]);
        } else {
            *((float4*)&sB[tile_load_row][tile_load_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        // --- Tile Computation ---
        // Each thread computes its 2x2 output using values from shared memory.
        for (int k = 0; k < TILE_DIM; ++k) {
            float regA[WPT_ROWS];
            for (int i = 0; i < WPT_ROWS; ++i) {
                regA[i] = sA[ty * WPT_ROWS + i][k];
            }

            float regB[WPT_COLS];
            for (int j = 0; j < WPT_COLS; ++j) {
                regB[j] = sB[k][tx * WPT_COLS + j];
            }
            
            for (int i = 0; i < WPT_ROWS; ++i) {
                for (int j = 0; j < WPT_COLS; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // --- Store Results to Global Memory with Bias and ReLU ---
    for (int i = 0; i < WPT_ROWS; ++i) {
        for (int j = 0; j < WPT_COLS; ++j) {
            int gmem_C_row = block_row + ty * WPT_ROWS + i;
            int gmem_C_col = block_col + tx * WPT_COLS + j;

            if (gmem_C_row < M && gmem_C_col < N) {
                float final_val = acc[i][j] + bias[gmem_C_col];
                C[gmem_C_row * N + gmem_C_col] = fmaxf(0.0f, final_val);
            }
        }
    }
}

torch::Tensor tiled_gemm_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(1) == weight.size(1), "Input and weight inner dimensions must match");
    TORCH_CHECK(weight.size(0) == bias.size(0), "Weight and bias dimensions must match");
    
    // For float4 vectorized loads, memory must be 16-byte aligned.
    TORCH_CHECK(((uintptr_t)input.data_ptr()) % 16 == 0, "Input tensor is not aligned to 16 bytes");

    const int M = input.size(0); // batch_size
    const int K = input.size(1); // in_features
    const int N = weight.size(0); // out_features

    // The kernel expects the weight matrix to be transposed for coalesced access.
    // PyTorch's nn.Linear weight is (out_features, in_features).
    // The GEMM is C(M,N) = A(M,K) * B(K,N).
    // A = input(M,K). B should be weight.T(K,N).
    // So we transpose weight from (N,K) to (K,N).
    auto weight_t = weight.transpose(0, 1).contiguous();
    TORCH_CHECK(((uintptr_t)weight_t.data_ptr()) % 16 == 0, "Transposed weight tensor is not aligned to 16 bytes");

    auto output = torch::zeros({M, N}, input.options());

    const dim3 block(BLOCK_COLS, BLOCK_ROWS);
    const dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    tiled_gemm_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight_t.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, K, N
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

cpp_source = "torch::Tensor tiled_gemm_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# JIT compile the CUDA kernel
tiled_gemm_relu_module = load_inline(
    name='tiled_gemm_relu_v2', # Changed name to avoid cache conflicts
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['tiled_gemm_relu_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        self.layers = nn.ModuleList()
        current_input_size = input_size
        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size

        self.final_linear = nn.Linear(current_input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        The hidden layers use the custom fused Linear+ReLU CUDA kernel.
        The final layer uses the standard PyTorch Linear layer.
        """
        for layer in self.layers:
            # The input tensor for the custom kernel must be contiguous.
            x_contiguous = x.contiguous()
            x = tiled_gemm_relu_module.tiled_gemm_relu_cuda(
                x_contiguous, layer.weight, layer.bias
            )
        
        x = self.final_linear(x)
        return x