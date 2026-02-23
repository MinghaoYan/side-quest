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

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 0):
        super(ModelNew, self).__init__()

        matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// --- Kernel Configuration ---
// Dimensions of the C tile computed by a thread block
#define TILE_M 64
#define TILE_N 64
// Dimension of the tile loaded along the K dimension
#define TILE_K 16
// Thread block dimensions
#define BLOCK_ROWS 16
#define BLOCK_COLS 16
// Padding for shared memory to avoid bank conflicts
#define PAD_N 4
#define TILE_N_PAD (TILE_N + PAD_N)


// Work per thread (size of C sub-matrix) - Automatically updated
#define WPT_M (TILE_M / BLOCK_ROWS)
#define WPT_N (TILE_N / BLOCK_COLS)

// Split-K factor: Number of partitions along the K dimension.
// Increased from 128 to 256 to test effect of higher grid-level parallelism.
#define SPLIT_K_FACTOR 256

__global__ void matmul_kernel_splitk_256(const float* A, const float* B, float* C,
                                         const int M, const int N, const int K) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_M][TILE_K];
    // Use padded shared memory for B to avoid bank conflicts
    __shared__ float Bs[TILE_K][TILE_N_PAD];

    // Thread and block identification
    const int tx = threadIdx.x; // Thread's x-index within the block (0-15)
    const int ty = threadIdx.y; // Thread's y-index within the block (0-15)
    const int bx = blockIdx.x;  // Block's x-index in the grid
    const int by = blockIdx.y;  // Block's y-index in the grid
    const int bz = blockIdx.z;  // Block's z-index for Split-K

    // Linear thread ID for loading data
    const int tid = ty * BLOCK_COLS + tx;

    // Pointers to the start of the C tile this block will compute
    const int c_row_start = by * TILE_M;
    const int c_col_start = bx * TILE_N;

    // Register array to hold the WPT_M x WPT_N C sub-matrix
    float Csub[WPT_M][WPT_N] = {{0.0f}};
    
    // --- K-dimension split calculation ---
    const int k_per_split = (K + gridDim.z - 1) / gridDim.z;
    const int k_start = bz * k_per_split;
    const int k_end = min(k_start + k_per_split, K);

    // --- Main loop over the assigned slice of the K dimension ---
    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += TILE_K) {
        // --- Load tile from Global Memory to Shared Memory using vectorized float4 loads ---
        
        // Load A's tile (TILE_M x TILE_K = 64x16)
        // 256 threads load 256 float4s total (1024 floats)
        const int a_s_row = tid / (TILE_K / 4); // Each thread's target row in As
        const int a_s_col_4 = tid % (TILE_K / 4); // Each thread's target float4 column index
        const int a_g_row = c_row_start + a_s_row;
        const int a_g_col = k_tile_start + a_s_col_4 * 4;

        if (a_g_row < M && a_g_col + 3 < K) {
            *(reinterpret_cast<float4*>(&As[a_s_row][a_s_col_4 * 4])) = *(reinterpret_cast<const float4*>(&A[a_g_row * K + a_g_col]));
        } else { // Boundary case
            for (int i = 0; i < 4; ++i) {
                if (a_g_row < M && (a_g_col + i) < K) {
                    As[a_s_row][a_s_col_4 * 4 + i] = A[a_g_row * K + a_g_col + i];
                } else {
                    As[a_s_row][a_s_col_4 * 4 + i] = 0.0f;
                }
            }
        }

        // Load B's tile (TILE_K x TILE_N = 16x64)
        // 256 threads load 256 float4s total (1024 floats)
        const int b_s_row = tid / (TILE_N / 4); // Each thread's target row in Bs
        const int b_s_col_4 = tid % (TILE_N / 4); // Each thread's target float4 column index
        const int b_g_row = k_tile_start + b_s_row;
        const int b_g_col = c_col_start + b_s_col_4 * 4;

        if (b_g_row < K && b_g_col + 3 < N) {
            *(reinterpret_cast<float4*>(&Bs[b_s_row][b_s_col_4 * 4])) = *(reinterpret_cast<const float4*>(&B[b_g_row * N + b_g_col]));
        } else { // Boundary case
             for (int i = 0; i < 4; ++i) {
                if (b_g_row < K && (b_g_col + i) < N) {
                    Bs[b_s_row][b_s_col_4 * 4 + i] = B[b_g_row * N + b_g_col + i];
                } else {
                    Bs[b_s_row][b_s_col_4 * 4 + i] = 0.0f;
                }
            }
        }

        __syncthreads();

        // --- Compute matrix multiplication from Shared Memory to Registers ---
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float A_reg[WPT_M];
            #pragma unroll
            for(int m = 0; m < WPT_M; ++m) A_reg[m] = As[ty * WPT_M + m][k];
            
            float B_reg[WPT_N];
            #pragma unroll
            for(int n = 0; n < WPT_N; ++n) B_reg[n] = Bs[k][tx * WPT_N + n];

            #pragma unroll
            for (int m = 0; m < WPT_M; ++m) {
                #pragma unroll
                for (int n = 0; n < WPT_N; ++n) {
                    Csub[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }
        __syncthreads();
    }

    // --- Write partial results from Registers to Global Memory using Atomic Add ---
    #pragma unroll
    for (int m = 0; m < WPT_M; ++m) {
        #pragma unroll
        for (int n = 0; n < WPT_N; ++n) {
            int global_row_C = c_row_start + ty * WPT_M + m;
            int global_col_C = c_col_start + tx * WPT_N + n;
            if (global_row_C < M && global_col_C < N) {
                atomicAdd(&C[global_row_C * N + global_col_C], Csub[m][n]);
            }
        }
    }
}


torch::Tensor matmul_cuda_wrapper(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input tensor B must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 2, "Input tensor A must be 2-dimensional");
    TORCH_CHECK(b.dim() == 2, "Input tensor B must be 2-dimensional");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions are not compatible for multiplication");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    auto c = torch::zeros({M, N}, a.options());
    
    dim3 threadsPerBlock(BLOCK_COLS, BLOCK_ROWS);
    dim3 numBlocks((N + TILE_N - 1) / TILE_N,
                   (M + TILE_M - 1) / TILE_M,
                   SPLIT_K_FACTOR);
                   
    matmul_kernel_splitk_256<<<numBlocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return c;
}
"""

        matmul_cpp_source = """
torch::Tensor matmul_cuda_wrapper(torch::Tensor a, torch::Tensor b);
"""

        # JIT compilation of the CUDA kernel
        self.matmul_cuda = load_inline(
            name='matmul_cuda_splitk256',
            cpp_sources=matmul_cpp_source,
            cuda_sources=matmul_source,
            functions=['matmul_cuda_wrapper'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # The forward function signature in the problem description is `forward(self, x: torch.Tensor)`,
        # but the logic requires two tensors. The provided state-of-the-art implementation uses two tensors
        # (A, B). Adhering to the logic of the baseline for a valid implementation.
        A_cuda = A.contiguous().cuda()
        B_cuda = B.contiguous().cuda()
        
        return self.matmul_cuda.matmul_cuda_wrapper(A_cuda, B_cuda)