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

# Set CUDA architecture for A100-SXM4-40GB
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// --- Kernel Configuration ---
// Using a symmetric 16x16 thread block.
#define THREADS_X 16
#define THREADS_Y 16

// Each thread computes a 4x4 tile of the output matrix C.
#define VEC_C_Y 4
#define VEC_C_X 4

// Block dimensions are derived from thread config. This gives a 64x64 C-tile per block.
#define BLOCK_ROWS (THREADS_Y * VEC_C_Y) // 16 * 4 = 64
#define BLOCK_COLS (THREADS_X * VEC_C_X) // 16 * 4 = 64

// Shared memory tile size for the K dimension.
#define K_STEP 64

// Number of splits along the K dimension. Increased to 128 to maximize parallelism.
#define SPLIT_K_FACTOR 128


__global__ void matmul_splitk_atomic_kernel(const float* A, const float* B, float* C,
                                                int M, int N, int K) {
    // Shared memory for tiles of A and B.
    __shared__ float As[BLOCK_ROWS][K_STEP]; // 64 x 64
    __shared__ float Bs[K_STEP][BLOCK_COLS]; // 64 x 64

    // Thread indices
    const int tx = threadIdx.x; // 0-15
    const int ty = threadIdx.y; // 0-15

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Base indices for the C matrix tile this thread will compute
    const int c_base_row = by * BLOCK_ROWS + ty * VEC_C_Y;
    const int c_base_col = bx * BLOCK_COLS + tx * VEC_C_X;

    // Accumulators for the 4x4 output tile
    float accum[VEC_C_Y][VEC_C_X] = {{0.0f}};

    // K dimension is split into SPLIT_K_FACTOR chunks.
    const int K_per_split = K / SPLIT_K_FACTOR;
    const int k_start_idx = bz * K_per_split;
    const int num_tiles_per_split = K_per_split / K_STEP;

    // Loop over tiles within the assigned K-split chunk
    for (int t = 0; t < num_tiles_per_split; ++t) {
        const int global_k_idx = k_start_idx + t * K_STEP;

        // --- Vectorized Load of A tile (64x64) into shared memory using float4 ---
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int smem_row = ty * 4 + i;
            int gmem_row = by * BLOCK_ROWS + smem_row;

            if (gmem_row < M) {
                reinterpret_cast<float4*>(As[smem_row])[tx] =
                    reinterpret_cast<const float4*>(&A[gmem_row * K + global_k_idx])[tx];
            } else {
                reinterpret_cast<float4*>(As[smem_row])[tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // --- Vectorized Load of B tile (64x64) into shared memory using float4 ---
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int smem_row = ty * 4 + i;
            int gmem_row = global_k_idx + smem_row;

            if (gmem_row < K) {
                reinterpret_cast<float4*>(Bs[smem_row])[tx] =
                    reinterpret_cast<const float4*>(&B[gmem_row * N + bx * BLOCK_COLS])[tx];
            } else {
                reinterpret_cast<float4*>(Bs[smem_row])[tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        __syncthreads();

        // --- Compute matrix multiplication for the tile ---
        #pragma unroll
        for (int k = 0; k < K_STEP; ++k) {
            float a_reg[VEC_C_Y];
            float b_reg[VEC_C_X];

            #pragma unroll
            for(int i = 0; i < VEC_C_Y; ++i) {
                a_reg[i] = As[ty * VEC_C_Y + i][k];
            }

            #pragma unroll
            for(int j = 0; j < VEC_C_X; ++j) {
                b_reg[j] = Bs[k][tx * VEC_C_X + j];
            }

            #pragma unroll
            for(int i = 0; i < VEC_C_Y; ++i) {
                #pragma unroll
                for(int j = 0; j < VEC_C_X; ++j) {
                    accum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // Atomically add partial results to the final output tensor C
    #pragma unroll
    for(int i = 0; i < VEC_C_Y; ++i) {
        #pragma unroll
        for(int j = 0; j < VEC_C_X; ++j) {
            if (c_base_row + i < M && c_base_col + j < N) {
                atomicAdd(&C[(c_base_row + i) * N + (c_base_col + j)], accum[i][j]);
            }
        }
    }
}


torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input tensor B must be a CUDA tensor");
    a = a.contiguous();
    b = b.contiguous();

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    TORCH_CHECK(K % (SPLIT_K_FACTOR * K_STEP) == 0, "K must be divisible by SPLIT_K_FACTOR * K_STEP (128 * 64 = 8192)");
    TORCH_CHECK(N % 4 == 0, "N must be divisible by 4 for vectorized loads");

    // Output tensor C must be initialized to zero for atomic additions.
    auto c = torch::zeros({M, N}, a.options());

    // --- Launch Single Matmul Kernel with Atomic Add ---
    dim3 threadsPerBlock(THREADS_X, THREADS_Y);
    dim3 numBlocks((N + BLOCK_COLS - 1) / BLOCK_COLS,
                   (M + BLOCK_ROWS - 1) / BLOCK_ROWS,
                   SPLIT_K_FACTOR);

    matmul_splitk_atomic_kernel<<<numBlocks, threadsPerBlock>>>(
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
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

# Use load_inline to JIT compile the CUDA kernels
matmul_cuda_module = load_inline(
    name='matmul_cuda_splitk128_atomic',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=['matmul_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 0): # num_features is unused but kept for signature consistency
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A @ B using a single-kernel, highly parallel
        Split-K approach with atomic additions. This design merges the two kernels of the
        previous state-of-the-art into one. It maintains the extreme K-dimension
        parallelism (SPLIT_K_FACTOR = 128) but eliminates the need for a large
        intermediate tensor and a second kernel launch by having all blocks
        atomically accumulate their partial results directly into the final output matrix.
        This tests whether the overhead of atomics is less than the overhead of the
        previous two-kernel synchronization and memory traffic.

        Args:
            A (torch::Tensor): The first input matrix of shape (M, K).
            B (torch::Tensor): The second input matrix of shape (K, N).

        Returns:
            torch::Tensor: The output matrix C of shape (M, N).
        """
        a_cuda = A if A.is_cuda else A.cuda()
        b_cuda = B if B.is_cuda else B.cuda()

        return self.matmul_cuda.matmul_cuda(a_cuda, b_cuda)