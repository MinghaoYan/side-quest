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

# Set CUDA architecture for A100-SXM4-40GB, which has compute capability 8.0
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernels for the forward and backward passes.
# The key change is modifying the block tile shape from 64x128 to 32x256.
cuda_source_fp16 = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <mma.h> // For WMMA intrinsics

// --- Tiling and WMMA Configuration ---
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Each thread block computes a 32x256 tile of the output matrix.
#define BLOCK_TILE_M 32   // Changed from 64
#define BLOCK_TILE_N 256  // Changed from 128
#define BLOCK_TILE_K 16

// Each warp computes a 16x16 tile. A 32x256 block tile needs 32 warps (2x16 grid).
#define WARPS_IN_M_DIM (BLOCK_TILE_M / WMMA_M) // 2
#define WARPS_IN_N_DIM (BLOCK_TILE_N / WMMA_N) // 16
#define WARPS_PER_BLOCK (WARPS_IN_M_DIM * WARPS_IN_N_DIM) // 32
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32) // 1024

using namespace nvcuda;

// --- Fused Forward Pass Kernel using WMMA ---
__global__ void fused_matmul_bias_relu_wmma_kernel(
    const __half* __restrict__ A, // input (M, K)
    const __half* __restrict__ B, // weight (N, K), must be used as B.T
    const __half* __restrict__ bias,
    __half* __restrict__ C, // output (M, N)
    int M, int N, int K)
{
    // Shared memory for tiles of A, B, accumulator, and bias
    __shared__ __half sh_A[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ __half sh_B[BLOCK_TILE_K][BLOCK_TILE_N];
    __shared__ float sh_acc[BLOCK_TILE_M][BLOCK_TILE_N];
    __shared__ __half sh_bias[BLOCK_TILE_N];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Map 1024 threads to 32 warps, and map warps to a 2x16 grid
    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid_in_block / 32;
    int warp_row_in_block = warp_id / WARPS_IN_N_DIM;
    int warp_col_in_block = warp_id % WARPS_IN_N_DIM;

    int warp_dst_row = warp_row_in_block * WMMA_M;
    int warp_dst_col = warp_col_in_block * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_TILE_K) {
        __syncthreads();
        
        // Scalar load for sh_A. 512 halfs / 1024 threads. Only first 512 threads load.
        int sh_a_idx = tid_in_block;
        if (sh_a_idx < BLOCK_TILE_M * BLOCK_TILE_K) {
            int a_row_in_tile = sh_a_idx / BLOCK_TILE_K;
            int a_col_in_tile = sh_a_idx % BLOCK_TILE_K;
            int g_a_row = block_row * BLOCK_TILE_M + a_row_in_tile;
            int g_a_col = k_tile_start + a_col_in_tile;

            if (g_a_row < M && g_a_col < K) {
                sh_A[a_row_in_tile][a_col_in_tile] = A[g_a_row * K + g_a_col];
            } else {
                sh_A[a_row_in_tile][a_col_in_tile] = __float2half(0.0f);
            }
        }
        
        // Coalesced, vectorized load for sh_B with on-the-fly transpose.
        // Load a 256x16 tile from B(N,K) and store as 16x256 in sh_B.
        // 4096 halfs / 1024 threads = 4 halfs/thread.
        int sh_b_idx = tid_in_block * 4;
        int src_row_in_tile = sh_b_idx / BLOCK_TILE_K; // row in 256x16 source tile (0..255)
        int src_col_in_tile_start = sh_b_idx % BLOCK_TILE_K; // col in 256x16 source tile (0..12)
        
        int g_src_row = block_col * BLOCK_TILE_N + src_row_in_tile;
        int g_src_col_start = k_tile_start + src_col_in_tile_start;

        if (g_src_row < N && g_src_col_start + 3 < K) {
            uint2 val = *reinterpret_cast<const uint2*>(&B[g_src_row * K + g_src_col_start]);
            const __half* vals = reinterpret_cast<const __half*>(&val);
            sh_B[src_col_in_tile_start + 0][src_row_in_tile] = vals[0];
            sh_B[src_col_in_tile_start + 1][src_row_in_tile] = vals[1];
            sh_B[src_col_in_tile_start + 2][src_row_in_tile] = vals[2];
            sh_B[src_col_in_tile_start + 3][src_row_in_tile] = vals[3];
        } else { // Boundary case
            for(int i = 0; i < 4; ++i) {
                int src_col_in_tile = src_col_in_tile_start + i;
                int g_src_col = k_tile_start + src_col_in_tile;
                if (g_src_row < N && g_src_col < K) {
                    sh_B[src_col_in_tile][src_row_in_tile] = B[g_src_row * K + g_src_col];
                } else {
                    sh_B[src_col_in_tile][src_row_in_tile] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        wmma::load_matrix_sync(a_frag, &sh_A[warp_dst_row][0], BLOCK_TILE_K);
        wmma::load_matrix_sync(b_frag, &sh_B[0][warp_dst_col], BLOCK_TILE_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(&sh_acc[warp_dst_row][warp_dst_col], acc_frag, BLOCK_TILE_N, wmma::mem_row_major);
    __syncthreads();

    // Load bias into shared memory
    if (tid_in_block < BLOCK_TILE_N) { // only first 256 threads
        int bias_col = block_col * BLOCK_TILE_N + tid_in_block;
        if (bias_col < N) { sh_bias[tid_in_block] = bias[bias_col]; } 
        else { sh_bias[tid_in_block] = __float2half(0.0f); }
    }
    __syncthreads();

    // Epilogue: bias + relu + store. Each thread handles 8192 / 1024 = 8 elements.
    for (int i = 0; i < 8; ++i) { 
        int idx = tid_in_block + i * THREADS_PER_BLOCK;
        int row = idx / BLOCK_TILE_N; 
        int col = idx % BLOCK_TILE_N; 
        int g_row = block_row * BLOCK_TILE_M + row;
        int g_col = block_col * BLOCK_TILE_N + col;
        if (g_row < M && g_col < N) {
            float val = sh_acc[row][col] + __half2float(sh_bias[col]);
            C[g_row * N + g_col] = __float2half(fmaxf(val, 0.0f));
        }
    }
}


// --- Backward Pass Kernels using WMMA ---

// grad_input = grad_output @ weight
__global__ void fused_drelu_grad_input_wmma_kernel(
    const __half* __restrict__ grad_output, // (M, K)
    const __half* __restrict__ output,      // (M, K)
    const __half* __restrict__ weight,      // (K, N)
    __half* __restrict__ grad_input,        // (M, N)
    int M, int N, int K)
{
    __shared__ __half sh_A[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ __half sh_B[BLOCK_TILE_K][BLOCK_TILE_N];
    __shared__ float sh_acc[BLOCK_TILE_M][BLOCK_TILE_N];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid_in_block / 32;
    int warp_row_in_block = warp_id / WARPS_IN_N_DIM;
    int warp_col_in_block = warp_id % WARPS_IN_N_DIM;
    
    int warp_dst_row = warp_row_in_block * WMMA_M;
    int warp_dst_col = warp_col_in_block * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_TILE_K) {
        __syncthreads();
        // Scalar load grad_output with dReLU fusion into sh_A. 512 halfs / 1024 threads.
        int sh_a_idx = tid_in_block;
        if (sh_a_idx < BLOCK_TILE_M * BLOCK_TILE_K) {
            int a_row_in_tile = sh_a_idx / BLOCK_TILE_K;
            int a_col_in_tile = sh_a_idx % BLOCK_TILE_K;
            int g_a_row = block_row * BLOCK_TILE_M + a_row_in_tile;
            int g_a_col = k_tile_start + a_col_in_tile;
            
            int g_idx = g_a_row * K + g_a_col;
            if (g_a_row < M && g_a_col < K) {
                sh_A[a_row_in_tile][a_col_in_tile] = __hle(output[g_idx], __float2half(0.0f)) ? __float2half(0.0f) : grad_output[g_idx];
            } else {
                sh_A[a_row_in_tile][a_col_in_tile] = __float2half(0.0f);
            }
        }
        
        // Coalesced, vectorized load for weight into sh_B. 4096 halfs / 1024 threads.
        int sh_b_idx = tid_in_block * 4;
        int row_in_tile = sh_b_idx / BLOCK_TILE_N;
        int col_in_tile_start = sh_b_idx % BLOCK_TILE_N;
        int g_row = k_tile_start + row_in_tile;
        int g_col_start = block_col * BLOCK_TILE_N + col_in_tile_start;

        if (g_row < K && g_col_start + 3 < N) {
             *reinterpret_cast<uint2*>(&sh_B[row_in_tile][col_in_tile_start]) = *reinterpret_cast<const uint2*>(&weight[g_row * N + g_col_start]);
        } else {
            for(int i = 0; i < 4; ++i) {
                int col_in_tile = col_in_tile_start + i;
                int g_col = block_col * BLOCK_TILE_N + col_in_tile;
                if (g_row < K && g_col < N) {
                    sh_B[row_in_tile][col_in_tile] = weight[g_row * N + g_col];
                } else {
                    sh_B[row_in_tile][col_in_tile] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        wmma::load_matrix_sync(a_frag, &sh_A[warp_dst_row][0], BLOCK_TILE_K);
        wmma::load_matrix_sync(b_frag, &sh_B[0][warp_dst_col], BLOCK_TILE_K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    wmma::store_matrix_sync(&sh_acc[warp_dst_row][warp_dst_col], acc_frag, BLOCK_TILE_N, wmma::mem_row_major);
    __syncthreads();

    for (int i = 0; i < 8; ++i) {
        int idx = tid_in_block + i * THREADS_PER_BLOCK;
        int row = idx / BLOCK_TILE_N;
        int col = idx % BLOCK_TILE_N;
        int g_row = block_row * BLOCK_TILE_M + row;
        int g_col = block_col * BLOCK_TILE_N + col;
        if (g_row < M && g_col < N) {
            grad_input[g_row * N + g_col] = __float2half(sh_acc[row][col]);
        }
    }
}

// grad_weight = grad_output.T @ input
__global__ void fused_drelu_grad_weight_wmma_kernel(
    const __half* __restrict__ grad_output, // Access as (K, M)
    const __half* __restrict__ output,      // Access as (K, M)
    const __half* __restrict__ input,       // (K, N)
    __half* __restrict__ grad_weight,       // (M, N)
    int M, int N, int K, int D_out)
{
    __shared__ __half sh_A[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ __half sh_B[BLOCK_TILE_K][BLOCK_TILE_N];
    __shared__ float sh_acc[BLOCK_TILE_M][BLOCK_TILE_N];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid_in_block / 32;
    int warp_row_in_block = warp_id / WARPS_IN_N_DIM;
    int warp_col_in_block = warp_id % WARPS_IN_N_DIM;

    int warp_dst_row = warp_row_in_block * WMMA_M;
    int warp_dst_col = warp_col_in_block * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

    for (int k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_TILE_K) {
        __syncthreads();
        // Scalar load grad_output.T with dReLU (transposed access, not coalesced)
        int sh_a_idx = tid_in_block;
        if (sh_a_idx < BLOCK_TILE_M * BLOCK_TILE_K) {
            int row = sh_a_idx / BLOCK_TILE_K;
            int col = sh_a_idx % BLOCK_TILE_K;
            int g_row_A = block_row * BLOCK_TILE_M + row;
            int g_col_A = k_tile_start + col;
            if (g_row_A < M && g_col_A < K) {
                int g_idx = g_col_A * D_out + g_row_A; 
                sh_A[row][col] = __hle(output[g_idx], __float2half(0.0f)) ? __float2half(0.0f) : grad_output[g_idx];
            } else {
                sh_A[row][col] = __float2half(0.0f);
            }
        }
        
        // Coalesced, vectorized load for input into sh_B
        int sh_b_idx = tid_in_block * 4;
        int row_in_tile = sh_b_idx / BLOCK_TILE_N;
        int col_in_tile_start = sh_b_idx % BLOCK_TILE_N;
        int g_row = k_tile_start + row_in_tile;
        int g_col_start = block_col * BLOCK_TILE_N + col_in_tile_start;

        if (g_row < K && g_col_start + 3 < N) {
            *reinterpret_cast<uint2*>(&sh_B[row_in_tile][col_in_tile_start]) = *reinterpret_cast<const uint2*>(&input[g_row * N + g_col_start]);
        } else {
            for(int i = 0; i < 4; ++i) {
                int col_in_tile = col_in_tile_start + i;
                int g_col = block_col * BLOCK_TILE_N + col_in_tile;
                if (g_row < K && g_col < N) {
                    sh_B[row_in_tile][col_in_tile] = input[g_row * N + g_col];
                } else {
                    sh_B[row_in_tile][col_in_tile] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        wmma::load_matrix_sync(a_frag, &sh_A[warp_dst_row][0], BLOCK_TILE_K);
        wmma::load_matrix_sync(b_frag, &sh_B[0][warp_dst_col], BLOCK_TILE_K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    wmma::store_matrix_sync(&sh_acc[warp_dst_row][warp_dst_col], acc_frag, BLOCK_TILE_N, wmma::mem_row_major);
    __syncthreads();
    
    for (int i = 0; i < 8; ++i) {
        int idx = tid_in_block + i * THREADS_PER_BLOCK;
        int row = idx / BLOCK_TILE_N;
        int col = idx % BLOCK_TILE_N;
        int g_row = block_row * BLOCK_TILE_M + row;
        int g_col = block_col * BLOCK_TILE_N + col;
        if (g_row < M && g_col < N) {
            grad_weight[g_row * N + g_col] = __float2half(sh_acc[row][col]);
        }
    }
}

// grad_bias kernel (Identical to SOTA)
__global__ void fused_drelu_grad_bias_kernel(
    const __half* __restrict__ grad_output,
    const __half* __restrict__ output,
    __half* __restrict__ grad_bias,
    int B, int D_out)
{
    int col_idx = blockIdx.x;
    if (col_idx >= D_out) return;

    extern __shared__ __half sdata[];
    int tid = threadIdx.x;
    sdata[tid] = __float2half(0.0f);
    
    for (int row_idx = tid; row_idx < B; row_idx += blockDim.x) {
        int idx = row_idx * D_out + col_idx;
        __half grad_val = __hle(output[idx], __float2half(0.0f)) ? __float2half(0.0f) : grad_output[idx];
        sdata[tid] = __hadd(sdata[tid], grad_val);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = __hadd(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[col_idx] = sdata[0];
    }
}


// --- C++ Wrappers ---

torch::Tensor fused_matmul_bias_relu_wmma(
    const torch::Tensor& input, 
    const torch::Tensor& weight, 
    const torch::Tensor& bias) 
{
    const auto M = input.size(0);
    const auto K = input.size(1);
    const auto N = weight.size(0);
    auto output = torch::empty({M, N}, input.options());
    dim3 threads(32, 32); // 1024 threads
    dim3 blocks((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    fused_matmul_bias_relu_wmma_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()), M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return output;
}

torch::Tensor fused_drelu_grad_input(const torch::Tensor grad_output, const torch::Tensor output, const torch::Tensor weight) {
    const auto B = grad_output.size(0);
    const auto D_out = grad_output.size(1);
    const auto D_in = weight.size(1);
    auto grad_input = torch::empty({B, D_in}, grad_output.options());

    const int M = B, K = D_out, N = D_in;
    dim3 threads(32, 32); // 1024 threads
    dim3 blocks((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    fused_drelu_grad_input_wmma_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()), M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return grad_input;
}

torch::Tensor fused_drelu_grad_weight(const torch::Tensor grad_output, const torch::Tensor output, const torch::Tensor input) {
    const auto B = grad_output.size(0);
    const auto D_out = grad_output.size(1);
    const auto D_in = input.size(1);
    auto grad_weight = torch::empty({D_out, D_in}, grad_output.options());

    const int M = D_out, K = B, N = D_in;
    dim3 threads(32, 32); // 1024 threads
    dim3 blocks((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    
    fused_drelu_grad_weight_wmma_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(grad_weight.data_ptr<at::Half>()), M, N, K, D_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return grad_weight;
}

torch::Tensor fused_drelu_grad_bias(const torch::Tensor grad_output, const torch::Tensor output) {
    const auto B = grad_output.size(0);
    const auto D_out = grad_output.size(1);
    auto grad_bias = torch::empty({D_out}, grad_output.options());
    const int threadsPerBlock = 256;
    const int numBlocks = D_out;
    const int shared_mem_size = threadsPerBlock * sizeof(__half);
    fused_drelu_grad_bias_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(
        reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(grad_bias.data_ptr<at::Half>()), B, D_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return grad_bias;
}

"""

cpp_source_fp16 = """
torch::Tensor fused_matmul_bias_relu_wmma(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias);
torch::Tensor fused_drelu_grad_input(const torch::Tensor grad_output, const torch::Tensor output, const torch::Tensor weight);
torch::Tensor fused_drelu_grad_weight(const torch::Tensor grad_output, const torch::Tensor output, const torch::Tensor input);
torch::Tensor fused_drelu_grad_bias(const torch::Tensor grad_output, const torch::Tensor output);
"""

# Use load_inline to JIT compile the CUDA/C++ code.
fp16_op_graphed = load_inline(
    name='fp16_op_wmma_32x256_tile',
    cpp_sources=[cpp_source_fp16],
    cuda_sources=[cuda_source_fp16],
    functions=['fused_matmul_bias_relu_wmma', 'fused_drelu_grad_input', 'fused_drelu_grad_weight', 'fused_drelu_grad_bias'],
    verbose=False,
    extra_cuda_cflags=['-std=c++17']
)

class GraphedLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, input, weight, bias):
        ctx.module = module
        ctx.save_for_backward(input, weight)
        
        # Replay forward graph
        module.static_fwd_input.copy_(input)
        module.fwd_graph.replay()
        # The output of the replay is in module.static_fwd_output
        return module.static_fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Replay backward graph
        module.static_bwd_input.copy_(input)
        # The forward output was saved in static_fwd_output and is an input to backward
        module.static_bwd_output.copy_(module.static_fwd_output) 
        module.static_bwd_grad_output.copy_(grad_output)

        module.bwd_graph.replay()

        if ctx.needs_input_grad[1]: # Corresponds to 'input'
            grad_input = module.static_grad_input
        if ctx.needs_input_grad[2]: # Corresponds to 'weight'
            grad_weight = module.static_grad_weight
        if ctx.needs_input_grad[3]: # Corresponds to 'bias'
            grad_bias = module.static_grad_bias
            
        return None, grad_input, grad_weight, grad_bias

class LinearReLUHalf(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearReLUHalf, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.half))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.half))
        self.reset_parameters()
        
        # CUDA graph objects
        self.fwd_graph = None
        self.bwd_graph = None

        # State for graph capture
        self.calls = 0
        self.warmup_iters = 3
        self.is_capturing = False # Prevents re-entrant graph capture

        # Placeholders for static tensors
        self.static_fwd_input = None
        self.static_fwd_output = None
        
        self.static_bwd_input = None
        self.static_bwd_output = None
        self.static_bwd_grad_output = None
        self.static_grad_input = None
        self.static_grad_weight = None
        self.static_grad_bias = None

    def reset_parameters(self):
        with torch.no_grad():
            weight_fp32 = torch.empty_like(self.weight, dtype=torch.float)
            nn.init.kaiming_uniform_(weight_fp32, a=math.sqrt(5))
            self.weight.copy_(weight_fp32)
            
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bias_fp32 = torch.empty_like(self.bias, dtype=torch.float)
            nn.init.uniform_(bias_fp32, -bound, bound)
            self.bias.copy_(bias_fp32)
    
    def _graph_capture(self, input):
        # --- Initialize static tensors based on first input ---
        self.static_fwd_input = torch.empty_like(input)
        
        # Forward pass capture
        self.fwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.fwd_graph):
            static_out = fp16_op_graphed.fused_matmul_bias_relu_wmma(
                self.static_fwd_input, self.weight, self.bias
            )
        self.static_fwd_output = static_out

        # --- Initialize backward static tensors ---
        self.static_bwd_grad_output = torch.empty_like(self.static_fwd_output)
        self.static_bwd_input = torch.empty_like(self.static_fwd_input)
        self.static_bwd_output = torch.empty_like(self.static_fwd_output)
        self.static_grad_input = torch.empty_like(self.static_fwd_input)
        self.static_grad_weight = torch.empty_like(self.weight)
        self.static_grad_bias = torch.empty_like(self.bias)

        # Backward pass capture
        self.bwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.bwd_graph):
            grad_input_2d = fp16_op_graphed.fused_drelu_grad_input(self.static_bwd_grad_output, self.static_bwd_output, self.weight)
            grad_weight = fp16_op_graphed.fused_drelu_grad_weight(self.static_bwd_grad_output, self.static_bwd_output, self.static_bwd_input)
            grad_bias = fp16_op_graphed.fused_drelu_grad_bias(self.static_bwd_grad_output, self.static_bwd_output)
            self.static_grad_input.copy_(grad_input_2d)
            self.static_grad_weight.copy_(grad_weight)
            self.static_grad_bias.copy_(grad_bias)

    def forward(self, input):
        orig_shape = input.shape
        if input.dim() > 2:
            input = input.reshape(-1, self.in_features)

        if self.calls < self.warmup_iters:
            # Run kernels directly during warmup
            output = fp16_op_graphed.fused_matmul_bias_relu_wmma(input, self.weight, self.bias)
            
            # On the last warmup iteration, capture the graph
            if self.calls == self.warmup_iters - 1 and not self.is_capturing:
                self.is_capturing = True
                self._graph_capture(input)
                self.is_capturing = False
            
            self.calls += 1
        else:
            # After warmup, run the graphed version
            output = GraphedLinearReLUFunction.apply(self, input, self.weight, self.bias)

        if len(orig_shape) > 2:
            return output.view(*orig_shape[:-1], self.out_features)
        return output

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 1000):
        super(ModelNew, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            LinearReLUHalf(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            LinearReLUHalf(4096, 4096),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_features)
        )
        
        self.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x_classifier = self.classifier(x)
        
        return x_classifier.float()