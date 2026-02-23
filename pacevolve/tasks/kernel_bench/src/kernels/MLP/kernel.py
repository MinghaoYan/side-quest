# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100-SXM4-40GB, which has compute capability 8.0
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# CUDA and C++ source code for the optimized kernel
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile width for shared memory
#define TILE_WIDTH 32
// Work-per-thread: each thread computes a 2x2 block of output
#define WPT_Y 2
#define WPT_X 2
// Padding to avoid shared memory bank conflicts during transpose
#define PADDING 1

// CUDA kernel for Fused Linear + ReLU using tiled matrix multiplication.
// This version uses:
// - A 2x2 work-per-thread mapping for better register reuse.
// - Software pipelining (double buffering) to hide memory latency.
// - float4 vectorized loads for BOTH input matrix A and weight matrix B to maximize bandwidth.
// - An in-shared-memory transpose for the B tile with padding to avoid bank conflicts.
__global__ void linear_relu_pipelined_2x2_vec_AB_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    // A: M x K (input)
    // B: N x K (weight) -> accessed as B^T
    // C: M x N (output)

    // Shared memory for tiles of A and B, double buffered for pipelining.
    __shared__ float As[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[2][TILE_WIDTH][TILE_WIDTH + PADDING];
    __shared__ float B_temp[TILE_WIDTH][TILE_WIDTH];

    // Thread indices
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int linear_tid = ty * blockDim.x + tx;

    // Global row and column index for the top-left corner of the 2x2 output block this thread computes
    const int row_out_start = blockIdx.y * TILE_WIDTH + ty * WPT_Y;
    const int col_out_start = blockIdx.x * TILE_WIDTH + tx * WPT_X;

    // Accumulators for the 2x2 output elements
    float acc[WPT_Y][WPT_X] = {{0.0f}};

    const int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // --- Prefetch first tile (k_tile_idx=0) into buffer 0 ---
    int k_tile_idx = 0;
    int k_base_prefetch = k_tile_idx * TILE_WIDTH;

    // --- Prefetch A tile ---
    // Vectorized load for A tile. Each thread loads one float4.
    const int a_tile_row = linear_tid / (TILE_WIDTH / 4);
    const int a_tile_col_vec = linear_tid % (TILE_WIDTH / 4);

    const int gmem_a_row = blockIdx.y * TILE_WIDTH + a_tile_row;
    const int gmem_a_col = k_base_prefetch + a_tile_col_vec * 4;

    if (gmem_a_row < M && (gmem_a_col + 3) < K) {
        *reinterpret_cast<float4*>(&As[0][a_tile_row][a_tile_col_vec*4]) = *reinterpret_cast<const float4*>(&A[gmem_a_row * K + gmem_a_col]);
    } else { // Handle boundary
        #pragma unroll
        for (int i=0; i<4; ++i) {
            if (gmem_a_row < M && (gmem_a_col + i) < K) {
                As[0][a_tile_row][a_tile_col_vec*4 + i] = A[gmem_a_row * K + gmem_a_col + i];
            } else {
                As[0][a_tile_row][a_tile_col_vec*4 + i] = 0.0f;
            }
        }
    }

    // --- Prefetch B tile (vectorized load + shared memory transpose) ---
    // 1. Vectorized load into a temporary shared buffer
    const int b_tile_row = linear_tid / (TILE_WIDTH / 4);
    const int b_tile_col_vec = linear_tid % (TILE_WIDTH / 4);
    const int gmem_b_row = blockIdx.x * TILE_WIDTH + b_tile_row;
    const int gmem_b_col = k_base_prefetch + b_tile_col_vec * 4;
    
    if (gmem_b_row < N && (gmem_b_col + 3) < K) {
        *reinterpret_cast<float4*>(&B_temp[b_tile_row][b_tile_col_vec * 4]) = *reinterpret_cast<const float4*>(&B[gmem_b_row * K + gmem_b_col]);
    } else { // Handle boundary
        #pragma unroll
        for (int i=0; i<4; ++i) {
            if (gmem_b_row < N && (gmem_b_col + i) < K) {
                B_temp[b_tile_row][b_tile_col_vec*4 + i] = B[gmem_b_row * K + gmem_b_col + i];
            } else {
                B_temp[b_tile_row][b_tile_col_vec*4 + i] = 0.0f;
            }
        }
    }
    __syncthreads();

    // 2. Transpose from B_temp into Bs[0] using the 16x16 thread block
    Bs[0][tx][ty] = B_temp[ty][tx];
    Bs[0][tx][ty + 16] = B_temp[ty + 16][tx];
    Bs[0][tx + 16][ty] = B_temp[ty][tx + 16];
    Bs[0][tx + 16][ty + 16] = B_temp[ty + 16][tx + 16];
    __syncthreads();

    int compute_buf_idx = 0;

    // --- Main loop: Compute tile p, while prefetching tile p+1 ---
    for (k_tile_idx = 1; k_tile_idx < num_tiles; ++k_tile_idx) {
        int prefetch_buf_idx = 1 - compute_buf_idx;
        k_base_prefetch = k_tile_idx * TILE_WIDTH;

        // Prefetch next A tile
        const int gmem_a_row_prefetch = blockIdx.y * TILE_WIDTH + a_tile_row;
        const int gmem_a_col_prefetch = k_base_prefetch + a_tile_col_vec * 4;
        if (gmem_a_row_prefetch < M && (gmem_a_col_prefetch + 3) < K) {
            *reinterpret_cast<float4*>(&As[prefetch_buf_idx][a_tile_row][a_tile_col_vec*4]) = *reinterpret_cast<const float4*>(&A[gmem_a_row_prefetch * K + gmem_a_col_prefetch]);
        } else {
            #pragma unroll
            for (int i=0; i<4; ++i) {
                if (gmem_a_row_prefetch < M && (gmem_a_col_prefetch + i) < K) {
                    As[prefetch_buf_idx][a_tile_row][a_tile_col_vec*4 + i] = A[gmem_a_row_prefetch * K + gmem_a_col_prefetch + i];
                } else {
                    As[prefetch_buf_idx][a_tile_row][a_tile_col_vec*4 + i] = 0.0f;
                }
            }
        }
        
        // Prefetch next B tile
        const int gmem_b_row_prefetch = blockIdx.x * TILE_WIDTH + b_tile_row;
        const int gmem_b_col_prefetch = k_base_prefetch + b_tile_col_vec * 4;
        if (gmem_b_row_prefetch < N && (gmem_b_col_prefetch + 3) < K) {
            *reinterpret_cast<float4*>(&B_temp[b_tile_row][b_tile_col_vec * 4]) = *reinterpret_cast<const float4*>(&B[gmem_b_row_prefetch * K + gmem_b_col_prefetch]);
        } else {
            #pragma unroll
            for (int i=0; i<4; ++i) {
                if (gmem_b_row_prefetch < N && (gmem_b_col_prefetch + i) < K) {
                    B_temp[b_tile_row][b_tile_col_vec*4 + i] = B[gmem_b_row_prefetch * K + gmem_b_col_prefetch + i];
                } else {
                    B_temp[b_tile_row][b_tile_col_vec*4 + i] = 0.0f;
                }
            }
        }
        
        // Compute with tile k_tile_idx-1
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            float a_vals[WPT_Y];
            #pragma unroll
            for(int wy=0; wy < WPT_Y; ++wy) {
                a_vals[wy] = As[compute_buf_idx][ty * WPT_Y + wy][k];
            }
            #pragma unroll
            for(int wx=0; wx < WPT_X; ++wx) {
                float b_val = Bs[compute_buf_idx][k][tx * WPT_X + wx];
                #pragma unroll
                for(int wy=0; wy < WPT_Y; ++wy) {
                    acc[wy][wx] += a_vals[wy] * b_val;
                }
            }
        }
        __syncthreads(); // Wait for compute to finish before starting next prefetch

        // Transpose the prefetched B tile
        Bs[prefetch_buf_idx][tx][ty] = B_temp[ty][tx];
        Bs[prefetch_buf_idx][tx][ty + 16] = B_temp[ty + 16][tx];
        Bs[prefetch_buf_idx][tx + 16][ty] = B_temp[ty][tx + 16];
        Bs[prefetch_buf_idx][tx + 16][ty + 16] = B_temp[ty + 16][tx + 16];
        __syncthreads(); // Wait for all prefetching to complete
        compute_buf_idx = 1 - compute_buf_idx;
    }

    // --- Compute with the last tile ---
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
        float a_vals[WPT_Y];
        #pragma unroll
        for(int wy=0; wy < WPT_Y; ++wy) {
            a_vals[wy] = As[compute_buf_idx][ty * WPT_Y + wy][k];
        }
        #pragma unroll
        for(int wx=0; wx < WPT_X; ++wx) {
            float b_val = Bs[compute_buf_idx][k][tx * WPT_X + wx];
            #pragma unroll
            for(int wy=0; wy < WPT_Y; ++wy) {
                acc[wy][wx] += a_vals[wy] * b_val;
            }
        }
    }

    // --- Write results to C, add bias, and apply ReLU ---
    #pragma unroll
    for (int wy = 0; wy < WPT_Y; ++wy) {
        int row = row_out_start + wy;
        if (row < M) {
            #pragma unroll
            for (int wx = 0; wx < WPT_X; ++wx) {
                int col = col_out_start + wx;
                if (col < N) {
                    float val = acc[wy][wx] + bias[col];
                    C[row * N + col] = fmaxf(0.0f, val);
                }
            }
        }
    }
}

// C++ wrapper function to be called from Python
torch::Tensor linear_relu_pipelined_2x2_vec_AB_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);

    TORCH_CHECK(K == weight.size(1), "Input and weight dimensions are incompatible for matmul");
    TORCH_CHECK(N == bias.size(0), "Weight and bias dimensions are incompatible");

    auto output = torch::zeros({M, N}, input.options());

    // Block dimensions are (TILE_WIDTH/WPT_X, TILE_WIDTH/WPT_Y) -> (16, 16)
    dim3 threadsPerBlock(TILE_WIDTH / WPT_X, TILE_WIDTH / WPT_Y);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    linear_relu_pipelined_2x2_vec_AB_kernel<<<numBlocks, threadsPerBlock>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

cpp_source = "torch::Tensor linear_relu_pipelined_2x2_vec_AB_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Load the CUDA kernel inline using PyTorch's C++ extension utility
# This compiles the C++/CUDA code at runtime
try:
    linear_relu_2x2_op = load_inline(
        name='linear_relu_pipelined_2x2_vec_AB',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['linear_relu_pipelined_2x2_vec_AB_cuda'],
        verbose=True,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
    )
except Exception as e:
    print(f"Failed to load CUDA kernel: {e}")
    linear_relu_2x2_op = None


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        if linear_relu_2x2_op is None:
            raise RuntimeError("CUDA kernel failed to compile. Cannot initialize ModelNew.")

        # Store the custom CUDA function
        self.custom_op = linear_relu_2x2_op.linear_relu_pipelined_2x2_vec_AB_cuda
        
        # Dynamically create layers based on the provided sizes
        self.hidden_layers = nn.ModuleList()
        current_input_size = input_size
        for layer_size in layer_sizes:
            self.hidden_layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size

        # The final layer is kept as a standard PyTorch layer
        self.final_layer = nn.Linear(current_input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        Uses the custom fused Linear+ReLU CUDA kernel for the hidden layers.
        """
        # Ensure input is on the correct device
        if self.hidden_layers:
             x = x.to(self.hidden_layers[0].weight.device)
        else:
             x = x.to(self.final_layer.weight.device)
        
        # Apply custom fused kernel to hidden layers
        for layer in self.hidden_layers:
            # Ensure tensors are contiguous before passing to CUDA kernel
            x = x.contiguous()
            x = self.custom_op(x, layer.weight, layer.bias)
        
        # Final output layer without custom activation
        x = self.final_layer(x)
        
        return x
# RegexTagCustomPruningAlgorithmEnd