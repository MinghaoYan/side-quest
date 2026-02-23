# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile sizes for shared memory - Increased to 32 based on Idea 1
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                             int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column for this thread's output element
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Accumulator for the output element
    float sum = 0.0f;
    
    // Loop over tiles in the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load a tile of A into shared memory
        // Each thread loads one element of As
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
            
        // Load a tile of B into shared memory
        // Each thread loads one element of Bs
        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
            
        __syncthreads();
        
        // Multiply the tiles from shared memory and accumulate
        // This loop is unrolled by the compiler.
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
            
        __syncthreads();
    }
    
    // Write the final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "Input tensor A must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "Input tensor B must be on CUDA");
    TORCH_CHECK(a.dim() == 2, "Input tensor A must be 2D");
    TORCH_CHECK(b.dim() == 2, "Input tensor B must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions mismatch");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    auto c = torch::zeros({M, N}, a.options());
    
    const dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    const dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                         (M + TILE_SIZE - 1) / TILE_SIZE);
                   
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K);
        
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
        
    return c;
}
"""

matmul_cpp_source = """
#include <torch/extension.h>
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

# JIT compilation of the CUDA kernel
matmul_cuda_module = load_inline(
    name='matmul_cuda_v2',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=['matmul_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # The prompt signature of __init__(self, num_features: int) is inconsistent
        # with the provided state-of-the-art code, which takes no arguments.
        # Following the state-of-the-art example's structure.
        self.matmul_cuda = matmul_cuda_module.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # The prompt signature of forward(self, x: torch.Tensor) is inconsistent
        # with the matmul operation and state-of-the-art code.
        # Following the state-of-the-art example's forward signature.
        return self.matmul_cuda(A.cuda(), B.cuda())

# The following functions are for testing and are not part of the required output.
# They are included here to ensure the model is runnable and to match the
# provided SOTA template.

M = 256
N = 256 
K = 131072

def get_inputs():
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []

# RegexTagCustomPruningAlgorithmEnd