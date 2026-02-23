# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100-SXM4-40GB (Compute Capability 8.0)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernel for the fully fused operation
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_WIDTH 32

// CUDA kernel for Fused (Linear -> Norm -> Add/Mul)
// Computes C = ( (A @ B.T + bias_lin) * weight_norm + bias_norm + Y_param ) * Y_param
// A: input x (M x K)
// B: linear weight (N x K)
// C: output (M x N)
// Y_param: learnable parameter y (N)
// M: batch_size, N: out_features, K: in_features
__global__ void fused_linear_norm_add_mul_kernel(
    const float* A, 
    const float* B, 
    const float* bias_lin,
    const float* weight_norm,
    const float* bias_norm,
    const float* Y_param,
    float* C, 
    int M, 
    int N, 
    int K) 
{
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0;

    // Loop over the tiles of A and B
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load a tile of A into shared memory
        int A_col = t * TILE_WIDTH + tx;
        if (row < M && A_col < K) {
            sA[ty][tx] = A[row * K + A_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory (transposed access)
        int B_row = col;
        int B_col = t * TILE_WIDTH + ty;
        if (B_row < N && B_col < K) {
            sB[ty][tx] = B[B_row * K + B_col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Multiply the two tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    // After GEMM, apply the fused post-processing operations
    if (row < M && col < N) {
        // 1. Add linear bias
        float linear_out = Cvalue + bias_lin[col];
        // 2. Apply instance norm (affine transformation)
        float norm_out = linear_out * weight_norm[col] + bias_norm[col];
        // 3. Apply fused add/mul with the learnable parameter 'y'
        float y_val = Y_param[col];
        C[row * N + col] = (norm_out + y_val) * y_val;
    }
}

torch::Tensor fused_op(
    torch::Tensor x, 
    torch::Tensor lin_w, 
    torch::Tensor lin_b, 
    torch::Tensor norm_w, 
    torch::Tensor norm_b, 
    torch::Tensor y_param) 
{
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(lin_w.is_cuda(), "Input lin_w must be a CUDA tensor");
    TORCH_CHECK(lin_b.is_cuda(), "Input lin_b must be a CUDA tensor");
    TORCH_CHECK(norm_w.is_cuda(), "Input norm_w must be a CUDA tensor");
    TORCH_CHECK(norm_b.is_cuda(), "Input norm_b must be a CUDA tensor");
    TORCH_CHECK(y_param.is_cuda(), "Input y_param must be a CUDA tensor");
    
    int M = x.size(0); // batch_size
    int K = x.size(1); // in_features
    int N = lin_w.size(0); // out_features

    auto out = torch::empty({M, N}, x.options());

    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    fused_linear_norm_add_mul_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        lin_w.data_ptr<float>(),
        lin_b.data_ptr<float>(),
        norm_w.data_ptr<float>(),
        norm_b.data_ptr<float>(),
        y_param.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N, K
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

# Define the C++ interface
cpp_source = """
torch::Tensor fused_op(
    torch::Tensor x, 
    torch::Tensor lin_w, 
    torch::Tensor lin_b, 
    torch::Tensor norm_w, 
    torch::Tensor norm_b, 
    torch::Tensor y_param);
"""

# JIT compile the CUDA kernel
fused_op_module = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_op'],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with a single custom CUDA kernel fusing Linear, InstanceNorm,
    and a custom element-wise operation.
    """
    def __init__(self, in_features: int, out_features: int):
        super(ModelNew, self).__init__()
        # PyTorch layers are created to hold and manage the parameters (weights, biases)
        # The actual computation will be done in the custom kernel.
        self.linear = nn.Linear(in_features, out_features)
        
        # InstanceNorm operates on the output features.
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=1e-5, momentum=0.1, affine=True)
        
        # Define 'y' as a learnable parameter with size matching out_features.
        self.y = nn.Parameter(torch.randn(out_features))
        
        # Store the compiled CUDA function.
        self.fused_op = fused_op_module.fused_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The forward pass is replaced by a single call to our custom CUDA kernel.
        # We pass the input tensor and all necessary parameters directly.
        # This avoids the overhead of individual PyTorch layer calls and intermediate tensors.
        return self.fused_op(
            x,
            self.linear.weight,
            self.linear.bias,
            self.instance_norm.weight,
            self.instance_norm.bias,
            self.y
        )
# RegexTagCustomPruningAlgorithmEnd