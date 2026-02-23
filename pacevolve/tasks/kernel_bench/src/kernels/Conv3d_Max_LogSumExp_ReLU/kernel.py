# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Custom CUDA kernel for fused logsumexp and ReLU using parallel reduction
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

// Device function for block-level reduction to find the maximum value.
__device__ void block_reduce_max(volatile float* data) {
    // Tree-based reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            data[threadIdx.x] = fmaxf(data[threadIdx.x], data[threadIdx.x + s]);
        }
        __syncthreads();
    }
}

// Device function for block-level reduction to sum values.
__device__ void block_reduce_sum(volatile float* data) {
    // Tree-based reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
        __syncthreads();
    }
}

__global__ void logsumexp_relu_kernel_parallel_reduce(const float* x, float* y, int N, int C, int D, int H, int W)
{
    // Dynamically allocated shared memory
    extern __shared__ float sdata[];

    // Each block computes one output element for a given (n, d, h, w)
    int index = blockIdx.x;
    int total_outputs = N * D * H * W;
    if (index >= total_outputs) return;

    // Deconstruct flattened index to 5D coordinates
    int n = index / (D * H * W);
    int rem = index % (D * H * W);
    int d = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;

    // Calculate base pointer to x(n, 0, d, h, w)
    size_t base_offset = (size_t)n * C * D * H * W + (size_t)d * H * W + (size_t)h * W + w;
    const float* x_ptr = x + base_offset;
    const size_t c_stride = (size_t)D * H * W;

    // --- Pass 1: Find max value in parallel ---
    float thread_max = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_max = fmaxf(thread_max, x_ptr[c * c_stride]);
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Reduce max values in shared memory
    block_reduce_max(sdata);
    __syncthreads();
    float m = sdata[0];

    // --- Pass 2: Compute sum in parallel ---
    float thread_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        thread_sum += expf(x_ptr[c * c_stride] - m);
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce sum values in shared memory
    block_reduce_sum(sdata);
    __syncthreads();
    float s = sdata[0];

    // --- Final computation and write-back (only by thread 0) ---
    if (threadIdx.x == 0) {
        float lse = m + logf(s);
        y[index] = fmaxf(lse, 0.0f);
    }
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(x.dim() == 5, "Input must be a 5D tensor");

    const int N = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);

    // Output tensor has the same spatial/depth dims, but C is reduced to 1.
    auto y = torch::empty({N, D, H, W}, x.options());

    const int total_outputs = N * D * H * W;
    if (total_outputs == 0) {
        return y.reshape({N, 1, D, H, W});
    }

    // Kernel launch configuration
    const int threads_per_block = 256;
    const int blocks = total_outputs;
    const size_t shared_mem_size = threads_per_block * sizeof(float);

    logsumexp_relu_kernel_parallel_reduce<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, D, H, W
    );
    
    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    // Reshape output to match original model's expectation {N, 1, D, H, W}
    return y.reshape({N, 1, D, H, W});
}
"""

cpp_source = """
torch::Tensor logsumexp_relu_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
logsumexp_relu_module = load_inline(
    name='logsumexp_relu_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['logsumexp_relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized Model with custom CUDA operator for fused logsumexp and ReLU.
    This version uses a parallel reduction algorithm with shared memory.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        # Call the custom CUDA kernel
        x = logsumexp_relu_module.logsumexp_relu_cuda(x)
        return x
# RegexTagCustomPruningAlgorithmEnd