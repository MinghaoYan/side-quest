# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100-SXM4-40GB, which has compute capability 8.0
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# CUDA source code for a vectorized RMS Normalization on NCHW tensors
# Corrected version to avoid misaligned memory access errors.
rms_norm_nchw_vectorized_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for RMS Normalization on NCHW tensor format.
// This kernel operates directly on the NCHW layout, avoiding expensive permutations.
// It manually unrolls the channel loop by a factor of 4 to increase instruction-level parallelism,
// fixing the misaligned address error from the previous float4 cast approach.
// Assumes C is a multiple of 4.
__global__ void rms_norm_nchw_vectorized_kernel(const float* __restrict__ x, float* __restrict__ out,
                                                int N, int H, int W, int C, float eps) {
    // Each block processes one feature vector across the channel dimension
    // for a given (n, h, w) coordinate.
    const int nhw_idx = blockIdx.x;
    if (nhw_idx >= N * H * W) {
        return;
    }

    // Deconstruct the flattened (N, H, W) index to get individual indices.
    const int w = nhw_idx % W;
    const int h = (nhw_idx / W) % H;
    const int n = nhw_idx / (H * W);

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int HW = H * W;

    // Base pointers for the current (n, h, w) location. This points to channel 0.
    const float* x_ptr = x + n * C * HW + h * W + w;
    float* out_ptr = out + n * C * HW + h * W + w;

    // Step 1: Calculate sum of squares with unrolled loads.
    float sum_sq = 0.0f;
    const int C_vec = C / 4; // Number of 4-float groups.

    for (int i = tid; i < C_vec; i += block_size) {
        // Pointer to the first of four channels for this iteration.
        const float* current_x_ptr = x_ptr + (i * 4) * HW;
        
        // Manually load 4 float values. Their addresses are strided by HW,
        // so a single float4 load is not possible.
        float v0 = current_x_ptr[0];
        float v1 = current_x_ptr[HW];
        float v2 = current_x_ptr[2 * HW];
        float v3 = current_x_ptr[3 * HW];
        
        sum_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Step 2: Perform parallel reduction on the sum of squares in shared memory.
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Step 3: Thread 0 calculates the inverse RMS and broadcasts it via shared memory.
    if (tid == 0) {
        float mean_sq = sdata[0] / C;
        sdata[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float inv_rms = sdata[0];

    // Step 4: Apply the normalization with unrolled loads and stores.
    for (int i = tid; i < C_vec; i += block_size) {
        const float* current_x_ptr = x_ptr + (i * 4) * HW;
        float* current_out_ptr = out_ptr + (i * 4) * HW;

        // Read, normalize, and write back the four float values.
        current_out_ptr[0]        = current_x_ptr[0]        * inv_rms;
        current_out_ptr[HW]       = current_x_ptr[HW]       * inv_rms;
        current_out_ptr[2 * HW]   = current_x_ptr[2 * HW]   * inv_rms;
        current_out_ptr[3 * HW]   = current_x_ptr[3 * HW]   * inv_rms;
    }
}

// C++ wrapper function to launch the CUDA kernel from PyTorch.
torch::Tensor rms_norm_nchw_vectorized_cuda(torch::Tensor x, float eps) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    TORCH_CHECK(C % 4 == 0, "Number of channels must be a multiple of 4 for this kernel.");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous (default NCHW format)");

    const int total_vectors = N * H * W;
    auto out = torch::empty_like(x);

    const int threads_per_block = 256;
    const int blocks = total_vectors;
    const int shared_mem_size = threads_per_block * sizeof(float);

    rms_norm_nchw_vectorized_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, H, W, C,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

# C++ source for function declaration
rms_norm_nchw_vectorized_cpp_source = """
torch::Tensor rms_norm_nchw_vectorized_cuda(torch::Tensor x, float eps);
"""

# JIT compile the CUDA extension
rms_norm_nchw_vectorized_module = load_inline(
    name='rms_norm_nchw_vectorized_module',
    cpp_sources=rms_norm_nchw_vectorized_cpp_source,
    cuda_sources=rms_norm_nchw_vectorized_cuda_source,
    functions=['rms_norm_nchw_vectorized_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized RMS Normalization model that uses a custom CUDA kernel operating
    directly on NCHW data. The kernel leverages loop unrolling to process channels
    in groups of four, mitigating the performance impact of strided memory access
    and avoiding the overhead of transposing the tensor layout.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        if num_features % 4 != 0:
            raise ValueError("Number of features must be a multiple of 4 for this optimized kernel.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RMS Normalization.
        Args:
            x (torch.Tensor): Input tensor in NCHW format.
        Returns:
            torch.Tensor: Normalized output tensor in NCHW format.
        """
        return rms_norm_nchw_vectorized_module.rms_norm_nchw_vectorized_cuda(x, self.eps)

# RegexTagCustomPruningAlgorithmEnd