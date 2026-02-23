# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void batch_norm_kernel_float4(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float epsilon
) {
    // Each thread processes one float4 element.
    // The grid is 1D and sized for the number of float4 elements.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_float4 = (N * C * H * W) / 4;

    if (idx < total_float4) {
        // Since we iterate along the W dimension, all 4 floats in a float4 belong to the same channel.
        // We calculate the linear index of the first float in the vector.
        int linear_idx_float = idx * 4;
        
        // From the linear index in NCHW format, we derive the channel index 'c'.
        // linear_idx = n*C*H*W + c*H*W + h*W + w
        // c = (linear_idx / (H*W)) % C
        int c_idx = (linear_idx_float / (H * W)) % C;

        // Fetch channel-specific parameters once per float4 vector.
        const float mean_val = running_mean[c_idx];
        const float var_val = running_var[c_idx];
        const float gamma_val = gamma[c_idx];
        const float beta_val = beta[c_idx];
        
        // Compute the normalization factor. Using rsqrtf is generally faster than 1/sqrtf.
        const float inv_std = rsqrtf(var_val + epsilon);

        // Load 4 floats at once using float4.
        float4 in_vec = ((const float4*)input)[idx];
        
        // Apply batch norm to each component of the vector.
        float4 out_vec;
        out_vec.x = gamma_val * (in_vec.x - mean_val) * inv_std + beta_val;
        out_vec.y = gamma_val * (in_vec.y - mean_val) * inv_std + beta_val;
        out_vec.z = gamma_val * (in_vec.z - mean_val) * inv_std + beta_val;
        out_vec.w = gamma_val * (in_vec.w - mean_val) * inv_std + beta_val;

        // Store 4 floats at once.
        ((float4*)output)[idx] = out_vec;
    }
}

// The C++ wrapper function that launches the CUDA kernel.
std::vector<torch::Tensor> batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
) {
    // Ensure input tensors are on the correct device and contiguous.
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (NCHW)");
    TORCH_CHECK(input.size(3) % 4 == 0, "Input tensor's width must be a multiple of 4 for vectorized kernel");

    auto output = torch::empty_like(input);
    
    const auto sizes = input.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int H = sizes[2];
    const int W = sizes[3];
    
    const int total_elements = N * C * H * W;
    const int total_float4 = total_elements / 4;

    // Kernel launch configuration.
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_float4 + threads_per_block - 1) / threads_per_block;
    
    batch_norm_kernel_float4<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        epsilon
    );
    
    // Check for any CUDA errors after kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return {output};
}
"""

batch_norm_cpp_source = """
#include <torch/extension.h>
std::vector<torch::Tensor> batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta, 
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
);
"""

batch_norm_cuda = load_inline(
    name='batch_norm_vectorized',
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=['batch_norm_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.batch_norm = batch_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The custom kernel expects contiguous NCHW tensors.
        x_cont = x.contiguous()
        return self.batch_norm.batch_norm_cuda(
            x_cont,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps
        )[0]
# RegexTagCustomPruningAlgorithmEnd