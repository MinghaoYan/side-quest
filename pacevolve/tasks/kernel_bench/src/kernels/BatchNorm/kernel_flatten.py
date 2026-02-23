import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm> // For std::min

__global__ void batch_norm_fused_vectorized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    const long long total_elements_vec, // This is total_elements / 4
    const int C,
    const int HW,
    const float epsilon
) {
    // This kernel combines a flattened 1D grid-stride loop with float4 vectorization.
    // Each thread processes a float4 vector, effectively handling 4 elements at a time.
    // This maximizes parallelism like a standard grid-stride loop while also increasing
    // memory throughput by issuing wider memory transactions.
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long grid_stride = (long long)gridDim.x * blockDim.x;

    for (long long idx4 = i; idx4 < total_elements_vec; idx4 += grid_stride) {
        // Calculate the channel index 'c'. All 4 elements in a float4 load
        // belong to the same channel as they are contiguous along the width dimension.
        // We use the index of the first element (idx4 * 4) to derive 'c'.
        const long long base_idx = idx4 * 4;
        const int c = (base_idx / HW) % C;

        // Fetch channel-specific parameters once per float4. These reads are cached
        // efficiently by the L1/L2 hierarchy.
        const float mean = running_mean[c];
        const float var = running_var[c];
        const float g = gamma[c];
        const float b = beta[c];
        
        // Pre-calculate the scale and bias for this channel.
        const float inv_stddev = rsqrtf(var + epsilon);
        const float scale = g * inv_stddev;
        const float bias = b - mean * scale;
        
        // Perform vectorized load.
        const float4 in_vec = ((const float4*)input)[idx4];
        
        // Apply the batch normalization transformation to all 4 elements.
        float4 out_vec;
        out_vec.x = in_vec.x * scale + bias;
        out_vec.y = in_vec.y * scale + bias;
        out_vec.z = in_vec.z * scale + bias;
        out_vec.w = in_vec.w * scale + bias;
        
        // Perform vectorized store.
        ((float4*)output)[idx4] = out_vec;
    }
}


// C++ wrapper function to launch the CUDA kernel
std::vector<torch::Tensor> batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
) {
    // Ensure tensors are on the GPU
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "Running mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "Running var must be a CUDA tensor");

    // Ensure input is contiguous for predictable memory layout
    auto input_contiguous = input.contiguous();
    auto output = torch::empty_like(input_contiguous);
    
    const int N = input_contiguous.size(0);
    const int C = input_contiguous.size(1); 
    const int H = input_contiguous.size(2);
    const int W = input_contiguous.size(3);

    // Vectorized kernel requires the last dimension to be a multiple of 4
    TORCH_CHECK(W % 4 == 0, "For vectorized kernel, tensor width must be a multiple of 4");
    
    const long long total_elements = input_contiguous.numel();
    if (total_elements == 0) {
        return {output};
    }
    const long long total_elements_vec = total_elements / 4;
    const int HW = H * W;
    
    // Kernel launch configuration
    const int threads_per_block = 256;
    
    // Calculate number of blocks for the vectorized problem size
    const int target_blocks = (total_elements_vec + threads_per_block - 1) / threads_per_block;
    
    // Cap the number of blocks to saturate the GPU without excessive launch overhead
    const int num_blocks = std::min(target_blocks, 4096);
    
    batch_norm_fused_vectorized_kernel<<<num_blocks, threads_per_block>>>(
        input_contiguous.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements_vec,
        C,
        HW,
        epsilon
    );
    
    // Check for any CUDA errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return {output};
}
"""

cpp_source = """
#include <vector>
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

# Use load_inline to JIT compile the C++/CUDA code
custom_batch_norm_lib = load_inline(
    name='custom_batch_norm_lib_vec',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batch_norm_cuda'],
    verbose=True,
    extra_cuda_cflags=["-arch=sm_80"] # For A100 GPU
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        # These parameters will be moved to the GPU by the runner
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        # Assign the JIT-compiled function
        self.batch_norm_cuda_op = custom_batch_norm_lib.batch_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The calling convention expects a list of tensors, so we take the first element.
        # A check for the width dimension is inside the C++ wrapper.
        return self.batch_norm_cuda_op(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps
        )[0]