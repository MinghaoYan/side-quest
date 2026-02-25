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

__global__ void batch_norm_kernel_float4(
    const float4* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float4* __restrict__ output,
    int channels,
    int spatial_size_4,
    int size4,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        int c = (idx / spatial_size_4) % channels;

        // Fetch running statistics and parameters
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float shift = beta[c];

        // Precompute scale and shift for this thread to reduce math instructions
        float inv_std = 1.0f / sqrt(var + epsilon);
        float w = scale * inv_std;
        float b = shift - mean * w;

        // Vectorized read, compute, and write
        float4 in4 = input[idx];
        float4 out4;
        out4.x = in4.x * w + b;
        out4.y = in4.y * w + b;
        out4.z = in4.z * w + b;
        out4.w = in4.w * w + b;

        output[idx] = out4;
    }
}

__global__ void batch_norm_kernel_float1(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int channels,
    int spatial_size,
    int size,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / spatial_size) % channels;

        // Fetch running statistics and parameters
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float shift = beta[c];

        // Precompute scale and shift
        float inv_std = 1.0f / sqrt(var + epsilon);
        float w = scale * inv_std;
        float b = shift - mean * w;

        output[idx] = input[idx] * w + b;
    }
}

std::vector<torch::Tensor> batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
) {
    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int channels = input.size(1); 
    int height = input.size(2);
    int width = input.size(3);
    
    int spatial_size = height * width;
    int size = batch_size * channels * spatial_size;
    const int threads = 256;
    
    if (size == 0) {
        return {output};
    }

    // Check if we can use vectorized float4 memory accesses for higher bandwidth
    if (spatial_size % 4 == 0 && 
        ((unsigned long long)input.data_ptr<float>()) % 16 == 0 && 
        ((unsigned long long)output.data_ptr<float>()) % 16 == 0) {
        
        int size4 = size / 4;
        int spatial_size_4 = spatial_size / 4;
        const int blocks = (size4 + threads - 1) / threads;
        batch_norm_kernel_float4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            channels,
            spatial_size_4,
            size4,
            epsilon
        );
    } else {
        const int blocks = (size + threads - 1) / threads;
        batch_norm_kernel_float1<<<blocks, threads>>>(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            output.data_ptr<float>(),
            channels,
            spatial_size,
            size,
            epsilon
        );
    }
    
    return {output};
}
"""

batch_norm_cpp_source = """
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
    name='batch_norm_ext',
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
        return self.batch_norm.batch_norm_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.bias.cuda(),
            self.running_mean.cuda(),
            self.running_var.cuda(),
            self.eps
        )[0]
# RegexTagCustomPruningAlgorithmEnd