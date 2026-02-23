# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100-SXM4-40GB. This is compute capability 8.0.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Custom CUDA kernel for fused Linear + ReLU using vectorized memory access
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_linear_relu_forward_vectorized(
    const float* __restrict__ input,        // [batch_size, in_features]
    const float* __restrict__ weight,       // [out_features, in_features]
    const float* __restrict__ bias,         // [out_features]
    float* __restrict__ output,             // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features) {

    // Each thread computes one output element
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // Output feature index

    if (row < batch_size && col < out_features) {
        float acc = bias[col];
        
        const float* input_row = input + row * in_features;
        const float* weight_row = weight + col * in_features;

        const int vec_size = in_features / 4;

        // Vectorized part for features divisible by 4
        for (int j = 0; j < vec_size; ++j) {
            const float4 input_vec = ((const float4*)input_row)[j];
            const float4 weight_vec = ((const float4*)weight_row)[j];

            acc += input_vec.x * weight_vec.x;
            acc += input_vec.y * weight_vec.y;
            acc += input_vec.z * weight_vec.z;
            acc += input_vec.w * weight_vec.w;
        }
        
        // Handle remainder elements if in_features is not a multiple of 4
        for (int j = vec_size * 4; j < in_features; ++j) {
            acc += input_row[j] * weight_row[j];
        }

        // Apply ReLU activation
        output[row * out_features + col] = fmaxf(acc, 0.0f);
    }
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    
    // Ensure input is contiguous for vectorized access
    input = input.contiguous();

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, out_features}, options);

    // Standard block size, good for many GPUs
    const dim3 blockSize(16, 16);
    const dim3 gridSize((out_features + blockSize.x - 1) / blockSize.x,
                         (batch_size + blockSize.y - 1) / blockSize.y);

    fused_linear_relu_forward_vectorized<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features);
        
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

fused_linear_relu_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
# Using a unique name to avoid conflicts if running multiple experiments
fused_linear_relu_vec = load_inline(
    name='fused_linear_relu_vec',
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=['fused_linear_relu_cuda'],
    verbose=True,
)

class FusedLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # We need the input to be contiguous for our kernel
        output = fused_linear_relu_vec.fused_linear_relu_cuda(input.contiguous(), weight, bias)
        # Backward is not required by the problem, so it's not implemented.
        # For a real scenario, we would save tensors needed for the backward pass.
        # ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is not implemented as per instructions.
        raise NotImplementedError("Backward pass not implemented for custom fused Linear-ReLU.")

class FusedLinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(FusedLinearReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return FusedLinearReLUFunction.apply(input, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        # The convolutional part of the model remains unchanged, as per AlexNet architecture.
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # The classifier part where we apply our custom fused layers.
        # The input features to the first fully connected layer are 256 * 6 * 6 = 9216
        # Both 9216 and 4096 are divisible by 4, making them suitable for float4 vectorization.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0), # Dropout is kept at 0.0 as in the baseline example
            FusedLinearReLU(in_features=256 * 6 * 6, out_features=4096),
            nn.Dropout(p=0.0),
            FusedLinearReLU(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# RegexTagCustomPruningAlgorithmEnd