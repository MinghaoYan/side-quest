# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100 optimizations
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Vectorized FP32 kernel
__global__ void relu_kernel_float4(const float4* __restrict__ input, float4* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float4 in = input[idx];
        float4 out;
        out.x = in.x > 0.0f ? in.x : 0.0f;
        out.y = in.y > 0.0f ? in.y : 0.0f;
        out.z = in.z > 0.0f ? in.z : 0.0f;
        out.w = in.w > 0.0f ? in.w : 0.0f;
        output[idx] = out;
    }
}

// Fallback scalar FP32 kernel
__global__ void relu_kernel_float(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float in = input[idx];
        output[idx] = in > 0.0f ? in : 0.0f;
    }
}

// Mixed Precision Vectorized kernel (Half2)
// Processes FP16 (Half) inputs but utilizes FP32 mathematical conversions/accumulations in registers
__global__ void relu_kernel_half2(const half2* __restrict__ input, half2* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        half2 in = input[idx];
        
        // Convert input half2 to float2 for high precision evaluation 
        float2 val = __half22float2(in);
        
        // Computation logic in FP32
        val.x = val.x > 0.0f ? val.x : 0.0f;
        val.y = val.y > 0.0f ? val.y : 0.0f;
        
        // Output safely casted back to FP16
        output[idx] = __float22half2_rn(val);
    }
}

// Fallback scalar FP16 kernel
__global__ void relu_kernel_half(const half* __restrict__ input, half* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(input[idx]);
        val = val > 0.0f ? val : 0.0f;
        output[idx] = __float2half(val);
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    if (size == 0) return output;

    const int threads = 256;

    if (input.dtype() == torch::kFloat32) {
        // Fast-path: Apply 128-bit vectorization if alignments match
        if (size % 4 == 0 && 
            reinterpret_cast<std::uintptr_t>(input.data_ptr<float>()) % 16 == 0 && 
            reinterpret_cast<std::uintptr_t>(output.data_ptr<float>()) % 16 == 0) {
            
            int size4 = size / 4;
            const int blocks = (size4 + threads - 1) / threads;
            relu_kernel_float4<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(input.data_ptr<float>()),
                reinterpret_cast<float4*>(output.data_ptr<float>()),
                size4
            );
        } else {
            const int blocks = (size + threads - 1) / threads;
            relu_kernel_float<<<blocks, threads>>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                size
            );
        }
    } else if (input.dtype() == torch::kFloat16) {
        // Fast-path: Apply 32-bit (Half2) vectorization if alignments match
        if (size % 2 == 0 && 
            reinterpret_cast<std::uintptr_t>(input.data_ptr<at::Half>()) % 4 == 0 && 
            reinterpret_cast<std::uintptr_t>(output.data_ptr<at::Half>()) % 4 == 0) {
            
            int size2 = size / 2;
            const int blocks = (size2 + threads - 1) / threads;
            relu_kernel_half2<<<blocks, threads>>>(
                reinterpret_cast<const half2*>(input.data_ptr<at::Half>()),
                reinterpret_cast<half2*>(output.data_ptr<at::Half>()),
                size2
            );
        } else {
            const int blocks = (size + threads - 1) / threads;
            relu_kernel_half<<<blocks, threads>>>(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                size
            );
        }
    } else {
        // Safety Fallback
        return torch::relu(input);
    }
    
    // Ensure accurate synchronization
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in relu_cuda: %s\\n", cudaGetErrorString(err));
    }
    
    return output;
}
"""

relu_cpp_source = """
torch::Tensor relu_cuda(torch::Tensor input);
"""

# Compile kernel ensuring strict optimization routines (A100 capabilities)
relu_cuda_module = load_inline(
    name='relu_optimized',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=['relu_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '-lineinfo', '-use_fast_math']
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        # Store requested features per guidelines
        self.num_features = num_features
        self.relu = relu_cuda_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass guaranteed contiguous representation matching PyTorch native speeds
        return self.relu.relu_cuda(x.contiguous().cuda())
# RegexTagCustomPruningAlgorithmEnd