import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// __device__ function for a fast GELU approximation using an explicit FMA instruction.
// This approach leverages both the highly optimized `tanhf` intrinsic (via --use_fast_math)
// and the FMA instruction to maximize arithmetic throughput. FMA combines a multiply and
// an add into a single instruction.
// Original: 0.5f * x * (1.0f + tanh_val) is equivalent to 0.5f * x + 0.5f * x * tanh_val
__device__ __forceinline__ float fast_gelu(float x) {
    const float k_sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2.0/M_PI)
    const float k_coeff = 0.044715f;
    const float x_cubed = x * x * x;
    const float inner_arg = k_sqrt_2_over_pi * fmaf(k_coeff, x_cubed, x);
    const float tanh_val = tanhf(inner_arg);
    return fmaf(0.5f * x, tanh_val, 0.5f * x);
}

// CUDA kernel optimized with a 16-element workload to maximize instruction-level
// parallelism and hide memory latency. It uses a split-loop strategy and
// hoisted pointer casting for maximum efficiency.
__global__ void gelu_forward_kernel_fma_vec16(const float* __restrict__ input, float* __restrict__ output, int size) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;

    // Hoist the reinterpret_cast outside the loop to avoid redundant address calculations.
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    // --- Main Vectorized Loop (Branch-Free) ---
    // This loop processes 16 elements per thread per iteration.
    const int vectorized_size_16 = size / 16;
    for (int chunk_idx = thread_id; chunk_idx < vectorized_size_16; chunk_idx += grid_stride) {
        // Each chunk corresponds to 16 elements (4 float4s)
        const int base_idx = chunk_idx * 4;

        // Load four float4 vectors using __ldg for cached reads
        const float4 in1 = __ldg(&input4[base_idx]);
        const float4 in2 = __ldg(&input4[base_idx + 1]);
        const float4 in3 = __ldg(&input4[base_idx + 2]);
        const float4 in4 = __ldg(&input4[base_idx + 3]);

        float4 out1, out2, out3, out4;

        // Unconditional computation for all 16 elements using the FMA-based GELU.
        out1.x = fast_gelu(in1.x); out1.y = fast_gelu(in1.y); out1.z = fast_gelu(in1.z); out1.w = fast_gelu(in1.w);
        out2.x = fast_gelu(in2.x); out2.y = fast_gelu(in2.y); out2.z = fast_gelu(in2.z); out2.w = fast_gelu(in2.w);
        out3.x = fast_gelu(in3.x); out3.y = fast_gelu(in3.y); out3.z = fast_gelu(in3.z); out3.w = fast_gelu(in3.w);
        out4.x = fast_gelu(in4.x); out4.y = fast_gelu(in4.y); out4.z = fast_gelu(in4.z); out4.w = fast_gelu(in4.w);

        // Store four float4 vectors
        output4[base_idx] = out1;
        output4[base_idx + 1] = out2;
        output4[base_idx + 2] = out3;
        output4[base_idx + 3] = out4;
    }

    // --- Scalar Remainder Loop ---
    // This loop efficiently handles the final 1-15 elements.
    const int remainder_start = vectorized_size_16 * 16;
    for (int i = remainder_start + thread_id; i < size; i += grid_stride) {
        output[i] = fast_gelu(input[i]);
    }
}

// C++ wrapper function to launch the CUDA kernel from PyTorch.
torch::Tensor gelu_forward_cu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    auto output = torch::empty_like(input);
    int size = input.numel();

    if (size == 0) {
        return output;
    }

    const int block_size = 256;
    // Calculate grid size for a 16-element workload. The grid-stride loop ensures scalability.
    const int num_chunks = (size + 15) / 16;
    const int num_blocks = (num_chunks + block_size - 1) / block_size;

    gelu_forward_kernel_fma_vec16<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}
'''

cpp_source = "torch::Tensor gelu_forward_cu(torch::Tensor input);"

gelu_custom_module = load_inline(
    name='gelu_fma_vec16_crossover',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gelu_forward_cu'],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False
)

class ModelNew(nn.Module):
    '''
    A model that performs GELU activation using a custom CUDA kernel optimized
    with a 16-element workload per thread to maximize instruction-level parallelism,
    combined with FMA arithmetic for peak throughput.
    '''
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies GELU activation using the optimized custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor. Must be contiguous and float32.

        Returns:
            torch.Tensor: Output tensor with GELU applied.
        '''
        return gelu_custom_module.gelu_forward_cu(x.contiguous())