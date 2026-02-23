# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100-SXM4-40GB, which has compute capability 8.0
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernel.
# This version replaces the custom warp-shuffle reduction with the highly optimized
# cub::BlockReduce primitive. The hypothesis is that for this specific workload
# and block size (256), the general-purpose, architecture-aware CUB library
# may outperform the hand-written reduction used in the SOTA. All other aspects
# of the high-performing semi-coarse-grained (4 channels/block) kernel,
# such as vectorized memory access and the main computation loop, are preserved.
fused_relu_maxpool_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// --- KERNEL: Fused ReLU + MaxPool3d + Reduction (using CUB::BlockReduce) ---
// This kernel processes 4 channels per block to reduce atomic contention.
// It leverages the CUB library for a highly optimized block-wide reduction.
__global__ void fused_relu_maxpool_reduction_kernel(
    const half* __restrict__ input,
    const half* __restrict__ bias,
    float* __restrict__ output,
    const int total_global_channels, // batch_size * out_channels
    const int out_channels,
    const int D_in, const int H_in, const int W_in,
    const float inv_num_pooled,
    const float divisor,
    const int channels_per_block) {

    // Define constants for the CUB reduction
    const int BLOCK_THREADS = 256;
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;

    // Allocate shared memory for the CUB reduction
    extern __shared__ typename BlockReduce::TempStorage temp_storage[];

    const int tid = threadIdx.x;
    const int group_start_idx = blockIdx.x * channels_per_block;

    // Calculate output (post-pool) dimensions once
    const int D_out = D_in / 2;
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;
    const int num_pooled_elements = D_out * H_out * W_out;

    // Define a zero vector for the vectorized ReLU operation.
    const __half zero_h_val = __float2half(0.0f);
    const half2 zero_v = __halves2half2(zero_h_val, zero_h_val);

    // This block processes `channels_per_block` channels sequentially.
    #pragma unroll
    for (int k = 0; k < channels_per_block; ++k) {
        const int global_idx = group_start_idx + k;
        if (global_idx >= total_global_channels) {
            continue; // Boundary check for the last block
        }
        
        // Decompose global channel index into batch and channel indices
        const int b = global_idx / out_channels;
        const int c = global_idx % out_channels;

        // Pointer to the start of the data for the current channel.
        const long long int channel_offset = (long long int)global_idx * D_in * H_in * W_in;
        const half* channel_ptr = input + channel_offset;

        // Each thread computes a partial sum for the current channel
        float my_sum = 0.0f;
        
        // Grid-stride loop over the *output* (pooled) spatial elements
        for (int i = tid; i < num_pooled_elements; i += blockDim.x) {
            // Decompose linear output index 'i' into 3D coords (d_out, h_out, w_out)
            const int d_out = i / (H_out * W_out);
            const int rem = i % (H_out * W_out);
            const int h_out = rem / W_out;
            const int w_out = rem % W_out;

            // Find top-left-front corner in the input tensor for the 2x2x2 pooling window
            const int d_in_start = d_out * 2;
            const int h_in_start = h_out * 2;
            const int w_in_start = w_out * 2;
            
            // Use half2 vectorized loads for the 8 values in the 2x2x2 cube.
            const long long int slice_stride = (long long int)H_in * W_in;
            const long long int row_stride = W_in;

            const half* ptr_d0_h0 = channel_ptr + (long long int)d_in_start * slice_stride + (long long int)h_in_start * row_stride + w_in_start;
            const half* ptr_d0_h1 = ptr_d0_h0 + row_stride;
            const half* ptr_d1_h0 = ptr_d0_h0 + slice_stride;
            const half* ptr_d1_h1 = ptr_d1_h0 + row_stride;
            
            const half2 v00 = *reinterpret_cast<const half2*>(ptr_d0_h0);
            const half2 v01 = *reinterpret_cast<const half2*>(ptr_d0_h1);
            const half2 v10 = *reinterpret_cast<const half2*>(ptr_d1_h0);
            const half2 v11 = *reinterpret_cast<const half2*>(ptr_d1_h1);
            
            // --- Fused Operation ---
            // 1. Apply ReLU to all 8 values using the vectorized __hmax2 intrinsic.
            const half2 v00_relu = __hmax2(v00, zero_v);
            const half2 v01_relu = __hmax2(v01, zero_v);
            const half2 v10_relu = __hmax2(v10, zero_v);
            const half2 v11_relu = __hmax2(v11, zero_v);

            // 2. Perform MaxPool3d using a vectorized reduction tree on the post-ReLU values.
            const half2 max_half_v0 = __hmax2(v00_relu, v01_relu);
            const half2 max_half_v1 = __hmax2(v10_relu, v11_relu);
            const half2 final_max_v = __hmax2(max_half_v0, max_half_v1);
            const half max_val_h = __hmax(final_max_v.x, final_max_v.y);
            
            // 3. Accumulate sum for the final reduction
            my_sum += __half2float(max_val_h);
        }

        // Perform the efficient block-wide reduction using CUB.
        __syncthreads();
        const float channel_sum = BlockReduce(temp_storage).Sum(my_sum);
        __syncthreads();

        // Thread 0 computes the final value and adds to the output atomically.
        if (tid == 0) {
            const float channel_avg = channel_sum * inv_num_pooled;
            const float processed_val = channel_avg / divisor + __half2float(bias[c]);
            atomicAdd(&output[b], processed_val);
        }
    }
}


// --- C++ Wrapper Function ---
// This function is called from Python and launches the CUDA kernel.
torch::Tensor fused_relu_maxpool_reduction_cuda(torch::Tensor input, torch::Tensor bias, float divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Input must be a Half tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "Bias must be a Half tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    const int batch_size = input.size(0);
    const int out_channels = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    auto output_options = torch::dtype(torch::kFloat).device(input.device());
    auto output = torch::zeros({batch_size}, output_options);
    
    const int total_global_channels = batch_size * out_channels;
    if (total_global_channels == 0) {
        return output.view({batch_size, 1, 1, 1});
    }

    // Pre-compute the reciprocal for the averaging division on the host
    const int D_out = D_in / 2;
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;
    const int num_pooled_elements = D_out * H_out * W_out;
    const float inv_num_pooled = (num_pooled_elements > 0) ? (1.0f / (float)num_pooled_elements) : 0.0f;

    // Kernel launch configuration.
    const int block_size = 256;
    const int channels_per_block = 4;
    const int num_blocks = (total_global_channels + channels_per_block - 1) / channels_per_block;
    
    // Calculate shared memory size needed for CUB::BlockReduce
    // This is determined at compile time so we can't use a variable.
    // We hardcode the block size here to match the kernel.
    void* temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, input.data_ptr<at::Half>(), output.data_ptr<float>(), 1);
    
    cudaError_t err;

    fused_relu_maxpool_reduction_kernel<<<num_blocks, block_size, temp_storage_bytes>>>(
        (const half*)input.data_ptr<at::Half>(),
        (const half*)bias.data_ptr<at::Half>(),
        output.data_ptr<float>(),
        total_global_channels,
        out_channels,
        D_in, H_in, W_in,
        inv_num_pooled,
        divisor,
        channels_per_block
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output.view({batch_size, 1, 1, 1});
}
"""

fused_relu_maxpool_reduction_cpp_source = "torch::Tensor fused_relu_maxpool_reduction_cuda(torch::Tensor input, torch::Tensor bias, float divisor);"

# JIT compile the CUDA code
fused_relu_maxpool_reduction = load_inline(
    name='fused_relu_maxpool_reduction_cub_v1', # Give a new name to avoid cache conflicts
    cpp_sources=fused_relu_maxpool_reduction_cpp_source,
    cuda_sources=fused_relu_maxpool_reduction_source,
    functions=['fused_relu_maxpool_reduction_cuda'],
    verbose=False,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized Model that fuses ReLU, MaxPool3d, and subsequent reduction
    operations into a single, monolithic CUDA kernel. This version replaces the
    SOTA's custom warp-shuffle reduction with the CUB library's BlockReduce
    primitive. It retains the highly effective semi-coarse-grained launch
    configuration (4 channels/block) to minimize global atomic contention,
    while testing the hypothesis that the general-purpose CUB reduction may
    offer superior performance due to its extensive, architecture-specific
    optimizations.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, divisor: float, pool_size: int, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        # The Conv3d layer is applied before the custom kernel.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Parameters for the custom kernel
        self.divisor = divisor
        # The bias needs to be reshaped from (C, 1, 1, 1) to (C) for the kernel
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze())
        
        # Convert model parameters to half precision for performance
        self.conv.half()
        self.bias.data = self.bias.data.half()
        
        # Custom fused kernel function
        self.fused_op = fused_relu_maxpool_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to half precision
        x_half = x.half()
        
        # Apply the convolution layer
        x_half = self.conv(x_half)
        
        # Ensure input to kernel is contiguous as required by the C++ wrapper
        x_half = x_half.contiguous()

        # Call the single fused kernel for ReLU, MaxPool3d, and all subsequent operations.
        output_float = self.fused_op.fused_relu_maxpool_reduction_cuda(x_half, self.bias, self.divisor)
        
        return output_float
# RegexTagCustomPruningAlgorithmEnd