# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernel for Max Pooling 2D with vectorization
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// Helper device function to find the max value in a float4 vector
__device__ inline float max_float4(float4 v) {
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

__global__ void max_pool2d_cuda_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    // Calculate the global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_height * output_width;

    if (index >= total_elements) {
        return;
    }

    // Decompose index to N, C, H_out, W_out
    int w_out = index % output_width;
    int h_out = (index / output_width) % output_height;
    int c = (index / (output_width * output_height)) % channels;
    int n = index / (channels * output_height * output_width);

    // Calculate the starting coordinates of the pooling window in the input tensor
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -FLT_MAX;
    const float* input_channel = input + (n * channels + c) * input_height * input_width;

    // Optimized path for the most common case: dilation = 1
    if (dilation == 1) {
        #pragma unroll
        for (int i = 0; i < kernel_size; ++i) {
            int h_in = h_start + i;
            // Check if the row is within the input height bounds
            if (h_in >= 0 && h_in < input_height) {
                const float* input_row = input_channel + h_in * input_width;
                int w_in = w_start;
                int w_end = w_start + kernel_size;

                // Process initial unaligned or out-of-bounds elements scalar-wise
                for (; w_in < w_end && (w_in < 0 || (w_in % 4 != 0)); ++w_in) {
                    if (w_in >= 0 && w_in < input_width) {
                        max_val = fmaxf(max_val, input_row[w_in]);
                    }
                }

                // Main vectorized loop for aligned data within bounds
                for (; w_in + 3 < w_end && w_in + 3 < input_width; w_in += 4) {
                    float4 val4 = *reinterpret_cast<const float4*>(input_row + w_in);
                    max_val = fmaxf(max_val, max_float4(val4));
                }

                // Process remaining elements scalar-wise
                for (; w_in < w_end; ++w_in) {
                    if (w_in >= 0 && w_in < input_width) {
                        max_val = fmaxf(max_val, input_row[w_in]);
                    }
                }
            }
        }
    } else { // Fallback to the baseline scalar implementation for dilation > 1
        #pragma unroll
        for (int i = 0; i < kernel_size; ++i) {
            #pragma unroll
            for (int j = 0; j < kernel_size; ++j) {
                int h_in = h_start + i * dilation;
                int w_in = w_start + j * dilation;
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    max_val = fmaxf(max_val, input_channel[h_in * input_width + w_in]);
                }
            }
        }
    }

    output[index] = max_val;
}

torch::Tensor max_pool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation) {
     // Ensure input is a contiguous float tensor on CUDA
     x = x.contiguous();

     // Get input dimensions
     int batch_size = x.size(0);
     int channels = x.size(1);
     int input_height = x.size(2);
     int input_width = x.size(3);

     // Compute output dimensions
     int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
     int output_width  = (input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

     // Create output tensor
     auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
     auto output = torch::empty({batch_size, channels, output_height, output_width}, options);

     // Handle case with no output elements
     if (output.numel() == 0) {
         return output;
     }

     // Configure and launch CUDA kernel
     int total_threads = output.numel();
     const int threads_per_block = 256;
     int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

     max_pool2d_cuda_kernel_vectorized<<<blocks, threads_per_block>>> (
         x.data_ptr<float>(),
         output.data_ptr<float>(),
         batch_size,
         channels,
         input_height,
         input_width,
         output_height,
         output_width,
         kernel_size,
         stride,
         padding,
         dilation
     );

     // Check for kernel launch errors
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         throw std::runtime_error(cudaGetErrorString(err));
     }
     
     return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor max_pool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code for Max Pooling 2D
max_pool2d_cuda_lib = load_inline(
    name='max_pool2d_cuda_lib',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['max_pool2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.max_pool2d_cuda_op = max_pool2d_cuda_lib
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        return self.max_pool2d_cuda_op.max_pool2d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)
# RegexTagCustomPruningAlgorithmEnd