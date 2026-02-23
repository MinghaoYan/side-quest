import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    '''
    Simple model that performs RMS Normalization using a custom, highly-optimized
    fused CUDA kernel. This version integrates a grid-stride loop, NVIDIA's CUB
    library for reduction, and loop unrolling to maximize performance and scalability.
    '''
    def __init__(self, num_features: int, eps: float = 1e-5):
        '''
        Initializes the RMSNorm layer with a custom CUDA kernel.

        Args:
            num_features (int): Number of features in the input tensor. (Used for API compatibility)
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        '''
        super(ModelNew, self).__init__()
        self.eps = eps

        # JIT compilation of the CUDA kernel
        self.rms_norm_forward_cuda = self._load_kernel()

    def _load_kernel(self):
        cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cub/cub.cuh>

template <typename T, int BLOCK_SIZE>
__global__ void rms_norm_cub_grid_stride_unrolled_kernel(const T* __restrict__ x, T* __restrict__ y, int num_rows, int num_features, T eps) {
    // Shared memory for CUB's reduction algorithm.
    __shared__ typename cub::BlockReduce<T, BLOCK_SIZE>::TempStorage temp_storage;
    // Shared memory for broadcasting the inv_rms_val.
    __shared__ T s_inv_rms_val;

    const int num_features_vec = num_features / 4;

    // Grid-stride loop to process multiple rows per block. This ensures the GPU remains
    // saturated with work, providing scalability across different input sizes.
    for (int row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x) {
        const float4* x_vec_row = reinterpret_cast<const float4*>(x + (size_t)row_idx * num_features);
        float4* y_vec_row = reinterpret_cast<float4*>(y + (size_t)row_idx * num_features);

        // 1. Calculate sum of squares with 2x loop unrolling to increase ILP.
        T sum_sq = static_cast<T>(0.0f);
        for (int i = threadIdx.x; i < num_features_vec; i += BLOCK_SIZE * 2) {
            float4 val_vec1 = x_vec_row[i];
            sum_sq += val_vec1.x * val_vec1.x + val_vec1.y * val_vec1.y + val_vec1.z * val_vec1.z + val_vec1.w * val_vec1.w;

            if (i + BLOCK_SIZE < num_features_vec) {
                float4 val_vec2 = x_vec_row[i + BLOCK_SIZE];
                sum_sq += val_vec2.x * val_vec2.x + val_vec2.y * val_vec2.y + val_vec2.z * val_vec2.z + val_vec2.w * val_vec2.w;
            }
        }

        // 2. Perform block-wide reduction using CUB's highly optimized primitive.
        T block_sum_sq = cub::BlockReduce<T, BLOCK_SIZE>(temp_storage).Sum(sum_sq);

        // 3. Thread 0 computes the inverse RMS value and broadcasts it via shared memory.
        if (threadIdx.x == 0) {
            T mean_sq = block_sum_sq / num_features;
            s_inv_rms_val = rsqrtf(mean_sq + eps);
        }
        __syncthreads();
        const T inv_rms_val = s_inv_rms_val;

        // 4. Apply normalization factor with 2x loop unrolling.
        for (int i = threadIdx.x; i < num_features_vec; i += BLOCK_SIZE * 2) {
            float4 val_vec1 = x_vec_row[i];
            val_vec1.x *= inv_rms_val;
            val_vec1.y *= inv_rms_val;
            val_vec1.z *= inv_rms_val;
            val_vec1.w *= inv_rms_val;
            y_vec_row[i] = val_vec1;

            if (i + BLOCK_SIZE < num_features_vec) {
                float4 val_vec2 = x_vec_row[i + BLOCK_SIZE];
                val_vec2.x *= inv_rms_val;
                val_vec2.y *= inv_rms_val;
                val_vec2.z *= inv_rms_val;
                val_vec2.w *= inv_rms_val;
                y_vec_row[i + BLOCK_SIZE] = val_vec2;
            }
        }

        // This sync is crucial for correctness within a grid-stride loop. It ensures that all
        // shared memory operations for one row are complete before any thread begins the next.
        __syncthreads();
    }
}

torch::Tensor rms_norm_launcher(const torch::Tensor& x, float eps) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() >= 2, "Input tensor must have at least 2 dimensions");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "Only float tensors are supported by this optimized kernel");

    const auto x_sizes = x.sizes();
    const int num_features = x_sizes[1];
    const int num_rows = x.numel() / num_features;

    TORCH_CHECK(num_features % 4 == 0, "Number of features must be divisible by 4 for the vectorized kernel.");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "Input tensor must be 16-byte aligned for float4 access.");

    auto y = torch::empty_like(x);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0, "Output tensor must be 16-byte aligned for float4 access.");

    // Launch configuration:
    // BLOCK_SIZE: Increased to 512 for better parallelism and reduction efficiency with CUB.
    // grid_size: Tuned to the number of SMs on an A100 GPU (108) to maximize work per block,
    // thereby amortizing kernel launch overhead. The grid-stride loop handles work distribution.
    constexpr int BLOCK_SIZE = 512;
    const int grid_size = 108;
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);

    rms_norm_cub_grid_stride_unrolled_kernel<float, BLOCK_SIZE><<<grid, block>>>(
        x.const_data_ptr<float>(),
        y.data_ptr<float>(),
        num_rows,
        num_features,
        static_cast<float>(eps)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch error: ") + cudaGetErrorString(err));
    }

    return y;
}
        '''
        try:
            # Using a unique name to avoid JIT cache conflicts.
            kernel_module = load_inline(
                name='rms_norm_cub_grid_stride_unrolled_tuned',
                cpp_sources='',
                cuda_sources=cuda_source,
                functions=['rms_norm_launcher'],
                with_cuda=True,
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17'],
                verbose=False
            )
            return kernel_module.rms_norm_launcher
        except Exception as e:
            print(f"Failed to load custom CUDA kernel: {e}")
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies RMS Normalization using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, ...).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        '''
        # Check conditions for using the custom kernel.
        # It requires CUDA, contiguous memory, 2+ dimensions, float type,
        # num_features divisible by 4, and 16-byte alignment for float4 operations.
        use_cuda_kernel = (
            self.rms_norm_forward_cuda and
            x.is_cuda and
            x.is_contiguous() and
            x.dim() >= 2 and
            x.dtype == torch.float32 and
            x.size(1) % 4 == 0 and
            x.data_ptr() % 16 == 0
        )

        if use_cuda_kernel:
             return self.rms_norm_forward_cuda(x, self.eps)
        else:
            # Fallback to the original PyTorch implementation for CPU, non-contiguous tensors,
            # or if the kernel failed to compile or conditions are not met.
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
            return x / rms