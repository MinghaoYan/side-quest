#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Inlined device function to perform the sequence of operations on a single float.
// Using __forceinline__ encourages the compiler to inline this, reducing function call overhead.
__device__ __forceinline__ float apply_fused_ops(float val, const float bias, const float scaling_factor) {
    val += bias;
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    val *= scaling_factor;
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    val /= scaling_factor;
    return val;
}

__global__ void custom_kernel_vectorized(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    const int total_size, 
    const int feature_map_size, 
    const int out_channels, 
    const float scaling_factor) {
    
    const int total_size_vec = total_size / 4;
    // A grid-stride loop allows the kernel to be flexible to the number of threads launched.
    const int stride = gridDim.x * blockDim.x;

    // Vectorized part using float4
    // Each thread starts at its global ID and processes elements in increments of the grid size.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size_vec; i += stride) {
        const int base_idx_scalar = i * 4;
        
        // All 4 elements in a float4 load are from the same channel, given NCHW layout and W > 4.
        const int c = (base_idx_scalar / feature_map_size) % out_channels;
        const float bias_val = bias[c];
        
        // Load 4 floats at once using float4.
        float4 data_vec = reinterpret_cast<const float4*>(input)[i];
        
        // Apply operations element-wise to the vector components.
        data_vec.x = apply_fused_ops(data_vec.x, bias_val, scaling_factor);
        data_vec.y = apply_fused_ops(data_vec.y, bias_val, scaling_factor);
        data_vec.z = apply_fused_ops(data_vec.z, bias_val, scaling_factor);
        data_vec.w = apply_fused_ops(data_vec.w, bias_val, scaling_factor);
        
        // Store 4 floats at once.
        reinterpret_cast<float4*>(output)[i] = data_vec;
    }

    // Remainder part (handles cases where total_size is not a multiple of 4)
    // The same grid-stride loop pattern is used for the remaining scalar elements.
    const int remainder_start_idx = total_size_vec * 4;
    for (int i = remainder_start_idx + blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += stride) {
        const int c = (i / feature_map_size) % out_channels;
        output[i] = apply_fused_ops(input[i], bias[c], scaling_factor);
    }
}

// C++ wrapper function to launch the CUDA kernel from PyTorch
torch::Tensor custom_cuda(
    torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias tensor must be contiguous");

    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int out_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int total_size = batch_size * out_channels * height * width;
    if (total_size == 0) {
        return output;
    }
    const int feature_map_size = height * width;
    
    const int block_size = 256;
    // Calculate grid size. Launch enough threads to cover all vector elements for good parallelism.
    const int num_vec_elements = (total_size + 3) / 4;
    const int num_blocks = (num_vec_elements + block_size - 1) / block_size;
    
    custom_kernel_vectorized<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        total_size, 
        feature_map_size, 
        out_channels, 
        scaling_factor);
        
    // Check for CUDA errors after kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}
