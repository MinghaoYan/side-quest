import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    '''
    Highly optimized model that performs a Softmax activation using a custom CUDA kernel.
    This kernel fuses all operations by caching the input row in shared memory, minimizing
    global memory traffic. It uses float4 vectorization for high-bandwidth memory access
    and warp-level primitives for efficient reductions.
    '''
    def __init__(self):
        super(ModelNew, self).__init__()

        cuda_source = r'''
#include <cfloat>

// A warp-cooperative reduction helper for finding the maximum value.
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// A warp-cooperative reduction helper for summation.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused kernel that uses shared memory to reduce global memory I/O.
__global__ void softmax_smem_vectorized_kernel(const float* __restrict__ input, float* __restrict__ output, int num_features) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = block_size / 32;

    // Shared memory layout: [ s_data[num_features] | s_reduce[num_warps] ]
    extern __shared__ float s_mem[];
    float* s_data = s_mem;
    float* s_reduce = &s_mem[num_features];

    const float* row_input = input + row_idx * num_features;
    float* row_output = output + row_idx * num_features;
    const int num_features_vec = num_features - (num_features % 4);

    // --- Pass 1: Load row from global to shared memory (vectorized) ---
    for (int i = tid * 4; i < num_features_vec; i += block_size * 4) {
        *reinterpret_cast<float4*>(&s_data[i]) = *reinterpret_cast<const float4*>(&row_input[i]);
    }
    for (int i = num_features_vec + tid; i < num_features; i += block_size) {
        s_data[i] = row_input[i];
    }
    __syncthreads();

    // --- Pass 2: Find max value from shared memory (vectorized read) ---
    float thread_max = -FLT_MAX;
    for (int i = tid * 4; i < num_features_vec; i += block_size * 4) {
        float4 val4 = *reinterpret_cast<float4*>(&s_data[i]);
        thread_max = max(thread_max, max(max(val4.x, val4.y), max(val4.z, val4.w)));
    }
    for (int i = num_features_vec + tid; i < num_features; i += block_size) {
        thread_max = max(thread_max, s_data[i]);
    }
    float warp_max = warp_reduce_max(thread_max);
    if (lane_id == 0) s_reduce[warp_id] = warp_max;
    __syncthreads();
    float block_max = (tid < num_warps) ? s_reduce[tid] : -FLT_MAX;
    if (warp_id == 0) block_max = warp_reduce_max(block_max);
    if (tid == 0) s_reduce[0] = block_max;
    __syncthreads();
    const float row_max = s_reduce[0];

    // --- Pass 3: Compute exp(x-max), overwrite shared memory, and sum (vectorized) ---
    float thread_sum = 0.0f;
    for (int i = tid * 4; i < num_features_vec; i += block_size * 4) {
        float4 val4 = *reinterpret_cast<float4*>(&s_data[i]);
        val4 = {expf(val4.x - row_max), expf(val4.y - row_max), expf(val4.z - row_max), expf(val4.w - row_max)};
        *reinterpret_cast<float4*>(&s_data[i]) = val4;
        thread_sum += val4.x + val4.y + val4.z + val4.w;
    }
    for (int i = num_features_vec + tid; i < num_features; i += block_size) {
        float val = expf(s_data[i] - row_max);
        s_data[i] = val;
        thread_sum += val;
    }
    __syncthreads(); // Ensure s_data is fully overwritten before reduction sum
    float warp_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) s_reduce[warp_id] = warp_sum;
    __syncthreads();
    float block_sum = (tid < num_warps) ? s_reduce[tid] : 0.0f;
    if (warp_id == 0) block_sum = warp_reduce_sum(block_sum);
    if (tid == 0) s_reduce[0] = block_sum;
    __syncthreads();
    const float row_sum = s_reduce[0];
    const float inv_sum = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;

    // --- Pass 4: Normalize from shared memory and write to global (vectorized) ---
    for (int i = tid * 4; i < num_features_vec; i += block_size * 4) {
        float4 val4 = *reinterpret_cast<float4*>(&s_data[i]);
        val4.x *= inv_sum; val4.y *= inv_sum; val4.z *= inv_sum; val4.w *= inv_sum;
        *reinterpret_cast<float4*>(&row_output[i]) = val4;
    }
    for (int i = num_features_vec + tid; i < num_features; i += block_size) {
        row_output[i] = s_data[i] * inv_sum;
    }
}

void softmax_launcher(const float* input, float* output, int batch_size, int num_features) {
    const int block_size = 1024;
    const dim3 grid_dim(batch_size);
    const dim3 block_dim(block_size);
    const int num_warps = (block_size + 31) / 32;

    if (num_features > 0) {
        const size_t shared_mem_size = (num_features + num_warps) * sizeof(float);
        softmax_smem_vectorized_kernel<<<grid_dim, block_dim, shared_mem_size>>>(input, output, num_features);
    }
}
        '''

        cpp_source = r'''
#include <torch/extension.h>
#include <vector>

void softmax_launcher(const float* input, float* output, int batch_size, int num_features);

void softmax_forward_cuda(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(output.is_cuda(), "Output tensor must be on a CUDA device");

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    softmax_launcher(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, num_features);
}
        '''

        self.softmax_custom_op = load_inline(
            name="softmax_fused_smem_vec",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['softmax_forward_cuda'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies the custom fused Softmax activation to the input tensor.
        '''
        if x.dim() != 2:
            return torch.softmax(x, dim=1)

        # Fallback for large feature sizes that exceed shared memory capacity.
        # (12256 features + 32 warps) * 4 bytes/float = 49152 bytes = 48 KB.
        if x.shape[1] > 12256:
            return torch.softmax(x, dim=1)

        x_cont = x.contiguous()
        output = torch.empty_like(x_cont)

        self.softmax_custom_op.softmax_forward_cuda(x_cont, output)

        return output