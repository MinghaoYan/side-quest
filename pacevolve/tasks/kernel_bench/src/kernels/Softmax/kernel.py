# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100-SXM4-40GB
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

softmax_fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

constexpr int WARP_SIZE = 32;

// Struct to hold the max value and the sum of exponentials.
// This is used for the combined reduction.
struct MaxSumPair {
    float max_val;
    float sum_val;
};

// Functor to combine two MaxSumPair instances using a numerically
// stable online update rule. This is the core of the fused reduction.
struct CombineOp {
    __device__ __forceinline__ MaxSumPair operator()(MaxSumPair a, MaxSumPair b) const {
        if (a.max_val > b.max_val) {
            return {a.max_val, a.sum_val + b.sum_val * __expf(b.max_val - a.max_val)};
        } else {
            return {b.max_val, b.sum_val + a.sum_val * __expf(a.max_val - b.max_val)};
        }
    }
};

// Intra-warp reduction for the MaxSumPair struct.
// It works by shuffling the individual members of the struct.
__device__ __forceinline__ MaxSumPair warp_reduce_pair(MaxSumPair val, const CombineOp& op) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        MaxSumPair other;
        other.max_val = __shfl_down_sync(0xffffffff, val.max_val, offset);
        other.sum_val = __shfl_down_sync(0xffffffff, val.sum_val, offset);
        val = op(val, other);
    }
    return val;
}

// Block-wide reduction for the MaxSumPair struct.
__device__ __forceinline__ MaxSumPair block_reduce_pair(MaxSumPair val, const CombineOp& op, MaxSumPair identity) {
    // Shared memory for intermediate warp results.
    extern __shared__ MaxSumPair s_warp_results[];
    
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Each warp performs its own reduction.
    val = warp_reduce_pair(val, op);

    // Warp leaders write their partial result to shared memory.
    if (lane_id == 0) {
        s_warp_results[warp_id] = val;
    }
    
    __syncthreads();

    // The first warp reduces the results from all other warps.
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? s_warp_results[lane_id] : identity;
    
    if (warp_id == 0) {
        val = warp_reduce_pair(val, op);
    }
    
    // The final result is in lane 0 of warp 0. Broadcast it to all threads.
    if (threadIdx.x == 0) {
        s_warp_results[0] = val;
    }
    __syncthreads();
    
    return s_warp_results[0];
}


// This kernel fuses the max and sum reductions into a single pass.
// It reduces the number of loops over the data from 3 (in the baseline) to 2.
__global__ void softmax_fused_reduction_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* row_input = input + batch_idx * dim;
    float* row_output = output + batch_idx * dim;
    
    // === Pass 1: Fused calculation and reduction of (max, sum) pair ===
    MaxSumPair thread_pair = {-FLT_MAX, 0.0f};

    // Each thread computes its local (max, sum) pair in a single loop
    // using the online update algorithm.
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = row_input[i];
        if (val > thread_pair.max_val) {
            thread_pair.sum_val = thread_pair.sum_val * __expf(thread_pair.max_val - val) + 1.0f;
            thread_pair.max_val = val;
        } else {
            thread_pair.sum_val += __expf(val - thread_pair.max_val);
        }
    }

    // Reduce the pairs across the entire block.
    MaxSumPair identity = {-FLT_MAX, 0.0f};
    MaxSumPair block_pair = block_reduce_pair(thread_pair, CombineOp(), identity);
    
    float max_val = block_pair.max_val;
    float sum_val = block_pair.sum_val;

    // === Pass 2: Normalize and write to output ===
    // This pass re-reads the input, which is faster than storing intermediate
    // exponentiated values in global memory.
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        row_output[i] = __expf(row_input[i] - max_val) / sum_val;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");

    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 512;
    const int blocks = batch_size;
    
    // Shared memory size is for the intermediate warp results of the reduction.
    const int warps_per_block = (threads + WARP_SIZE - 1) / WARP_SIZE;
    size_t shared_mem_size = warps_per_block * sizeof(MaxSumPair);

    softmax_fused_reduction_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

softmax_cpp_source = """
#include <torch/extension.h>

torch::Tensor softmax_cuda(torch::Tensor input);
"""

# JIT compile the CUDA and C++ code
softmax_module = load_inline(
    name='softmax_cuda_fused',
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_fused_cuda_source,
    functions=['softmax_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 0):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the custom CUDA softmax operation with fused reduction.
        
        Args:
            x (torch.Tensor): A 2D tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: The output tensor after applying softmax, with the same shape as input.
        """
        return self.softmax_cuda(x)

# RegexTagCustomPruningAlgorithmEnd