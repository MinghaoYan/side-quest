# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100 (Compute Capability 8.0)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# C++ source for the function declaration (interface)
mean_reduction_cpp_source = """
#include <torch/extension.h>

// Forward declaration of the CUDA function to be called from Python
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);
"""

# CUDA source for the kernel and function definition (implementation)
mean_reduction_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// A fixed block size is used for this kernel. 512 is chosen as it's a power of two,
// which simplifies reduction logic, and generally performs well on modern GPUs like the A100.
#define BLOCK_SIZE 512

// Device function for an optimized block-level sum reduction.
// It combines fast intra-warp shuffles with shared memory for inter-warp reduction.
__device__ float blockReduceSum(float val) {
    // Shared memory is used to store the partial sum from each warp.
    // The size is the number of warps in the block (BLOCK_SIZE / 32).
    static __shared__ float shared_mem[BLOCK_SIZE / 32];

    int lane_id = threadIdx.x % 32; // Thread's lane index within its warp (0-31)
    int warp_id = threadIdx.x / 32; // The warp's index within the block

    // --- Step 1: Intra-warp reduction ---
    // Each warp reduces its 32 float values in parallel using shuffle-down instructions.
    // This is a tree-based reduction within the warp. After this loop, lane 0 of each
    // warp holds the sum of all 32 values in that warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // --- Step 2: Store warp-level sums in shared memory ---
    // The first thread (lane 0) of each warp writes its partial sum to shared memory.
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }

    // Synchronize all threads in the block to ensure all warp sums are written before proceeding.
    __syncthreads();

    // --- Step 3: Inter-warp reduction (performed by the first warp) ---
    // The first warp loads the partial sums from shared memory into its registers.
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared_mem[lane_id] : 0.0f;

    if (warp_id == 0) {
        // The first warp reduces the warp sums, again using shuffle-down instructions.
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
             val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        // The final sum for the entire block is now in lane 0 of warp 0.
        // We write it to a known location in shared memory (index 0) to be read by thread 0.
        if (lane_id == 0) {
            shared_mem[0] = val;
        }
    }

    // Synchronize again to ensure the final sum is visible to all threads.
    __syncthreads();
    
    // All threads return the final block-wide sum, which is read from shared_mem[0].
    return shared_mem[0];
}


__global__ void mean_reduction_kernel(const float* input, float* output, 
                                    const int reduce_dim_size, const long outer_size, const long inner_size) {
    // Each block is responsible for computing one element of the output tensor.
    // The block index 'idx' corresponds to a unique (outer_idx, inner_idx) pair.
    long idx = blockIdx.x;
    
    // Boundary check to prevent out-of-bounds access if the number of reductions
    // is not a multiple of the grid size.
    if (idx >= outer_size * inner_size) return;

    // De-flatten the block index to find the input data slice for this reduction.
    long outer_idx = idx / inner_size;
    long inner_idx = idx % inner_size;
    const float* input_ptr = input + outer_idx * reduce_dim_size * inner_size + inner_idx;
    
    float thread_sum = 0.0f;

    // Grid-stride loop: each thread sums a portion of the elements along the reduction dimension.
    // This allows a single block to handle reduction dimensions of any size, even larger
    // than the block size.
    for (int i = threadIdx.x; i < reduce_dim_size; i += BLOCK_SIZE) {
        thread_sum += input_ptr[i * inner_size];
    }

    // Use the optimized device function to reduce the sums from all threads in the block.
    float block_sum = blockReduceSum(thread_sum);

    // The first thread of the block writes the final mean value to the output tensor.
    if (threadIdx.x == 0) {
        if (reduce_dim_size > 0) {
            output[idx] = block_sum / reduce_dim_size;
        } else {
            // Handle reduction over a zero-sized dimension. The result should be NaN,
            // matching PyTorch's behavior.
            output[idx] = 0.0f / 0.0f;
        }
    }
}

// C++ wrapper function to be called from Python. It validates inputs,
// calculates dimensions, and launches the CUDA kernel.
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    // Input validation checks.
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous for direct pointer access");

    const auto input_sizes = input.sizes();
    const int ndim = input.dim();
    // Handle negative dimension indexing.
    dim = (dim < 0) ? (dim + ndim) : dim;

    TORCH_CHECK(ndim > 0 && dim >= 0 && dim < ndim, "Dimension out of range");
    
    const int reduce_dim_size = input_sizes[dim];
    
    // Calculate outer and inner dimensions for flattened view.
    long outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input_sizes[i];
    }
    
    long inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= input_sizes[i];
    }
    
    // Prepare output tensor with the reduced dimension removed.
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    
    auto output = torch::empty(output_sizes, input.options());
    
    long total_reductions = outer_size * inner_size;
    if (total_reductions == 0) {
        return output; // Return empty tensor if there is no work to do.
    }

    // Kernel launch configuration.
    const int num_blocks = total_reductions;
    const int threads_per_block = BLOCK_SIZE;
    
    mean_reduction_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reduce_dim_size,
        outer_size,
        inner_size
    );

    // Check for any CUDA errors during kernel execution.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    return output;
}
"""

# Use torch's JIT compiler to build the CUDA code.
mean_reduction_optimized = load_inline(
    name='mean_reduction_optimized',
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_cuda_source,
    functions=['mean_reduction_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        """
        Initializes the model.
        
        Args:
            num_features (int): This parameter is interpreted as the reduction
                                dimension 'dim' to adhere to the required function signature.
        """
        super(ModelNew, self).__init__()
        # To adhere to the specified __init__ signature `(self, num_features: int)`,
        # we interpret num_features as the dimension to reduce.
        self.dim = num_features
        self.mean_reduction = mean_reduction_optimized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model by applying the custom mean reduction.

        Args:
            x (torch.Tensor): The input tensor. Must be on a CUDA device.

        Returns:
            torch.Tensor: The output tensor after mean reduction.
        """
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)
# RegexTagCustomPruningAlgorithmEnd