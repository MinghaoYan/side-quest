# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100 to enable optimizations like __shfl_down_sync
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# The CUDA and C++ source code for the optimized RNN cell
rnn_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Device function for a fast, parallel sum reduction within a warp.
// It uses shuffle-down instructions to aggregate values without shared memory.
__device__ inline float warp_reduce_sum(float val) {
    // Each thread in a warp adds the value from the thread 'offset' lanes away.
    // This is done in log2(warpSize) steps.
    // The mask 0xffffffff ensures all threads in the warp participate.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // The final sum resides in the first thread (lane 0) of the warp.
    return val;
}

__global__ void rnn_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ hidden,
    const float* __restrict__ i2h_weight,
    const float* __restrict__ i2h_bias,
    const float* __restrict__ h2o_weight,
    const float* __restrict__ h2o_bias,
    float* __restrict__ new_hidden,
    float* __restrict__ output,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int output_size
) {
    // Use dynamic shared memory, sized at kernel launch.
    extern __shared__ float s_mem[];

    const int K = input_size + hidden_size;
    // Partition shared memory for the combined input and the new hidden state.
    float* s_x_combined = s_mem;
    float* s_new_hidden = &s_mem[K];

    // Each block processes one item in the batch.
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Cooperatively load the combined input and hidden state into shared memory.
    // This reduces global memory reads and improves performance.
    for (int i = tid; i < input_size; i += num_threads) {
        s_x_combined[i] = input[batch_idx * input_size + i];
    }
    for (int i = tid; i < hidden_size; i += num_threads) {
        s_x_combined[input_size + i] = hidden[batch_idx * hidden_size + i];
    }
    __syncthreads();

    // --- Part 1: Input-to-Hidden (i2h) Matrix-Vector Multiplication ---
    const int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int num_warps = num_threads / warp_size;

    // Each warp computes one or more output elements of the hidden state.
    for (int h = warp_id; h < hidden_size; h += num_warps) {
        float sum = 0.0f;
        // Each thread in the warp computes a partial sum of the dot product.
        for (int k = lane_id; k < K; k += warp_size) {
            sum += s_x_combined[k] * i2h_weight[h * K + k];
        }
        sum = warp_reduce_sum(sum);

        // Lane 0 of each warp writes the final result to shared memory.
        if (lane_id == 0) {
            s_new_hidden[h] = tanhf(sum + i2h_bias[h]);
        }
    }
    __syncthreads();

    // --- Part 2: Hidden-to-Output (h2o) Matrix-Vector Multiplication ---
    // Each warp computes one or more output elements.
    for (int o = warp_id; o < output_size; o += num_warps) {
        float sum = 0.0f;
        // Each thread computes a partial sum using the new hidden state from shared memory.
        for (int h = lane_id; h < hidden_size; h += warp_size) {
            sum += s_new_hidden[h] * h2o_weight[o * hidden_size + h];
        }
        sum = warp_reduce_sum(sum);

        // Lane 0 of each warp writes the final result to global memory.
        if (lane_id == 0) {
            output[batch_idx * output_size + o] = sum + h2o_bias[o];
        }
    }

    // --- Part 3: Write the new hidden state to global memory for the next time step ---
    for (int h = tid; h < hidden_size; h += num_threads) {
        new_hidden[batch_idx * hidden_size + h] = s_new_hidden[h];
    }
}

std::vector<torch::Tensor> rnn_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(hidden.is_cuda(), "Hidden must be a CUDA tensor");
    
    const at::cuda::CUDAGuard device_guard(input.device());

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = hidden.size(1);
    const int output_size = h2o_weight.size(0);
    const int K = input_size + hidden_size;

    auto new_hidden = torch::empty_like(hidden);
    auto output = torch::empty({batch_size, output_size}, input.options());

    const int threads_per_block = 256;
    const dim3 blocks(batch_size);
    const dim3 threads(threads_per_block);
    
    // Calculate required shared memory size for the combined inputs and intermediate hidden state.
    const size_t shared_mem_size = (K + hidden_size) * sizeof(float);

    // Launch the fused kernel.
    rnn_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        hidden.data_ptr<float>(),
        i2h_weight.data_ptr<float>(),
        i2h_bias.data_ptr<float>(),
        h2o_weight.data_ptr<float>(),
        h2o_bias.data_ptr<float>(),
        new_hidden.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        output_size
    );
    
    // Check for CUDA errors after kernel launch.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return {output, new_hidden};
}
"""

rnn_cpp_source = """
#include <vector>
#include <torch/extension.h>

// Forward declaration of the C++ function that will be callable from Python.
std::vector<torch::Tensor> rnn_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias
);
"""

# Use JIT compilation to build the C++/CUDA source code into a Python module.
rnn_cuda_lib = load_inline(
    name='rnn_cuda_lib',
    cpp_sources=rnn_cpp_source,
    cuda_sources=rnn_cuda_source,
    functions=['rnn_forward_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        # Correctly use the arguments provided by the evaluation script.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # The hidden state is maintained by the module.
        # It will be initialized on the first forward pass.
        self.hidden = None
        
        # Standard PyTorch layers whose weights will be passed to our custom kernel.
        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically initialize or resize the hidden state based on the input batch size.
        # This makes the model more flexible than using a fixed batch size.
        batch_size = x.size(0)
        if self.hidden is None or self.hidden.size(0) != batch_size:
            self.hidden = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        # Ensure hidden state is on the same device as the input tensor.
        self.hidden = self.hidden.to(x.device)
        
        # Call the custom CUDA kernel for the forward pass.
        output, self.hidden = rnn_cuda_lib.rnn_forward_cuda(
            x,
            self.hidden,
            self.i2h.weight,
            self.i2h.bias,
            self.h2o.weight,
            self.h2o.bias
        )
        return output
# RegexTagCustomPruningAlgorithmEnd