# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# This experiment builds upon the SoTA by combining two historically successful
# optimization strategies: increasing work-per-block and loop unrolling.
#
# Hypothesis:
# The SoTA kernel optimizes arithmetic by using nested loops to calculate memory
# addresses, but its 1-channel-per-block grid strategy may not be sufficient
# to hide global memory latency on modern GPUs. The best-performing historical
# kernels processed 4 channels per block, significantly improving performance.
# We hypothesize that combining this 4-channel-per-block grid strategy with
# the SoTA's efficient indexing will yield superior performance. Furthermore,
# we re-introduce 2x loop unrolling on the innermost loop (over the spatial
# plane), another proven technique from past experiments, to increase
# instruction-level parallelism.
#
# This experiment modifies the SoTA kernel to process 4 channels sequentially
# within each thread block. Both the statistics calculation and normalization
# loops are manually unrolled by a factor of 2.

multi_channel_layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

// A single fused kernel that computes statistics and applies normalization.
// It launches one block per FOUR channels. Block size is 256.
// The inner loop over the spatial dimension is unrolled by 2x.
__global__ void __launch_bounds__(256, 1) multi_channel_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,       // Chunked weight tensor
    const float* __restrict__ bias,         // Chunked bias tensor
    const int N_per_feature,                // Total elements per channel: B*H*W
    const int HW_plane_size,                // Elements in one plane: H*W
    const int C_chunk,
    const int B,
    const int C_total,
    const int c_start)
{
    // --- Map block to a group of 4 channels ---
    const int block_c_start = c_start + blockIdx.x * 4;

    extern __shared__ float sdata[];

    // Common constants for memory layout
    const long C_total_x_HW = (long)C_total * HW_plane_size;
    const int HW_plane_size_vec4 = HW_plane_size / 4;
    const int HW_plane_size_vec8 = HW_plane_size_vec4 / 2;

    // Outer loop over the 4 channels assigned to this block
    for (int c_idx = 0; c_idx < 4; ++c_idx) {
        const int c_global = block_c_start + c_idx;

        // Ensure we don't process channels beyond the tensor dimensions
        if (c_global >= c_start + C_chunk) {
            break;
        }

        const long c_global_x_HW = (long)c_global * HW_plane_size;

        // --- PART 1: Calculate Statistics (Nested Loop + 2x Unroll) ---
        float thread_sum = 0.0f;
        float thread_sum_sq = 0.0f;

        // Loop over batches
        for (int b = 0; b < B; ++b) {
            const float* input_b_c_ptr = input + (long)b * C_total_x_HW + c_global_x_HW;
            
            // Unrolled inner loop over the spatial plane
            for (int i = threadIdx.x; i < HW_plane_size_vec8; i += blockDim.x) {
                const float4 val1 = *reinterpret_cast<const float4*>(input_b_c_ptr + (long)i * 8);
                const float4 val2 = *reinterpret_cast<const float4*>(input_b_c_ptr + (long)i * 8 + 4);
                thread_sum += val1.x + val1.y + val1.z + val1.w;
                thread_sum_sq += val1.x * val1.x + val1.y * val1.y + val1.z * val1.z + val1.w * val1.w;
                thread_sum += val2.x + val2.y + val2.z + val2.w;
                thread_sum_sq += val2.x * val2.x + val2.y * val2.y + val2.z * val2.z + val2.w * val2.w;
            }
            // Cleanup loop for odd-length spatial planes
            if ((HW_plane_size_vec4 % 2 != 0) && (threadIdx.x == 0)) {
                const float4 val = *reinterpret_cast<const float4*>(input_b_c_ptr + (long)(HW_plane_size_vec4 - 1) * 4);
                thread_sum += val.x + val.y + val.z + val.w;
                thread_sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
            }
        }

        // --- Intra-block reduction ---
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
            thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
        }
        
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        const int num_warps = blockDim.x / 32;
        if (lane_id == 0) {
            sdata[warp_id] = thread_sum;
            sdata[warp_id + num_warps] = thread_sum_sq;
        }
        __syncthreads();
        
        if (warp_id == 0) {
            float warp_sum = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
            float warp_sum_sq = (lane_id < num_warps) ? sdata[lane_id + num_warps] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                if (num_warps > offset) {
                    warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
                    warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
                }
            }
            if (lane_id == 0) {
                const float N_float = static_cast<float>(N_per_feature);
                const float epsilon = 1e-5f;
                const float mu = warp_sum / N_float;
                sdata[0] = mu;
                sdata[1] = rsqrtf(warp_sum_sq / N_float - mu * mu + epsilon);
            }
        }
        __syncthreads();

        // --- PART 2: Apply Normalization (Nested Loop + 2x Unroll) ---
        const float mean_c = sdata[0];
        const float inv_var_c = sdata[1];
        const int c_local_in_chunk = blockIdx.x * 4 + c_idx;
        const float weight_c = weight[c_local_in_chunk];
        const float bias_c = bias[c_local_in_chunk];

        for (int b = 0; b < B; ++b) {
            const long base_idx = (long)b * C_total_x_HW + c_global_x_HW;
            const float* input_b_c_ptr = input + base_idx;
            float* output_b_c_ptr = output + base_idx;

            for (int i = threadIdx.x; i < HW_plane_size_vec8; i += blockDim.x) {
                const long offset = (long)i * 8;
                const float4 in_val1 = *reinterpret_cast<const float4*>(input_b_c_ptr + offset);
                const float4 in_val2 = *reinterpret_cast<const float4*>(input_b_c_ptr + offset + 4);
                float4 out_val1, out_val2;
                out_val1.x = (in_val1.x - mean_c) * inv_var_c * weight_c + bias_c;
                out_val1.y = (in_val1.y - mean_c) * inv_var_c * weight_c + bias_c;
                out_val1.z = (in_val1.z - mean_c) * inv_var_c * weight_c + bias_c;
                out_val1.w = (in_val1.w - mean_c) * inv_var_c * weight_c + bias_c;
                out_val2.x = (in_val2.x - mean_c) * inv_var_c * weight_c + bias_c;
                out_val2.y = (in_val2.y - mean_c) * inv_var_c * weight_c + bias_c;
                out_val2.z = (in_val2.z - mean_c) * inv_var_c * weight_c + bias_c;
                out_val2.w = (in_val2.w - mean_c) * inv_var_c * weight_c + bias_c;
                *reinterpret_cast<float4*>(output_b_c_ptr + offset) = out_val1;
                *reinterpret_cast<float4*>(output_b_c_ptr + offset + 4) = out_val2;
            }
            if ((HW_plane_size_vec4 % 2 != 0) && (threadIdx.x == 0)) {
                const long offset = (long)(HW_plane_size_vec4 - 1) * 4;
                const float4 in_val = *reinterpret_cast<const float4*>(input_b_c_ptr + offset);
                float4 out_val;
                out_val.x = (in_val.x - mean_c) * inv_var_c * weight_c + bias_c;
                out_val.y = (in_val.y - mean_c) * inv_var_c * weight_c + bias_c;
                out_val.z = (in_val.z - mean_c) * inv_var_c * weight_c + bias_c;
                out_val.w = (in_val.w - mean_c) * inv_var_c * weight_c + bias_c;
                *reinterpret_cast<float4*>(output_b_c_ptr + offset) = out_val;
            }
        }
    } // end of loop over 4 channels
}
} // anonymous namespace

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(input.is_contiguous(torch::MemoryFormat::Contiguous), "Input tensor must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4D");

    const auto B = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int N_per_feature = B * H * W;
    const int HW_plane_size = H * W;

    TORCH_CHECK(C % 4 == 0, "Input tensor channels must be divisible by 4.");
    TORCH_CHECK(W % 16 == 0, "Input tensor width must be divisible by 16.");

    auto output = torch::empty_like(input);

    const int num_streams = 4;
    std::vector<c10::cuda::CUDAStream> streams;
    for (int i = 0; i < num_streams; ++i) {
        streams.push_back(c10::cuda::getStreamFromPool());
    }
    const int C_per_stream = (C + num_streams - 1) / num_streams;

    for (int i = 0; i < num_streams; ++i) {
        c10::cuda::CUDAStreamGuard stream_guard(streams[i]);
        const int c_start = i * C_per_stream;
        const int c_end = std::min((int)C, c_start + C_per_stream);
        const int C_chunk = c_end - c_start;
        if (C_chunk <= 0) continue;

        const float* weight_ptr_chunk = weight.data_ptr<float>() + c_start;
        const float* bias_ptr_chunk = bias.data_ptr<float>() + c_start;

        const int threads_per_block = 256;
        const int C_chunk_div_4 = (C_chunk + 3) / 4;
        dim3 grid(C_chunk_div_4);
        dim3 block(threads_per_block);
        const int num_warps = threads_per_block / 32;
        size_t shared_mem_size = num_warps * 2 * sizeof(float);
        
        multi_channel_fused_kernel<<<grid, block, shared_mem_size, streams[i].stream()>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            weight_ptr_chunk,
            bias_ptr_chunk,
            N_per_feature, HW_plane_size, C_chunk, B, C, c_start);
    }
    
    return output;
}
"""

layer_norm_cpp_source = """
torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

layer_norm_multi_channel = load_inline(
    name="layer_norm_multi_channel",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=multi_channel_layernorm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__"
    ]
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Hard-coded parameters for the kernel
        self.min_width_multiple = 16
        self.channel_multiple = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % self.channel_multiple != 0:
            raise ValueError(f"The number of channels ({x.shape[1]}) must be divisible by {self.channel_multiple} for this implementation.")
        if x.shape[3] % self.min_width_multiple != 0:
            raise ValueError(f"The width of the input tensor must be divisible by {self.min_width_multiple}.")
        
        return layer_norm_multi_channel.layer_norm_cuda(x, self.weight, self.bias)
# RegexTagCustomPruningAlgorithmEnd