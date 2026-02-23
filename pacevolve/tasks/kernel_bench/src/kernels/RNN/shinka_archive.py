import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def load_fully_fused_rnn_kernel():
    cuda_source = r'''
    #include <torch/extension.h>
    #include <cmath>

    // --- Configuration ---
    // The number of output elements a single warp will compute in parallel.
    #define WORK_PER_WARP 4
    // The unroll factor for the main reduction loops.
    #define UNROLL_FACTOR 6

    // Helper for vectorized dot product.
    __device__ __forceinline__ float dot(const float4& a, const float4& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // This kernel fully fuses the RNN cell computation (i2h GEMM, tanh, h2o GEMM)
    // and employs multiple optimizations:
    // 1. Input Caching: `x` and `hidden` are cached in shared memory to reduce global memory traffic.
    // 2. Increased Work per Warp: Each warp computes WORK_PER_WARP=4 outputs to increase ILP.
    // 3. Float4 Vectorization: Maximizes memory bandwidth.
    // 4. Loop Unrolling: Increases ILP to hide latency.
    // 5. Fast Tanh Approximation: Uses a faster math implementation.
    __global__ void fully_fused_rnn_optimized_kernel(
        const float* __restrict__ x,          // Input tensor [B, I]
        const float* __restrict__ hidden,     // Hidden state tensor [B, H]
        const float* __restrict__ W_ih,       // Weight for i2h [H, I + H]
        const float* __restrict__ b_ih,       // Bias for i2h [H]
        const float* __restrict__ W_ho,       // Weight for h2o [O, H]
        const float* __restrict__ b_ho,       // Bias for h2o [O]
        float* __restrict__ new_hidden,     // Output new hidden state [B, H]
        float* __restrict__ output,         // Final output tensor [B, O]
        const int B, const int I, const int H, const int O) {

        const int b = blockIdx.x;
        if (b >= B) return;

        extern __shared__ float sh_mem[];
        float* sh_combined_input = sh_mem;
        float* sh_new_hidden = sh_mem + I + H;

        const int tid = threadIdx.x;
        const int warp_id = tid / warpSize;
        const int lane_id = tid % warpSize;
        const int num_warps_per_block = blockDim.x / warpSize;
        const int block_size = blockDim.x;

        const int I_VEC = I / 4;
        const int H_VEC = H / 4;
        const int IH_VEC = (I + H) / 4;

        // --- Part 0: Collaboratively load x and hidden into shared memory ---
        for (int k_vec = tid; k_vec < IH_VEC; k_vec += block_size) {
            if (k_vec < I_VEC) {
                ((float4*)sh_combined_input)[k_vec] = ((const float4*)x)[b * I_VEC + k_vec];
            } else {
                ((float4*)sh_combined_input)[k_vec] = ((const float4*)hidden)[b * H_VEC + (k_vec - I_VEC)];
            }
        }
        __syncthreads();

        // --- Part 1: Compute new_hidden (i2h GEMV + tanh) into shared memory ---
        for (int j_base = warp_id * WORK_PER_WARP; j_base < H; j_base += num_warps_per_block * WORK_PER_WARP) {
            float accumulators[WORK_PER_WARP] = {0.0f};

            // 6x Manually unrolled reduction loop with a separate epilogue loop.
            const int UNROLL_STEP_IH = warpSize * UNROLL_FACTOR;
            const int main_loop_end_ih = (IH_VEC / UNROLL_STEP_IH) * UNROLL_STEP_IH;

            for (int k_base = lane_id; k_base < main_loop_end_ih; k_base += UNROLL_STEP_IH) {
                const float4 in_chunk1 = ((const float4*)sh_combined_input)[k_base];
                const float4 in_chunk2 = ((const float4*)sh_combined_input)[k_base + warpSize];
                const float4 in_chunk3 = ((const float4*)sh_combined_input)[k_base + warpSize * 2];
                const float4 in_chunk4 = ((const float4*)sh_combined_input)[k_base + warpSize * 3];
                const float4 in_chunk5 = ((const float4*)sh_combined_input)[k_base + warpSize * 4];
                const float4 in_chunk6 = ((const float4*)sh_combined_input)[k_base + warpSize * 5];

                #pragma unroll
                for (int i = 0; i < WORK_PER_WARP; ++i) {
                    if (j_base + i < H) {
                        const int W_row_base = (j_base + i) * IH_VEC;
                        accumulators[i] += dot(in_chunk1, ((const float4*)W_ih)[W_row_base + k_base]);
                        accumulators[i] += dot(in_chunk2, ((const float4*)W_ih)[W_row_base + k_base + warpSize]);
                        accumulators[i] += dot(in_chunk3, ((const float4*)W_ih)[W_row_base + k_base + warpSize * 2]);
                        accumulators[i] += dot(in_chunk4, ((const float4*)W_ih)[W_row_base + k_base + warpSize * 3]);
                        accumulators[i] += dot(in_chunk5, ((const float4*)W_ih)[W_row_base + k_base + warpSize * 4]);
                        accumulators[i] += dot(in_chunk6, ((const float4*)W_ih)[W_row_base + k_base + warpSize * 5]);
                    }
                }
            }

            // Epilogue loop for remaining elements
            for (int k_base = main_loop_end_ih + lane_id; k_base < IH_VEC; k_base += warpSize) {
                const float4 in_chunk = ((const float4*)sh_combined_input)[k_base];
                #pragma unroll
                for (int i = 0; i < WORK_PER_WARP; ++i) {
                    if (j_base + i < H) {
                        accumulators[i] += dot(in_chunk, ((const float4*)W_ih)[(j_base + i) * IH_VEC + k_base]);
                    }
                }
            }

            // Warp-level reduction for each accumulator
            #pragma unroll
            for (int i = 0; i < WORK_PER_WARP; ++i) {
                #pragma unroll
                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                    accumulators[i] += __shfl_down_sync(0xFFFFFFFF, accumulators[i], offset);
                }
            }

            if (lane_id == 0) {
                #pragma unroll
                for (int i = 0; i < WORK_PER_WARP; ++i) {
                    if (j_base + i < H) {
                        const int j = j_base + i;
                        float val = accumulators[i] + b_ih[j];
                        float exp2x = __expf(2.0f * val);
                        float hidden_val = (exp2x - 1.0f) / (exp2x + 1.0f);
                        sh_new_hidden[j] = hidden_val;
                    }
                }
            }
        }
        __syncthreads();

        // --- Part 2: Compute final output (h2o GEMV) ---
        for (int o_base = warp_id * WORK_PER_WARP; o_base < O; o_base += num_warps_per_block * WORK_PER_WARP) {
            float out_accumulators[WORK_PER_WARP] = {0.0f};

            // 6x Manually unrolled reduction loop with a separate epilogue loop.
            const int UNROLL_STEP_H = warpSize * UNROLL_FACTOR;
            const int main_loop_end_h = (H_VEC / UNROLL_STEP_H) * UNROLL_STEP_H;

            for (int k_base = lane_id; k_base < main_loop_end_h; k_base += UNROLL_STEP_H) {
                const float4 h_chunk1 = ((const float4*)sh_new_hidden)[k_base];
                const float4 h_chunk2 = ((const float4*)sh_new_hidden)[k_base + warpSize];
                const float4 h_chunk3 = ((const float4*)sh_new_hidden)[k_base + warpSize * 2];
                const float4 h_chunk4 = ((const float4*)sh_new_hidden)[k_base + warpSize * 3];
                const float4 h_chunk5 = ((const float4*)sh_new_hidden)[k_base + warpSize * 4];
                const float4 h_chunk6 = ((const float4*)sh_new_hidden)[k_base + warpSize * 5];
                #pragma unroll
                for (int i = 0; i < WORK_PER_WARP; ++i) {
                    if (o_base + i < O) {
                        const int W_row_base = (o_base + i) * H_VEC;
                        out_accumulators[i] += dot(h_chunk1, ((const float4*)W_ho)[W_row_base + k_base]);
                        out_accumulators[i] += dot(h_chunk2, ((const float4*)W_ho)[W_row_base + k_base + warpSize]);
                        out_accumulators[i] += dot(h_chunk3, ((const float4*)W_ho)[W_row_base + k_base + warpSize * 2]);
                        out_accumulators[i] += dot(h_chunk4, ((const float4*)W_ho)[W_row_base + k_base + warpSize * 3]);
                        out_accumulators[i] += dot(h_chunk5, ((const float4*)W_ho)[W_row_base + k_base + warpSize * 4]);
                        out_accumulators[i] += dot(h_chunk6, ((const float4*)W_ho)[W_row_base + k_base + warpSize * 5]);
                    }
                }
            }

            // Epilogue loop for remaining elements
            for (int k_base = main_loop_end_h + lane_id; k_base < H_VEC; k_base += warpSize) {
                const float4 h_chunk = ((const float4*)sh_new_hidden)[k_base];
                #pragma unroll
                for (int i = 0; i < WORK_PER_WARP; ++i) {
                    if (o_base + i < O) {
                        out_accumulators[i] += dot(h_chunk, ((const float4*)W_ho)[(o_base + i) * H_VEC + k_base]);
                    }
                }
            }

            // Warp-level reduction
            #pragma unroll
            for (int i = 0; i < WORK_PER_WARP; ++i) {
                #pragma unroll
                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                    out_accumulators[i] += __shfl_down_sync(0xFFFFFFFF, out_accumulators[i], offset);
                }
            }

            if (lane_id == 0) {
                #pragma unroll
                for (int i = 0; i < WORK_PER_WARP; ++i) {
                    if (o_base + i < O) {
                        const int o = o_base + i;
                        output[b * O + o] = out_accumulators[i] + b_ho[o];
                    }
                }
            }
        }
        __syncthreads();

        // --- Part 3: Deferred global write of new_hidden ---
        // Collaboratively write the final hidden state from shared to global memory.
        for (int j_vec = tid; j_vec < H_VEC; j_vec += block_size) {
            ((float4*)new_hidden)[b * H_VEC + j_vec] = ((const float4*)sh_new_hidden)[j_vec];
        }
    }

    void launch_fully_fused_rnn_cell_kernel(
        const torch::Tensor& x, const torch::Tensor& hidden,
        const torch::Tensor& W_ih, const torch::Tensor& b_ih,
        const torch::Tensor& W_ho, const torch::Tensor& b_ho,
        torch::Tensor& new_hidden, torch::Tensor& output) {

        const int B = x.size(0);
        const int I = x.size(1);
        const int H = hidden.size(1);
        const int O = output.size(1);

        const dim3 threads_per_block(512);
        const dim3 num_blocks(B);
        const size_t shared_mem_size = (I + H + H) * sizeof(float);

        fully_fused_rnn_optimized_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
            x.data_ptr<float>(), hidden.data_ptr<float>(),
            W_ih.data_ptr<float>(), b_ih.data_ptr<float>(),
            W_ho.data_ptr<float>(), b_ho.data_ptr<float>(),
            new_hidden.data_ptr<float>(), output.data_ptr<float>(),
            B, I, H, O
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    '''

    cpp_source = "void launch_fully_fused_rnn_cell_kernel(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, torch::Tensor&);"

    fused_rnn_lib = load_inline(
        name='fully_fused_rnn_pipelined_lib',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['launch_fully_fused_rnn_cell_kernel'],
        with_cuda=True,
        extra_cuda_cflags=['-O3', '--use_fast_math']
    )
    return fused_rnn_lib

fused_rnn_lib = load_fully_fused_rnn_kernel()

class FullyFusedRNNCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hidden, weight_ih, bias_ih, weight_ho, bias_ho):
        new_hidden = torch.empty_like(hidden)
        output = torch.empty(x.size(0), weight_ho.size(0), device=x.device, dtype=x.dtype)

        fused_rnn_lib.launch_fully_fused_rnn_cell_kernel(
            x, hidden, weight_ih, bias_ih, weight_ho, bias_ho, new_hidden, output
        )

        return new_hidden, output

    @staticmethod
    def backward(ctx, grad_hidden, grad_output):
        # Backward pass is not implemented as it's not required for inference.
        return None, None, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.register_buffer('hidden', torch.randn(batch_size, hidden_size))

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        hidden = self.hidden.contiguous()

        new_hidden, output = FullyFusedRNNCellFunction.apply(
            x, hidden, self.i2h.weight, self.i2h.bias, self.h2o.weight, self.h2o.bias
        )

        self.hidden = new_hidden

        return output