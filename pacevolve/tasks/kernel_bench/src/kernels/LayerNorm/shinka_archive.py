import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math
import os

def load_layernorm_cuda_kernel():
    cuda_source = r'''
    #include <torch/extension.h>
    #include <c10/cuda/CUDAGuard.h>
    #include <c10/cuda/CUDAException.h>
    #include <cmath>
    #include <vector_types.h>
    #include <cuda_bf16.h>

    struct Float2 {
        float x, y;
    };

    __device__ __forceinline__ Float2 block_reduce_sum_two(float val1, float val2, float* shared_mem) {
        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp_id = tid / 32;

        float* smem1 = shared_mem;
        float* smem2 = smem1 + blockDim.x / 32;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val1 += __shfl_down_sync(0xFFFFFFFF, val1, offset);
            val2 += __shfl_down_sync(0xFFFFFFFF, val2, offset);
        }

        if (lane == 0) {
            smem1[warp_id] = val1;
            smem2[warp_id] = val2;
        }
        __syncthreads();

        val1 = (tid < blockDim.x / 32) ? smem1[lane] : 0.0f;
        val2 = (tid < blockDim.x / 32) ? smem2[lane] : 0.0f;
        if (warp_id == 0) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                val1 += __shfl_down_sync(0xFFFFFFFF, val1, offset);
                val2 += __shfl_down_sync(0xFFFFFFFF, val2, offset);
            }
        }

        if (tid == 0) {
            shared_mem[0] = val1;
            shared_mem[1] = val2;
        }
        __syncthreads();
        return {shared_mem[0], shared_mem[1]};
    }

    // Kernel 1: Computes partial sum/sum_sq for the entire row and stores as bfloat16.
    __global__ void layernorm_fwd_kernel1_partials_full(
        const float* __restrict__ x,
        __nv_bfloat162* __restrict__ partial_results,
        int N
    ) {
        extern __shared__ float shared_mem[];

        const int batch_idx = blockIdx.x;
        const int chunk_idx = blockIdx.y;
        const int num_chunks = gridDim.y;
        const int tid = threadIdx.x;
        const int block_size = blockDim.x;

        const float* x_batch = x + batch_idx * N;

        float sum = 0.0f;
        float sum_sq = 0.0f;

        const int N_vec = N / 4;
        const int elements_per_chunk_vec = (N_vec + num_chunks - 1) / num_chunks;
        const int chunk_start_vec = chunk_idx * elements_per_chunk_vec;
        const int chunk_end_vec = min(chunk_start_vec + elements_per_chunk_vec, N_vec);

        // Vectorized part
        for (int i = chunk_start_vec + tid; i < chunk_end_vec; i += block_size) {
            const float4 val4 = *reinterpret_cast<const float4*>(&x_batch[i * 4]);
            sum += val4.x + val4.y + val4.z + val4.w;
            sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
        }

        // Balanced remainder handling across all blocks in the row.
        const int remainder_start = N_vec * 4;
        const int threads_per_row = num_chunks * block_size;
        const int thread_in_row_idx = chunk_idx * block_size + tid;
        for (int i = remainder_start + thread_in_row_idx; i < N; i += threads_per_row) {
            float val = x_batch[i];
            sum += val;
            sum_sq += val * val;
        }

        Float2 partials = block_reduce_sum_two(sum, sum_sq, shared_mem);

        if (tid == 0) {
            int result_idx = batch_idx * num_chunks + chunk_idx;
            partial_results[result_idx] = __floats2bfloat162_rn(partials.x, partials.y);
        }
    }

    // Fused Kernel 2: REDUNDANTLY reduces partials in each block, then normalizes in parallel.
    __global__ void layernorm_fwd_kernel_reduce_and_normalize_hybrid(
        const float* __restrict__ x,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        const __nv_bfloat162* __restrict__ partial_results,
        float* __restrict__ y,
        int N,
        float epsilon,
        int num_chunks
    ) {
        extern __shared__ char shared_mem_char[];
        const int batch_idx = blockIdx.x;
        const int tid = threadIdx.x;
        const int block_size = blockDim.x;

        // --- Shared Memory Partitioning (Deinterleaved) ---
        __nv_bfloat16* smem_sums = (__nv_bfloat16*)shared_mem_char;
        __nv_bfloat16* smem_sum_sqs = smem_sums + num_chunks;
        float* reduce_smem = (float*)(smem_sum_sqs + num_chunks);

        // --- Redundant Reduction Phase ---
        const __nv_bfloat162* partials_batch = partial_results + batch_idx * num_chunks;

        // Vectorized and deinterleaved load of partials from global to shared memory
        const int num_chunks_vec = num_chunks / 2;
        for (int i = tid; i < num_chunks_vec; i += block_size) {
            const uint2 partials_vec = *reinterpret_cast<const uint2*>(&partials_batch[i*2]);
            const __nv_bfloat162 p1 = reinterpret_cast<const __nv_bfloat162*>(&partials_vec)[0];
            const __nv_bfloat162 p2 = reinterpret_cast<const __nv_bfloat162*>(&partials_vec)[1];

            smem_sums[i*2]   = p1.x;
            smem_sum_sqs[i*2] = p1.y;
            smem_sums[i*2+1] = p2.x;
            smem_sum_sqs[i*2+1] = p2.y;
        }
        __syncthreads();

        // Reduce from deinterleaved shared memory.
        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;
        for (int i = tid; i < num_chunks; i += block_size) {
            total_sum += __bfloat162float(smem_sums[i]);
            total_sum_sq += __bfloat162float(smem_sum_sqs[i]);
        }
        Float2 final_sums = block_reduce_sum_two(total_sum, total_sum_sq, reduce_smem);

        __shared__ float mean_val_s, rstd_val_s;
        if (tid == 0) {
            mean_val_s = final_sums.x / N;
            float var = final_sums.y / N - mean_val_s * mean_val_s;
            rstd_val_s = rsqrtf(var + epsilon);
        }
        __syncthreads();

        const float mean_val = mean_val_s;
        const float rstd_val = rstd_val_s;

        // --- Parallel Normalization Phase ---
        // Each block normalizes a unique slice of the row using a grid-stride loop.
        const float* x_batch = x + batch_idx * N;
        float* y_batch = y + batch_idx * N;
        const int N_vec = N / 4;
        const int threads_per_row = gridDim.y * blockDim.x;
        const int thread_in_row_idx = blockIdx.y * blockDim.x + tid;

        // Vectorized part using FMA optimization and __ldg for caching
        for (int i = thread_in_row_idx; i < N_vec; i += threads_per_row) {
            int idx = i * 4;
            const float4 x_val4 = __ldg(reinterpret_cast<const float4*>(&x_batch[idx]));
            const float4 w_val4 = __ldg(reinterpret_cast<const float4*>(&weight[idx]));
            const float4 b_val4 = __ldg(reinterpret_cast<const float4*>(&bias[idx]));

            float4 a, b, y_val4;
            a.x = rstd_val * w_val4.x;
            a.y = rstd_val * w_val4.y;
            a.z = rstd_val * w_val4.z;
            a.w = rstd_val * w_val4.w;
            b.x = b_val4.x - mean_val * a.x;
            b.y = b_val4.y - mean_val * a.y;
            b.z = b_val4.z - mean_val * a.z;
            b.w = b_val4.w - mean_val * a.w;
            y_val4.x = fmaf(x_val4.x, a.x, b.x);
            y_val4.y = fmaf(x_val4.y, a.y, b.y);
            y_val4.z = fmaf(x_val4.z, a.z, b.z);
            y_val4.w = fmaf(x_val4.w, a.w, b.w);

            *reinterpret_cast<float4*>(&y_batch[idx]) = y_val4;
        }

        // Remainder handling
        const int remainder_start = N_vec * 4;
        for (int i = remainder_start + thread_in_row_idx; i < N; i += threads_per_row) {
            const float x_val = __ldg(&x_batch[i]);
            const float w = __ldg(&weight[i]);
            const float b = __ldg(&bias[i]);
            const float a = rstd_val * w;
            const float B = b - mean_val * a;
            y_batch[i] = fmaf(x_val, a, B);
        }
    }

    void layer_norm_forward_cuda(
        const torch::Tensor& x,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        torch::Tensor& y,
        int64_t M,
        int64_t N,
        double eps
    ) {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

        const int BLOCKS_PER_ROW_PARTIALS = 128;

        auto partials_options = x.options().dtype(torch::kBFloat16);
        auto partial_results = torch::empty({M, BLOCKS_PER_ROW_PARTIALS, 2}, partials_options);

        // KERNEL 1: Compute partial statistics
        const int block_size_k1 = 512;
        const int reduce_smem_size_k1 = (block_size_k1 / 32) * 2 * sizeof(float);
        dim3 grid1(M, BLOCKS_PER_ROW_PARTIALS);
        layernorm_fwd_kernel1_partials_full<<<grid1, block_size_k1, reduce_smem_size_k1>>>(
            x.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat162*>(partial_results.data_ptr<at::BFloat16>()),
            N
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // KERNEL 2: Fused reduction and normalization
        const int block_size_k2 = 256;
        const int BLOCKS_PER_ROW_NORM = std::min(1024, (int)((N + block_size_k2 - 1) / block_size_k2));

        const int partials_smem_size_k2 = BLOCKS_PER_ROW_PARTIALS * sizeof(__nv_bfloat16) * 2;
        const int reduce_smem_size_k2 = (block_size_k2 / 32) * 2 * sizeof(float);
        const int shared_mem_size_k2 = partials_smem_size_k2 + reduce_smem_size_k2;

        dim3 grid2(M, BLOCKS_PER_ROW_NORM);
        layernorm_fwd_kernel_reduce_and_normalize_hybrid<<<grid2, block_size_k2, shared_mem_size_k2>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat162*>(partial_results.data_ptr<at::BFloat16>()),
            y.data_ptr<float>(),
            N,
            static_cast<float>(eps),
            BLOCKS_PER_ROW_PARTIALS
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    '''
    cpp_source = "void layer_norm_forward_cuda(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& y, int64_t M, int64_t N, double eps);"

    layernorm_lib = load_inline(
        name='layernorm_lib_2pass_hybrid',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['layer_norm_forward_cuda'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False
    )
    return layernorm_lib

layernorm_cuda_lib = load_layernorm_cuda_kernel()

class LayerNormCudaFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        x_shape = x.shape
        M = 1
        norm_dims = len(normalized_shape)
        for dim in x_shape[:-norm_dims]:
            M *= dim
        N = 1
        for dim in normalized_shape:
            N *= dim

        x_contiguous = x.contiguous()
        weight_contiguous = weight.contiguous()
        bias_contiguous = bias.contiguous()

        x_reshaped = x_contiguous.view(M, N)
        y = torch.empty_like(x_reshaped)

        layernorm_cuda_lib.layer_norm_forward_cuda(
            x_reshaped, weight_contiguous, bias_contiguous, y, M, N, eps
        )

        return y.view(x_shape)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass for LayerNormCudaFunc is not implemented.")

class ModelNew(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(ModelNew, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormCudaFunc.apply(x, self.normalized_shape, self.weight, self.bias, self.eps)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]