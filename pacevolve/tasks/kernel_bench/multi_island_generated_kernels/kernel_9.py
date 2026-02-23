# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
import math

# Set CUDA architecture for A100 to ensure compatibility and performance.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

hybrid_rms_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- NCHW Kernel Constants ---
constexpr int BLOCK_H = 32;
constexpr int BLOCK_W = 32;
constexpr int VEC_SIZE = 4; // Using float4
constexpr int TILE_W = BLOCK_W * VEC_SIZE; // 128
constexpr int REDUCE_PARTITION_SIZE = 8; // For hierarchical reduction

// Utility for using float4
using float4 = float4;


/**
 * @brief NCHW Kernel with 2x8 W-Tiling.
 *
 * This kernel extends the state-of-the-art 2x4 tiling strategy to a 2x8
 * configuration. Each thread block now processes a grid of 2x8 tiles: two
 * adjacent H-rows and eight adjacent W-tiles. By doubling the work-per-block
 * along the spatial W-dimension, this experiment aims to determine if further
 * reductions in kernel launch overhead can yield performance gains, or if
 * performance is strictly limited by other factors like register pressure at
 * this scale. This pushes the "more work per block" optimization philosophy
 * to its next logical step.
 */
__global__ void rms_norm_nchw_2x8_w_tiling_kernel(const float* __restrict__ x, float* __restrict__ out,
                                                 const int N, const int C, const int H, const int W,
                                                 const float eps, const float inv_c) {
    const int h_pair_idx = blockIdx.y;

    const int tx = threadIdx.x; // 0-31; corresponds to spatial W dimension
    const int ty = threadIdx.y; // 0-31; corresponds to channel C dimension

    const int C_H_W = C * H * W;
    const int H_W = H * W;
    const int UNROLL_STRIDE = BLOCK_H * 2;

    __shared__ float s_tile[BLOCK_H][TILE_W];

    // --- W-Tiling Loop ---
    // Process EIGHT adjacent 128-element tiles along the W dimension.
    #pragma unroll
    for (int w_iter = 0; w_iter < 8; ++w_iter) {
        const int w_idx_base = blockIdx.x * (TILE_W * 8) + w_iter * TILE_W;
        if (w_idx_base >= W) continue;

        const int w_offset_base = w_idx_base + tx * VEC_SIZE;

        // --- H-Tiling Loop ---
        // Process two adjacent H-rows.
        #pragma unroll
        for (int h_iter = 0; h_iter < 2; ++h_iter) {
            const int global_h_idx = h_pair_idx * 2 + h_iter;
            if (global_h_idx >= N * H) continue;

            const int n_idx = global_h_idx / H;
            const int h_idx = global_h_idx % H;

            // --- PASS 1: Reduction ---
            float sum_sq[VEC_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int w_offset = w_offset_base;

            for (int c_base = ty; c_base < C; c_base += UNROLL_STRIDE) {
                if (w_offset + VEC_SIZE - 1 < W) {
                    float4 val_vec_1 = *reinterpret_cast<const float4*>(&x[n_idx*C_H_W + c_base*H_W + h_idx*W + w_offset]);
                    sum_sq[0] += val_vec_1.x * val_vec_1.x;
                    sum_sq[1] += val_vec_1.y * val_vec_1.y;
                    sum_sq[2] += val_vec_1.z * val_vec_1.z;
                    sum_sq[3] += val_vec_1.w * val_vec_1.w;

                    const int c_base_2 = c_base + BLOCK_H;
                    if (c_base_2 < C) {
                        float4 val_vec_2 = *reinterpret_cast<const float4*>(&x[n_idx*C_H_W + c_base_2*H_W + h_idx*W + w_offset]);
                        sum_sq[0] += val_vec_2.x * val_vec_2.x;
                        sum_sq[1] += val_vec_2.y * val_vec_2.y;
                        sum_sq[2] += val_vec_2.z * val_vec_2.z;
                        sum_sq[3] += val_vec_2.w * val_vec_2.w;
                    }
                } else { // Handle ragged edge of W
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        if (w_offset + i < W) {
                            float val1 = x[n_idx*C_H_W + c_base*H_W + h_idx*W + w_offset + i];
                            sum_sq[i] += val1 * val1;
                            const int c_base_2 = c_base + BLOCK_H;
                            if (c_base_2 < C) {
                                float val2 = x[n_idx*C_H_W + c_base_2*H_W + h_idx*W + w_offset + i];
                                sum_sq[i] += val2 * val2;
                            }
                        }
                    }
                }
            }

            // --- Hierarchical Reduction in Shared Memory ---
            *reinterpret_cast<float4*>(&s_tile[ty][tx * VEC_SIZE]) = *reinterpret_cast<float4*>(sum_sq);
            __syncthreads();

            if ((ty % REDUCE_PARTITION_SIZE) == 0 && ty < BLOCK_H) {
                float4* my_sum_vec = reinterpret_cast<float4*>(&s_tile[ty][tx * VEC_SIZE]);
                #pragma unroll
                for (int i = 1; i < REDUCE_PARTITION_SIZE; ++i) {
                    const float4* other_row = reinterpret_cast<const float4*>(&s_tile[ty + i][tx * VEC_SIZE]);
                    my_sum_vec->x += other_row->x; my_sum_vec->y += other_row->y;
                    my_sum_vec->z += other_row->z; my_sum_vec->w += other_row->w;
                }
            }
            __syncthreads();

            if (ty == 0) {
                float4* final_sum_vec = reinterpret_cast<float4*>(&s_tile[0][tx * VEC_SIZE]);
                #pragma unroll
                for (int i = 1; i < BLOCK_H / REDUCE_PARTITION_SIZE; ++i) {
                    const float4* partition_sum = reinterpret_cast<const float4*>(&s_tile[i * REDUCE_PARTITION_SIZE][tx * VEC_SIZE]);
                    final_sum_vec->x += partition_sum->x; final_sum_vec->y += partition_sum->y;
                    final_sum_vec->z += partition_sum->z; final_sum_vec->w += partition_sum->w;
                }
                final_sum_vec->x = rsqrtf(final_sum_vec->x * inv_c + eps);
                final_sum_vec->y = rsqrtf(final_sum_vec->y * inv_c + eps);
                final_sum_vec->z = rsqrtf(final_sum_vec->z * inv_c + eps);
                final_sum_vec->w = rsqrtf(final_sum_vec->w * inv_c + eps);
            }
            __syncthreads();

            const float4 inv_rms_vec = *reinterpret_cast<const float4*>(&s_tile[0][tx * VEC_SIZE]);

            // --- PASS 2: Scaling ---
            for (int c_idx = ty; c_idx < C; c_idx += UNROLL_STRIDE) {
                if (w_offset + VEC_SIZE - 1 < W) {
                    float4 val_vec1 = *reinterpret_cast<const float4*>(&x[n_idx*C_H_W + c_idx*H_W + h_idx*W + w_offset]);
                    val_vec1.x *= inv_rms_vec.x; val_vec1.y *= inv_rms_vec.y; val_vec1.z *= inv_rms_vec.z; val_vec1.w *= inv_rms_vec.w;
                    *reinterpret_cast<float4*>(&out[n_idx*C_H_W + c_idx*H_W + h_idx*W + w_offset]) = val_vec1;
                    const int c_idx_2 = c_idx + BLOCK_H;
                    if (c_idx_2 < C) {
                        float4 val_vec2 = *reinterpret_cast<const float4*>(&x[n_idx*C_H_W + c_idx_2*H_W + h_idx*W + w_offset]);
                        val_vec2.x *= inv_rms_vec.x; val_vec2.y *= inv_rms_vec.y; val_vec2.z *= inv_rms_vec.z; val_vec2.w *= inv_rms_vec.w;
                        *reinterpret_cast<float4*>(&out[n_idx*C_H_W + c_idx_2*H_W + h_idx*W + w_offset]) = val_vec2;
                    }
                } else {
                    float inv_rms_arr[VEC_SIZE] = {inv_rms_vec.x, inv_rms_vec.y, inv_rms_vec.z, inv_rms_vec.w};
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        if (w_offset + i < W) {
                            out[n_idx*C_H_W + c_idx*H_W + h_idx*W + w_offset + i] = x[n_idx*C_H_W + c_idx*H_W + h_idx*W + w_offset + i] * inv_rms_arr[i];
                            const int c_idx_2 = c_idx + BLOCK_H;
                            if (c_idx_2 < C) {
                               out[n_idx*C_H_W + c_idx_2*H_W + h_idx*W + w_offset + i] = x[n_idx*C_H_W + c_idx_2*H_W + h_idx*W + w_offset + i] * inv_rms_arr[i];
                            }
                        }
                    }
                }
            }
        }
    }
}


/**
 * @brief NHWC (channels-last) Kernel for RMS Normalization.
 * (Unchanged)
 */
__global__ void rms_norm_nhwc_kernel(const float* __restrict__ x, float* __restrict__ out,
                                     const int num_vectors, const int C, const float eps, const float inv_c) {
    const int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (warp_id >= num_vectors) {
        return;
    }

    const int lane_id = threadIdx.x % 32;
    const float* x_vec = x + warp_id * C;
    float* out_vec = out + warp_id * C;

    float sum_sq = 0.0f;
    for (int i = lane_id * VEC_SIZE; i < C; i += 32 * VEC_SIZE) {
        if (i + VEC_SIZE <= C) {
            float4 val = *reinterpret_cast<const float4*>(&x_vec[i]);
            sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
        } else {
            for (int j = i; j < C; ++j) {
                float val = x_vec[j];
                sum_sq += val * val;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    const float inv_rms = rsqrtf(sum_sq * inv_c + eps);

    for (int i = lane_id * VEC_SIZE; i < C; i += 32 * VEC_SIZE) {
         if (i + VEC_SIZE <= C) {
            float4 val = *reinterpret_cast<const float4*>(&x_vec[i]);
            val.x *= inv_rms; val.y *= inv_rms; val.z *= inv_rms; val.w *= inv_rms;
            *reinterpret_cast<float4*>(&out_vec[i]) = val;
         } else {
            for (int j = i; j < C; ++j) {
                out_vec[j] = x_vec[j] * inv_rms;
            }
        }
    }
}

torch::Tensor rms_norm_hybrid_cuda(torch::Tensor x, float eps) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input tensor must be of type float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D");

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    if (N*C*H*W == 0) return out;
    const float inv_c = 1.0f / C;

    cudaError_t err;

    if (x.is_contiguous(at::MemoryFormat::Contiguous)) { // NCHW format
        const dim3 block_dim(BLOCK_W, BLOCK_H);
        // Grid x-dim is halved again due to extended W-Tiling (8 tiles per block).
        const dim3 grid_dim( (W + (TILE_W * 8) - 1) / (TILE_W * 8), (N * H + 1) / 2 );

        rms_norm_nchw_2x8_w_tiling_kernel<<<grid_dim, block_dim>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, eps, inv_c
        );
    } else if (x.is_contiguous(at::MemoryFormat::ChannelsLast)) { // NHWC format
        const int num_vectors = N * H * W;
        const int threads_per_block = 256;
        const int warps_per_block = threads_per_block / 32;
        const int blocks = (num_vectors + warps_per_block - 1) / warps_per_block;

        rms_norm_nhwc_kernel<<<blocks, threads_per_block>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), num_vectors, C, eps, inv_c
        );
    } else {
        AT_ERROR("Unsupported memory format. Input must be contiguous (NCHW) or channels_last (NHWC).");
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

hybrid_rms_norm_cpp_source = """
#include <torch/extension.h>
torch::Tensor rms_norm_hybrid_cuda(torch::Tensor x, float eps);
"""

# JIT compile the CUDA extension
hybrid_rms_norm_cached = load_inline(
    name='hybrid_rms_norm_v13_2x8_tiling',
    cpp_sources=hybrid_rms_norm_cpp_source,
    cuda_sources=hybrid_rms_norm_cuda_source,
    functions=['rms_norm_hybrid_cuda'],
    verbose=True,
    extra_cuda_cflags=['-std=c++17', '-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
)

class ModelNew(nn.Module):
    """
    An optimized RMS Normalization model using a 2x8 HW-Tiling strategy.

    This model tests the limits of the "more work per block" optimization
    philosophy by extending the state-of-the-art 2x4 tiling strategy to a 2x8
    configuration. Each thread block processes a grid of 2x8 tiles (two adjacent
    H-rows, eight adjacent W-tiles), further reducing the number of thread blocks
    in the grid and minimizing kernel launch overhead. This design serves to
    evaluate whether performance gains from this strategy have saturated or if
    further improvements are possible at the risk of increased register pressure.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm_hybrid = hybrid_rms_norm_cached.rms_norm_hybrid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMS Normalization. Dynamically dispatches to a specialized
        CUDA kernel based on the input tensor's memory layout (NCHW vs. NHWC).

        Args:
            x (torch.Tensor): A 4D input tensor of shape (B, C, H, W) and dtype
                              float32. It must be on a CUDA device and be in
                              either NCHW (contiguous) or NHWC (channels-last)
                              memory format.

        Returns:
            torch::Tensor: A normalized output tensor of the same shape, dtype, and
                           memory format as the input.
        """
        is_supported = (
            x.is_cuda and
            x.dtype == torch.float32 and
            x.dim() == 4 and
            (x.is_contiguous(memory_format=torch.contiguous_format) or
             x.is_contiguous(memory_format=torch.channels_last))
        )

        if not is_supported:
            # Fallback for unsupported inputs
            original_dtype = x.dtype
            original_format = torch.channels_last if x.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format

            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=1, keepdim=True)
            output = x_float * torch.rsqrt(variance + self.eps)

            return output.to(original_dtype).contiguous(memory_format=original_format)

        return self.rms_norm_hybrid(x, self.eps)