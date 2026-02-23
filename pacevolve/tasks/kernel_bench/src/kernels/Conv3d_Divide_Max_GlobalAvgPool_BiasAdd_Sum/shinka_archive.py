import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    '''
    Model that performs a 3D convolution, then a fused operation for the rest of the pipeline:
    (division, max pooling, global average pooling, bias addition).
    The final sum is performed in PyTorch.
    This version uses C++ templates to specialize the CUDA kernel for common pooling widths
    (e.g., 2, 4, 8), which allows the compiler to eliminate branching inside the
    performance-critical inner pooling loop.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divisor = divisor
        # Ensure pool_size is a tuple/list of 3 integers
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size, pool_size)
        self.pool_size_d, self.pool_size_h, self.pool_size_w = pool_size
        self.sum_dim = sum_dim

        self._compile_and_load_kernel()

    def _compile_and_load_kernel(self):
        cuda_source = r'''
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ inline float hmaxf4(const float4& v) {
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

__device__ inline float4 warp_reduce_sum4(float4 val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, offset);
        val.w += __shfl_down_sync(0xFFFFFFFF, val.w, offset);
    }
    return val;
}

// Each block processes 16 channels in parallel using four float4 vectors to maximize ILP.
#define CHANNELS_PER_BLOCK 16

template<int PD_T, int PH_T, int PW_T>
__global__ void fused_post_conv_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float divisor,
    const int N, const int C, const int D, const int H, const int W,
    const int PD_runtime, const int PH_runtime, const int PW_runtime)
{
    // Use compile-time constant if available (> 0), otherwise use runtime value
    const int PD = (PD_T == 0) ? PD_runtime : PD_T;
    const int PH = (PH_T == 0) ? PH_runtime : PH_T;
    const int PW = (PW_T == 0) ? PW_runtime : PW_T;

    // Shared memory for inter-warp reduction. Stores four float4s per warp.
    extern __shared__ float s_data[];
    float4* s_data_vec = (float4*)s_data;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = block_size / 32;

    const int OD = D / PD;
    const int OH = H / PH;
    const int OW = W / PW;
    const int mp_size = OD * OH * OW;
    const long long channel_vol = (long long)D * H * W;
    const long long batch_channel_vol = (long long)C * channel_vol;
    const int num_c_groups = (C + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;

    // Grid-stride loop for dynamic load balancing
    for (int work_idx = blockIdx.x; work_idx < N * num_c_groups; work_idx += gridDim.x) {
        const int n = work_idx / num_c_groups;
        const int c_group = work_idx % num_c_groups;
        const int c_base = c_group * CHANNELS_PER_BLOCK;

        if (mp_size == 0) {
            if (tid == 0) {
                #pragma unroll
                for (int i = 0; i < CHANNELS_PER_BLOCK; ++i) {
                    if (c_base + i < C) {
                        output[(long long)n * C + c_base + i] = bias[c_base + i];
                    }
                }
            }
            continue;
        }

        float4 thread_sum[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            thread_sum[j] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        const long long base_offset = (long long)n * batch_channel_vol + (long long)c_base * channel_vol;

        const float* in_c[CHANNELS_PER_BLOCK];
        #pragma unroll
        for (int i=0; i<CHANNELS_PER_BLOCK; ++i) {
            in_c[i] = conv_out + base_offset + (long long)i * channel_vol;
        }

        if (c_base + CHANNELS_PER_BLOCK - 1 < C) {
            // --- FAST PATH (no boundary checks) ---
            for (int i = tid; i < mp_size; i += block_size) {
                int w_out = i % OW;
                int h_out = (i / OW) % OH;
                int d_out = i / (OW * OH);

                int d_start = d_out * PD;
                int h_start = h_out * PH;
                int w_start = w_out * PW;

                float4 max_val[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    max_val[j] = {-1.0e38f, -1.0e38f, -1.0e38f, -1.0e38f};
                }

                #pragma unroll
                for (int pd = 0; pd < PD; ++pd) {
                    const float* row_ptrs[CHANNELS_PER_BLOCK];
                    #pragma unroll
                    for (int k=0; k<CHANNELS_PER_BLOCK; ++k) {
                        row_ptrs[k] = in_c[k] + ((long long)d_start + pd) * H * W + (long long)h_start * W;
                    }

                    #pragma unroll
                    for (int ph = 0; ph < PH; ++ph) {
                        int pw = 0;
                        #pragma unroll
                        for (; pw <= PW - 4; pw += 4) {
                            const long long base_idx = w_start + pw;
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                max_val[j].x = fmaxf(max_val[j].x, hmaxf4(*reinterpret_cast<const float4*>(row_ptrs[j*4+0] + base_idx)));
                                max_val[j].y = fmaxf(max_val[j].y, hmaxf4(*reinterpret_cast<const float4*>(row_ptrs[j*4+1] + base_idx)));
                                max_val[j].z = fmaxf(max_val[j].z, hmaxf4(*reinterpret_cast<const float4*>(row_ptrs[j*4+2] + base_idx)));
                                max_val[j].w = fmaxf(max_val[j].w, hmaxf4(*reinterpret_cast<const float4*>(row_ptrs[j*4+3] + base_idx)));
                            }
                        }
                        #pragma unroll
                        for (; pw <= PW - 2; pw += 2) {
                            const long long base_idx = w_start + pw;
                             #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const float2* d0 = reinterpret_cast<const float2*>(row_ptrs[j*4+0] + base_idx); max_val[j].x = fmaxf(max_val[j].x, fmaxf(d0->x, d0->y));
                                const float2* d1 = reinterpret_cast<const float2*>(row_ptrs[j*4+1] + base_idx); max_val[j].y = fmaxf(max_val[j].y, fmaxf(d1->x, d1->y));
                                const float2* d2 = reinterpret_cast<const float2*>(row_ptrs[j*4+2] + base_idx); max_val[j].z = fmaxf(max_val[j].z, fmaxf(d2->x, d2->y));
                                const float2* d3 = reinterpret_cast<const float2*>(row_ptrs[j*4+3] + base_idx); max_val[j].w = fmaxf(max_val[j].w, fmaxf(d3->x, d3->y));
                            }
                        }
                        #pragma unroll
                        for (; pw < PW; ++pw) {
                            long long in_idx = w_start + pw;
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                max_val[j].x = fmaxf(max_val[j].x, row_ptrs[j*4+0][in_idx]);
                                max_val[j].y = fmaxf(max_val[j].y, row_ptrs[j*4+1][in_idx]);
                                max_val[j].z = fmaxf(max_val[j].z, row_ptrs[j*4+2][in_idx]);
                                max_val[j].w = fmaxf(max_val[j].w, row_ptrs[j*4+3][in_idx]);
                            }
                        }

                        #pragma unroll
                        for(int k=0; k<CHANNELS_PER_BLOCK; ++k) row_ptrs[k] += W;
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    thread_sum[j].x += max_val[j].x; thread_sum[j].y += max_val[j].y; thread_sum[j].z += max_val[j].z; thread_sum[j].w += max_val[j].w;
                }
            }
        } else {
            // --- SLOW PATH (optimized with hoisted boundary check) ---
            const int C_rem = C - c_base;
            for (int i = tid; i < mp_size; i += block_size) {
                int w_out = i % OW;
                int h_out = (i / OW) % OH;
                int d_out = i / (OW * OH);
                int d_start = d_out * PD;
                int h_start = h_out * PH;
                int w_start = w_out * PW;

                float4 max_val[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) max_val[j] = {-1.0e38f, -1.0e38f, -1.0e38f, -1.0e38f};

                #pragma unroll
                for (int pd = 0; pd < PD; ++pd) {
                    const float* row_ptrs[CHANNELS_PER_BLOCK];
                    #pragma unroll
                    for (int k=0; k<CHANNELS_PER_BLOCK; ++k) {
                        row_ptrs[k] = in_c[k] + ((long long)d_start + pd) * H * W + (long long)h_start * W;
                    }

                    #pragma unroll
                    for (int ph = 0; ph < PH; ++ph) {
                        #pragma unroll
                        for (int pw = 0; pw < PW; ++pw) {
                            const long long in_idx = w_start + pw;
                            #pragma unroll
                            for (int j = 0; j < 4; j++) {
                                const int c_offset = j * 4;
                                if (C_rem > c_offset + 3) {
                                    max_val[j].x = fmaxf(max_val[j].x, row_ptrs[c_offset + 0][in_idx]);
                                    max_val[j].y = fmaxf(max_val[j].y, row_ptrs[c_offset + 1][in_idx]);
                                    max_val[j].z = fmaxf(max_val[j].z, row_ptrs[c_offset + 2][in_idx]);
                                    max_val[j].w = fmaxf(max_val[j].w, row_ptrs[c_offset + 3][in_idx]);
                                } else {
                                    if (C_rem > c_offset + 0) max_val[j].x = fmaxf(max_val[j].x, row_ptrs[c_offset + 0][in_idx]);
                                    if (C_rem > c_offset + 1) max_val[j].y = fmaxf(max_val[j].y, row_ptrs[c_offset + 1][in_idx]);
                                    if (C_rem > c_offset + 2) max_val[j].z = fmaxf(max_val[j].z, row_ptrs[c_offset + 2][in_idx]);
                                }
                            }
                        }
                        #pragma unroll
                        for(int k=0; k<CHANNELS_PER_BLOCK; ++k) row_ptrs[k] += W;
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    thread_sum[j].x += max_val[j].x; thread_sum[j].y += max_val[j].y; thread_sum[j].z += max_val[j].z; thread_sum[j].w += max_val[j].w;
                }
            }
        }

        #pragma unroll
        for(int j=0; j<4; ++j) thread_sum[j] = warp_reduce_sum4(thread_sum[j]);

        if (lane_id == 0) {
            #pragma unroll
            for(int j=0; j<4; ++j) s_data_vec[warp_id * 4 + j] = thread_sum[j];
        }
        __syncthreads();

        // --- Fully Parallel Inter-Warp Reduction ---
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            if (tid < offset * 4) {
                int read_idx = tid + offset * 4;
                s_data_vec[tid].x += s_data_vec[read_idx].x;
                s_data_vec[tid].y += s_data_vec[read_idx].y;
                s_data_vec[tid].z += s_data_vec[read_idx].z;
                s_data_vec[tid].w += s_data_vec[read_idx].w;
            }
            __syncthreads();
        }

        // --- Parallelized Final Write ---
        if (tid < CHANNELS_PER_BLOCK) {
            if (c_base + tid < C) {
                const float inv_mp_divisor = 1.0f / ((float)mp_size * divisor);
                const long long out_idx = (long long)n * C + c_base + tid;

                const int vec_idx = tid / 4;
                const int comp_idx = tid % 4;

                union { float4 vec; float arr[4]; } sum_converter;
                sum_converter.vec = s_data_vec[vec_idx];

                output[out_idx] = sum_converter.arr[comp_idx] * inv_mp_divisor + bias[c_base + tid];
            }
        }
        __syncthreads();
    }
}


void fused_post_conv_kernel_launcher(
    const torch::Tensor& conv_out,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const float divisor,
    const int N, const int C, const int D, const int H, const int W,
    const int PD, const int PH, const int PW)
{
    const int block_size = 256;
    dim3 blockDim(block_size);

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, conv_out.device().index());
    const int num_blocks = num_sms * 8;
    dim3 gridDim(num_blocks);

    const int warps_per_block = block_size / 32;
    size_t shared_mem_size = warps_per_block * 4 * sizeof(float4);

    // Dispatch to a templated kernel specialized for common pooling geometries.
    if (PD == 2 && PH == 2 && PW == 2) {
        fused_post_conv_kernel<2, 2, 2><<<gridDim, blockDim, shared_mem_size>>>(
            conv_out.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            divisor, N, C, D, H, W, PD, PH, PW);
    } else if (PW == 2) {
        fused_post_conv_kernel<0, 0, 2><<<gridDim, blockDim, shared_mem_size>>>(
            conv_out.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            divisor, N, C, D, H, W, PD, PH, PW);
    } else if (PW == 4) {
        fused_post_conv_kernel<0, 0, 4><<<gridDim, blockDim, shared_mem_size>>>(
            conv_out.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            divisor, N, C, D, H, W, PD, PH, PW);
    } else if (PW == 8) {
        fused_post_conv_kernel<0, 0, 8><<<gridDim, blockDim, shared_mem_size>>>(
            conv_out.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            divisor, N, C, D, H, W, PD, PH, PW);
    } else {
        fused_post_conv_kernel<0, 0, 0><<<gridDim, blockDim, shared_mem_size>>>(
            conv_out.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
            divisor, N, C, D, H, W, PD, PH, PW);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
'''

        cpp_source = r'''
#include <torch/extension.h>

void fused_post_conv_kernel_launcher(
    const torch::Tensor& conv_out, const torch::Tensor& bias, torch::Tensor& output,
    const float divisor,
    const int N, const int C, const int D, const int H, const int W,
    const int PD, const int PH, const int PW);

torch::Tensor forward_cpp(
    const torch::Tensor& conv_out, const torch::Tensor& bias, const float divisor,
    const int PD, const int PH, const int PW)
{
    TORCH_CHECK(conv_out.is_cuda(), "conv_out must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(conv_out.is_contiguous(), "conv_out must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(conv_out.dim() == 5, "conv_out must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");

    const int N = conv_out.size(0);
    const int C = conv_out.size(1);
    const int D = conv_out.size(2);
    const int H = conv_out.size(3);
    const int W = conv_out.size(4);

    TORCH_CHECK(bias.size(0) == C, "bias size must match channel dimension");
    TORCH_CHECK(D % PD == 0 && H % PH == 0 && W % PW == 0, "Input dimensions must be divisible by pool size for this optimized kernel");

    auto options = torch::TensorOptions().device(conv_out.device()).dtype(conv_out.dtype());
    auto output = torch::empty({N, C}, options);

    fused_post_conv_kernel_launcher(conv_out, bias, output, divisor, N, C, D, H, W, PD, PH, PW);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cpp, "Fused Post-Convolution Forward (CUDA)");
}
'''
        # JIT compilation of the CUDA/C++ extension
        self.fused_op = load_inline(
            name='fused_full_3d_template',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-arch=sm_80', '-O3', '--use_fast_math'],
            extra_cflags=['-O3'],
            verbose=False,
        )

    def forward(self, x):
        conv_out = self.conv(x)

        # Bias is expected to be per-channel, e.g. (1, C, 1, 1, 1).
        # We flatten it to 1D for easy access in the CUDA kernel.
        bias_flat = self.bias.view(-1)

        # Call the custom fused CUDA kernel
        intermediate = self.fused_op.forward(
            conv_out,
            bias_flat,
            self.divisor,
            self.pool_size_d,
            self.pool_size_h,
            self.pool_size_w
        )

        # Reshape the (N, C) output to (N, C, 1, 1, 1) to match the
        # original model's tensor shape before the final sum operation.
        N, C = intermediate.shape
        x = intermediate.view(N, C, 1, 1, 1)

        # The final sum is performed by PyTorch.
        x = torch.sum(x, dim=self.sum_dim)
        return x