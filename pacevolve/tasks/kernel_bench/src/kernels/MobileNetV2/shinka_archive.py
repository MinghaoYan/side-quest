import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

def load_fused_kernels():
    cuda_source = r'''
#include <cuda_runtime.h>
#include <cmath>

// --- HELPER for robust SHUFFLE on DOUBLE using inline PTX ---
// This provides a more direct instruction to the compiler, potentially generating more efficient code.
__device__ __forceinline__ double shfl_down_sync_double(unsigned mask, double var, int delta) {
    unsigned int lo, hi;
    asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
    lo = __shfl_down_sync(mask, lo, delta);
    hi = __shfl_down_sync(mask, hi, delta);
    asm volatile("mov.b64 %0, {%1, %2};" : "=d"(var) : "r"(lo), "r"(hi));
    return var;
}


// --- KERNEL IMPLEMENTATION ---
// Combines ILP optimizations (4x unrolling) with a robust two-stage "leader-write" warp-shuffle reduction.
template<bool IsVec4, bool ApplyReLU6>
__global__ void fused_bn_stats_kernel_impl(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    const int N, const int C, const int H, const int W)
{
    extern __shared__ double s_data[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = block_size / 32;
    const int spatial_dim = H * W;
    const long long elements_per_channel = (long long)N * spatial_dim;

    // Grid-stride loop to process multiple channels per block, ensuring high GPU utilization
    for (int c = blockIdx.x; c < C; c += gridDim.x) {

        // --- Pass 1: Reduction for sum and sum of squares with 4x unrolling and pre-loading ---
        double p_sum;
        double p_sum_sq;

        if (IsVec4) {
            float p_sum_x = 0.0f, p_sum_y = 0.0f, p_sum_z = 0.0f, p_sum_w = 0.0f;
            float p_sum_sq_x = 0.0f, p_sum_sq_y = 0.0f, p_sum_sq_z = 0.0f, p_sum_sq_w = 0.0f;
            for (int n = 0; n < N; ++n) {
                const float* plane_start = input + n * C * spatial_dim + c * spatial_dim;
                const float4* plane_start_vec = (const float4*)plane_start;
                const int limit = spatial_dim / 4;
                for (int i = tid; i < limit; i += block_size * 4) {
                    int i2 = i + block_size;
                    int i3 = i + block_size * 2;
                    int i4 = i + block_size * 3;

                    float4 val1 = plane_start_vec[i];
                    float4 val2, val3, val4;
                    if (i2 < limit) val2 = plane_start_vec[i2];
                    if (i3 < limit) val3 = plane_start_vec[i3];
                    if (i4 < limit) val4 = plane_start_vec[i4];

                    p_sum_x += val1.x; p_sum_y += val1.y; p_sum_z += val1.z; p_sum_w += val1.w;
                    p_sum_sq_x += val1.x * val1.x; p_sum_sq_y += val1.y * val1.y; p_sum_sq_z += val1.z * val1.z; p_sum_sq_w += val1.w * val1.w;

                    if (i2 < limit) {
                        p_sum_x += val2.x; p_sum_y += val2.y; p_sum_z += val2.z; p_sum_w += val2.w;
                        p_sum_sq_x += val2.x * val2.x; p_sum_sq_y += val2.y * val2.y; p_sum_sq_z += val2.z * val2.z; p_sum_sq_w += val2.w * val2.w;
                    }
                    if (i3 < limit) {
                        p_sum_x += val3.x; p_sum_y += val3.y; p_sum_z += val3.z; p_sum_w += val3.w;
                        p_sum_sq_x += val3.x * val3.x; p_sum_sq_y += val3.y * val3.y; p_sum_sq_z += val3.z * val3.z; p_sum_sq_w += val3.w * val3.w;
                    }
                    if (i4 < limit) {
                        p_sum_x += val4.x; p_sum_y += val4.y; p_sum_z += val4.z; p_sum_w += val4.w;
                        p_sum_sq_x += val4.x * val4.x; p_sum_sq_y += val4.y * val4.y; p_sum_sq_z += val4.z * val4.z; p_sum_sq_w += val4.w * val4.w;
                    }
                }
            }
            p_sum = (double)p_sum_x + (double)p_sum_y + (double)p_sum_z + (double)p_sum_w;
            p_sum_sq = (double)p_sum_sq_x + (double)p_sum_sq_y + (double)p_sum_sq_z + (double)p_sum_sq_w;
        } else {
            float p_sum1 = 0.0f, p_sum2 = 0.0f, p_sum3 = 0.0f, p_sum4 = 0.0f;
            float p_sum_sq1 = 0.0f, p_sum_sq2 = 0.0f, p_sum_sq3 = 0.0f, p_sum_sq4 = 0.0f;
            for (int n = 0; n < N; ++n) {
                const float* plane_start = input + n * C * spatial_dim + c * spatial_dim;
                const int limit = spatial_dim;
                for (int i = tid; i < limit; i += block_size * 4) {
                    float val1 = plane_start[i];
                    p_sum1 += val1; p_sum_sq1 += val1 * val1;
                    int i2 = i + block_size;
                    if (i2 < limit) { float val2 = plane_start[i2]; p_sum2 += val2; p_sum_sq2 += val2 * val2; }
                    int i3 = i + block_size * 2;
                    if (i3 < limit) { float val3 = plane_start[i3]; p_sum3 += val3; p_sum_sq3 += val3 * val3; }
                    int i4 = i + block_size * 3;
                    if (i4 < limit) { float val4 = plane_start[i4]; p_sum4 += val4; p_sum_sq4 += val4 * val4; }
                }
            }
            p_sum = (double)p_sum1 + (double)p_sum2 + (double)p_sum3 + (double)p_sum4;
            p_sum_sq = (double)p_sum_sq1 + (double)p_sum_sq2 + (double)p_sum_sq3 + (double)p_sum_sq4;
        }

        // --- Intra-Warp reduction using robust helper ---
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            p_sum += shfl_down_sync_double(0xffffffff, p_sum, offset);
            p_sum_sq += shfl_down_sync_double(0xffffffff, p_sum_sq, offset);
        }

        // --- Inter-Warp reduction via Shared Memory (first warp performs final reduction) ---
        if (lane_id == 0) {
            s_data[warp_id] = p_sum;
            s_data[warp_id + num_warps] = p_sum_sq;
        }
        __syncthreads();

        if (warp_id == 0) {
            p_sum = (lane_id < num_warps) ? s_data[lane_id] : 0.0;
            p_sum_sq = (lane_id < num_warps) ? s_data[lane_id + num_warps] : 0.0;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                p_sum += shfl_down_sync_double(0xffffffff, p_sum, offset);
                p_sum_sq += shfl_down_sync_double(0xffffffff, p_sum_sq, offset);
            }
            if (lane_id == 0) {
                const double inv_elements_per_channel = 1.0 / elements_per_channel;
                double mean = p_sum * inv_elements_per_channel;
                double var = p_sum_sq * inv_elements_per_channel - mean * mean;
                float2* s_params = (float2*)s_data;
                float scale_val = weight[c] * rsqrtf((float)var + eps);
                s_params[0] = make_float2(scale_val, bias[c] - (float)mean * scale_val);
            }
        }
        __syncthreads();

        const float2 params = ((float2*)s_data)[0];
        const float scale = params.x;
        const float effective_bias = params.y;

        // --- Pass 2: Apply normalization and optional activation with 4x unrolling ---
        for (int n = 0; n < N; ++n) {
            const float* plane_in_start = input + n * C * spatial_dim + c * spatial_dim;
            float* plane_out_start = output + n * C * spatial_dim + c * spatial_dim;
            if (IsVec4) {
                const float4* plane_in_vec = (const float4*)plane_in_start;
                float4* plane_out_vec = (float4*)plane_out_start;
                const int limit = spatial_dim / 4;
                for (int i = tid; i < limit; i += block_size * 4) {
                    int i2 = i + block_size;
                    int i3 = i + block_size * 2;
                    int i4 = i + block_size * 3;

                    float4 val1 = plane_in_vec[i];
                    float4 val2, val3, val4;
                    if (i2 < limit) val2 = plane_in_vec[i2];
                    if (i3 < limit) val3 = plane_in_vec[i3];
                    if (i4 < limit) val4 = plane_in_vec[i4];
                    
                    float4 out1;
                    out1.x = val1.x * scale + effective_bias; out1.y = val1.y * scale + effective_bias;
                    out1.z = val1.z * scale + effective_bias; out1.w = val1.w * scale + effective_bias;
                    if (ApplyReLU6) {
                        out1.x = fminf(fmaxf(0.0f, out1.x), 6.0f); out1.y = fminf(fmaxf(0.0f, out1.y), 6.0f);
                        out1.z = fminf(fmaxf(0.0f, out1.z), 6.0f); out1.w = fminf(fmaxf(0.0f, out1.w), 6.0f);
                    }
                    plane_out_vec[i] = out1;
                    
                    if (i2 < limit) {
                        float4 out2;
                        out2.x = val2.x * scale + effective_bias; out2.y = val2.y * scale + effective_bias;
                        out2.z = val2.z * scale + effective_bias; out2.w = val2.w * scale + effective_bias;
                        if (ApplyReLU6) {
                            out2.x = fminf(fmaxf(0.0f, out2.x), 6.0f); out2.y = fminf(fmaxf(0.0f, out2.y), 6.0f);
                            out2.z = fminf(fmaxf(0.0f, out2.z), 6.0f); out2.w = fminf(fmaxf(0.0f, out2.w), 6.0f);
                        }
                        plane_out_vec[i2] = out2;
                    }
                    if (i3 < limit) {
                        float4 out3;
                        out3.x = val3.x * scale + effective_bias; out3.y = val3.y * scale + effective_bias;
                        out3.z = val3.z * scale + effective_bias; out3.w = val3.w * scale + effective_bias;
                        if (ApplyReLU6) {
                            out3.x = fminf(fmaxf(0.0f, out3.x), 6.0f); out3.y = fminf(fmaxf(0.0f, out3.y), 6.0f);
                            out3.z = fminf(fmaxf(0.0f, out3.z), 6.0f); out3.w = fminf(fmaxf(0.0f, out3.w), 6.0f);
                        }
                        plane_out_vec[i3] = out3;
                    }
                    if (i4 < limit) {
                        float4 out4;
                        out4.x = val4.x * scale + effective_bias; out4.y = val4.y * scale + effective_bias;
                        out4.z = val4.z * scale + effective_bias; out4.w = val4.w * scale + effective_bias;
                        if (ApplyReLU6) {
                            out4.x = fminf(fmaxf(0.0f, out4.x), 6.0f); out4.y = fminf(fmaxf(0.0f, out4.y), 6.0f);
                            out4.z = fminf(fmaxf(0.0f, out4.z), 6.0f); out4.w = fminf(fmaxf(0.0f, out4.w), 6.0f);
                        }
                        plane_out_vec[i4] = out4;
                    }
                }
            } else {
                for (int i = tid; i < spatial_dim; i += block_size * 4) {
                    float bn_out1 = plane_in_start[i] * scale + effective_bias;
                    if (ApplyReLU6) plane_out_start[i] = fminf(fmaxf(0.0f, bn_out1), 6.0f);
                    else plane_out_start[i] = bn_out1;
                    
                    int i2 = i + block_size;
                    if (i2 < spatial_dim) {
                        float bn_out2 = plane_in_start[i2] * scale + effective_bias;
                        if (ApplyReLU6) plane_out_start[i2] = fminf(fmaxf(0.0f, bn_out2), 6.0f);
                        else plane_out_start[i2] = bn_out2;
                    }
                    int i3 = i + block_size * 2;
                    if (i3 < spatial_dim) {
                        float bn_out3 = plane_in_start[i3] * scale + effective_bias;
                        if (ApplyReLU6) plane_out_start[i3] = fminf(fmaxf(0.0f, bn_out3), 6.0f);
                        else plane_out_start[i3] = bn_out3;
                    }
                    int i4 = i + block_size * 3;
                    if (i4 < spatial_dim) {
                        float bn_out4 = plane_in_start[i4] * scale + effective_bias;
                        if (ApplyReLU6) plane_out_start[i4] = fminf(fmaxf(0.0f, bn_out4), 6.0f);
                        else plane_out_start[i4] = bn_out4;
                    }
                }
            }
        }
    }
}

// --- KERNEL LAUNCHERS ---
void bn_relu6_stats_forward_cuda(
    const at::Tensor& input, at::Tensor& output,
    const at::Tensor& weight, const at::Tensor& bias, float eps)
{
    const int N = input.size(0); const int C = input.size(1);
    const int H = input.size(2); const int W = input.size(3);
    const int spatial_dim = H * W;

    const int target_blocks = 1024;
    const int num_blocks = (C < target_blocks) ? C : target_blocks;
    if (spatial_dim % 4 == 0) {
        const int threads = 256;
        const int num_warps = threads / 32;
        const int smem = 2 * num_warps * sizeof(double);
        fused_bn_stats_kernel_impl<true, true><<<num_blocks, threads, smem>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
            bias.data_ptr<float>(), eps, N, C, H, W);
    } else {
        const int threads = 512;
        const int num_warps = threads / 32;
        const int smem = 2 * num_warps * sizeof(double);
        fused_bn_stats_kernel_impl<false, true><<<num_blocks, threads, smem>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
            bias.data_ptr<float>(), eps, N, C, H, W);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err)); }
}

void bn_stats_forward_cuda(
    const at::Tensor& input, at::Tensor& output,
    const at::Tensor& weight, const at::Tensor& bias, float eps)
{
    const int N = input.size(0); const int C = input.size(1);
    const int H = input.size(2); const int W = input.size(3);
    const int spatial_dim = H * W;

    const int target_blocks = 1024;
    const int num_blocks = (C < target_blocks) ? C : target_blocks;
    if (spatial_dim % 4 == 0) {
        const int threads = 256;
        const int num_warps = threads / 32;
        const int smem = 2 * num_warps * sizeof(double);
        fused_bn_stats_kernel_impl<true, false><<<num_blocks, threads, smem>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
            bias.data_ptr<float>(), eps, N, C, H, W);
    } else {
        const int threads = 512;
        const int num_warps = threads / 32;
        const int smem = 2 * num_warps * sizeof(double);
        fused_bn_stats_kernel_impl<false, false><<<num_blocks, threads, smem>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
            bias.data_ptr<float>(), eps, N, C, H, W);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err)); }
}
    '''

    cpp_source = r'''
#include <torch/extension.h>

void bn_relu6_stats_forward_cuda(const at::Tensor&, at::Tensor&, const at::Tensor&, const at::Tensor&, float);
void bn_stats_forward_cuda(const at::Tensor&, at::Tensor&, const at::Tensor&, const at::Tensor&, float);

at::Tensor bn_relu6_stats_forward(
    const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& bias, float eps)
{
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == at::kFloat, "Input must be a contiguous CUDA float tensor");
    auto output = at::empty_like(input);
    bn_relu6_stats_forward_cuda(input, output, weight, bias, eps);
    return output;
}

at::Tensor bn_stats_forward(
    const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& bias, float eps)
{
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == at::kFloat, "Input must be a contiguous CUDA float tensor");
    auto output = at::empty_like(input);
    bn_stats_forward_cuda(input, output, weight, bias, eps);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bn_relu6_stats", &bn_relu6_stats_forward, "Fused BatchNorm + ReLU6 with in-kernel stats (CUDA)");
    m.def("bn_stats", &bn_stats_forward, "Fused BatchNorm with in-kernel stats (CUDA)");
}
    '''

    extra_cuda_cflags = ['-O3', '--use_fast_math']
    build_directory = os.environ.get('TORCH_EXTENSIONS_DIR')

    fused_kernels = load_inline(
        name='fused_kernels_hybrid_best',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        build_directory=build_directory
    )
    return fused_kernels

fused_kernels = load_fused_kernels()

class FusedConvBNReLU6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return fused_kernels.bn_relu6_stats(
            x, self.bn.weight, self.bn.bias, self.bn.eps
        )

class FusedConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return fused_kernels.bn_stats(
            x, self.bn.weight, self.bn.bias, self.bn.eps
        )

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [FusedConvBNReLU6(3, input_channel, 3, 2, 1, bias=False)]

        # Building inverted residual blocks directly
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                hidden_dim = int(input_channel * t)
                if t != 1:
                    # Point-wise expansion
                    features.append(FusedConvBNReLU6(input_channel, hidden_dim, 1, 1, 0, bias=False))

                # Depth-wise convolution followed by point-wise linear convolution
                features.extend([
                    FusedConvBNReLU6(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    FusedConvBN(hidden_dim, output_channel, 1, 1, 0, bias=False),
                ])
                input_channel = output_channel

        # Building last several layers
        features.append(FusedConvBNReLU6(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x