import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.nn.modules.utils import _pair
import os

class ModelNew(nn.Module):
    '''
    Optimized Max Pooling 2D with a "Penta-Path ILP-Optimized Direct-to-Register" (D2R) CUDA kernel.
    This kernel expands on the hyper-specialization strategy by adding a new fast path for 2x2 stride 1 kernels.
    It further maximizes instruction-level parallelism (ILP) by re-associating reduction operations into balanced trees
    and ensures correctness by replacing the generic fallback path with a UB-safe implementation.
    '''
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        '''
        Initializes the custom Max Pooling 2D layer.

        Args:
            kernel_size (int or tuple): Size of the pooling window.
            stride (int or tuple): Stride of the pooling window.
            padding (int or tuple): Padding to be applied before pooling.
            dilation (int or tuple): Spacing between kernel elements.
        '''
        super(ModelNew, self).__init__()
        self.kernel_size = _pair(kernel_size)
        # Emulate nn.MaxPool2d behavior: if stride is not provided, it defaults to kernel_size
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.cuda_module = None

    def _load_cuda_module(self):
        if self.cuda_module:
            return

        cpp_source = '''
        #include <torch/extension.h>
        #include <vector>

        void maxpool2d_forward_launcher(
            const float* input,
            float* output,
            int N, int C, int H, int W,
            int H_out, int W_out,
            int kH, int kW, int sH, int sW,
            int pH, int pW, int dH, int dW);

        void maxpool2d_forward_cuda(
            torch::Tensor input,
            torch::Tensor output,
            int H, int W, int H_out, int W_out,
            int kH, int kW, int sH, int sW,
            int pH, int pW, int dH, int dW)
        {
            TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
            TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
            TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
            TORCH_CHECK(output.scalar_type() == torch::kFloat32, "Output must be a float32 tensor");
            TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
            TORCH_CHECK(output.is_contiguous(), "Output must be contiguous");

            const int N = input.size(0);
            const int C = input.size(1);

            maxpool2d_forward_launcher(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                N, C, H, W, H_out, W_out,
                kH, kW, sH, sW, pH, pW, dH, dW);
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &maxpool2d_forward_cuda, "MaxPool2d forward (CUDA, Tree-Reduction Unified D2R)");
        }
        '''

        cuda_source = '''
        #include <cuda_runtime.h>
        #include <device_launch_parameters.h>
        #include <limits>

        #define BLOCK_H 8
        #define BLOCK_W 32
        #define THREADS_PER_BLOCK (BLOCK_H * BLOCK_W)
        #define WORK_PER_THREAD_H 2
        #define WORK_PER_THREAD_W 2
        #define OUTPUT_TILE_H (BLOCK_H * WORK_PER_THREAD_H)
        #define OUTPUT_TILE_W (BLOCK_W * WORK_PER_THREAD_W)

        __forceinline__ __device__ float max_float(float a, float b) {
            return a > b ? a : b;
        }

        __global__ void __launch_bounds__(THREADS_PER_BLOCK) maxpool2d_forward_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            const int H, const int W,
            const int H_out, const int W_out,
            const int kH, const int kW,
            const int sH, const int sW,
            const int pH, const int pW,
            const int dH, const int dW)
        {
            const int w_out_base = blockIdx.x * OUTPUT_TILE_W;
            const int h_out_base = blockIdx.y * OUTPUT_TILE_H;
            const int nc = blockIdx.z;

            const float* input_nc = input + nc * H * W;
            float* output_nc = output + nc * H_out * W_out;

            const int tx = threadIdx.x;
            const int ty = threadIdx.y;

            const int h_out_thread_base = h_out_base + ty * WORK_PER_THREAD_H;
            const int w_out_thread_base = w_out_base + tx * WORK_PER_THREAD_W;

            if (h_out_thread_base >= H_out || w_out_thread_base >= W_out) return;

            float max_vals[4] = {
                -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()
            };

            const bool is_k3s2_case = (kH == 3 && kW == 3 && sH == 2 && sW == 2 && dH == 1 && dW == 1);
            const bool is_k3s1_case = (kH == 3 && kW == 3 && sH == 1 && sW == 1 && dH == 1 && dW == 1);
            const bool is_k2s2_case = (kH == 2 && kW == 2 && sH == 2 && sW == 2 && dH == 1 && dW == 1);
            const bool is_k2s1_case = (kH == 2 && kW == 2 && sH == 1 && sW == 1 && dH == 1 && dW == 1);
            const bool is_k5s1_case = (kH == 5 && kW == 5 && sH == 1 && sW == 1 && dH == 1 && dW == 1);
            const float infinity = -std::numeric_limits<float>::infinity();

            if (is_k3s2_case) {
                const int h_in_start = h_out_thread_base * sH - pH;
                const int w_in_start = w_out_thread_base * sW - pW;
                float r[5][5];
                #pragma unroll
                for (int i = 0; i < 5; ++i) {
                    const int g_h = h_in_start + i;
                    const bool h_valid = (g_h >= 0 && g_h < H);
                    const float* g_row_ptr = input_nc + g_h * W + w_in_start;
                    float4 val4;
                    val4.x = (h_valid && w_in_start + 0 >= 0 && w_in_start + 0 < W) ? g_row_ptr[0] : infinity;
                    val4.y = (h_valid && w_in_start + 1 >= 0 && w_in_start + 1 < W) ? g_row_ptr[1] : infinity;
                    val4.z = (h_valid && w_in_start + 2 >= 0 && w_in_start + 2 < W) ? g_row_ptr[2] : infinity;
                    val4.w = (h_valid && w_in_start + 3 >= 0 && w_in_start + 3 < W) ? g_row_ptr[3] : infinity;
                    *(reinterpret_cast<float4*>(&r[i][0])) = val4;
                    r[i][4] = (h_valid && w_in_start + 4 >= 0 && w_in_start + 4 < W) ? g_row_ptr[4] : infinity;
                }
                float col_max_top[5], col_max_bottom[5];
                #pragma unroll
                for(int i = 0; i < 5; ++i) {
                    col_max_top[i] = max_float(r[0][i], max_float(r[1][i], r[2][i]));
                    col_max_bottom[i] = max_float(r[2][i], max_float(r[3][i], r[4][i]));
                }
                max_vals[0] = max_float(col_max_top[0], max_float(col_max_top[1], col_max_top[2]));
                max_vals[1] = max_float(col_max_top[2], max_float(col_max_top[3], col_max_top[4]));
                max_vals[2] = max_float(col_max_bottom[0], max_float(col_max_bottom[1], col_max_bottom[2]));
                max_vals[3] = max_float(col_max_bottom[2], max_float(col_max_bottom[3], col_max_bottom[4]));
            } else if (is_k3s1_case) {
                const int h_in_start = h_out_thread_base * sH - pH;
                const int w_in_start = w_out_thread_base * sW - pW;
                float r[4][4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int g_h = h_in_start + i;
                    const bool h_valid = (g_h >= 0 && g_h < H);
                    const float* g_row_ptr = input_nc + g_h * W + w_in_start;
                    float4 val;
                    val.x = (h_valid && w_in_start + 0 >= 0 && w_in_start + 0 < W) ? g_row_ptr[0] : infinity;
                    val.y = (h_valid && w_in_start + 1 >= 0 && w_in_start + 1 < W) ? g_row_ptr[1] : infinity;
                    val.z = (h_valid && w_in_start + 2 >= 0 && w_in_start + 2 < W) ? g_row_ptr[2] : infinity;
                    val.w = (h_valid && w_in_start + 3 >= 0 && w_in_start + 3 < W) ? g_row_ptr[3] : infinity;
                    *(reinterpret_cast<float4*>(&r[i][0])) = val;
                }
                float col_max_012[4], col_max_123[4];
                #pragma unroll
                for(int i = 0; i < 4; ++i) {
                    col_max_012[i] = max_float(r[0][i], max_float(r[1][i], r[2][i]));
                    col_max_123[i] = max_float(r[1][i], max_float(r[2][i], r[3][i]));
                }
                max_vals[0] = max_float(col_max_012[0], max_float(col_max_012[1], col_max_012[2]));
                max_vals[1] = max_float(col_max_012[1], max_float(col_max_012[2], col_max_012[3]));
                max_vals[2] = max_float(col_max_123[0], max_float(col_max_123[1], col_max_123[2]));
                max_vals[3] = max_float(col_max_123[1], max_float(col_max_123[2], col_max_123[3]));
            } else if (is_k2s2_case) {
                const int h_in_start = h_out_thread_base * sH - pH;
                const int w_in_start = w_out_thread_base * sW - pW;
                float r[4][4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int g_h = h_in_start + i;
                    const bool h_valid = (g_h >= 0 && g_h < H);
                    const float* g_row_ptr = input_nc + g_h * W + w_in_start;
                    float4 val;
                    val.x = (h_valid && w_in_start + 0 >= 0 && w_in_start + 0 < W) ? g_row_ptr[0] : infinity;
                    val.y = (h_valid && w_in_start + 1 >= 0 && w_in_start + 1 < W) ? g_row_ptr[1] : infinity;
                    val.z = (h_valid && w_in_start + 2 >= 0 && w_in_start + 2 < W) ? g_row_ptr[2] : infinity;
                    val.w = (h_valid && w_in_start + 3 >= 0 && w_in_start + 3 < W) ? g_row_ptr[3] : infinity;
                    *(reinterpret_cast<float4*>(&r[i][0])) = val;
                }
                float col_max_top[4], col_max_bottom[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    col_max_top[i] = max_float(r[0][i], r[1][i]);
                    col_max_bottom[i] = max_float(r[2][i], r[3][i]);
                }
                max_vals[0] = max_float(col_max_top[0], col_max_top[1]);
                max_vals[1] = max_float(col_max_top[2], col_max_top[3]);
                max_vals[2] = max_float(col_max_bottom[0], col_max_bottom[1]);
                max_vals[3] = max_float(col_max_bottom[2], col_max_bottom[3]);
            } else if (is_k2s1_case) {
                const int h_in_start = h_out_thread_base * sH - pH;
                const int w_in_start = w_out_thread_base * sW - pW;
                float r[3][4];
                #pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const int g_h = h_in_start + i;
                    const bool h_valid = (g_h >= 0 && g_h < H);
                    const float* g_row_ptr = input_nc + g_h * W + w_in_start;
                    float4 val;
                    val.x = (h_valid && w_in_start + 0 >= 0 && w_in_start + 0 < W) ? g_row_ptr[0] : infinity;
                    val.y = (h_valid && w_in_start + 1 >= 0 && w_in_start + 1 < W) ? g_row_ptr[1] : infinity;
                    val.z = (h_valid && w_in_start + 2 >= 0 && w_in_start + 2 < W) ? g_row_ptr[2] : infinity;
                    val.w = (h_valid && w_in_start + 3 >= 0 && w_in_start + 3 < W) ? g_row_ptr[3] : infinity;
                    *(reinterpret_cast<float4*>(&r[i][0])) = val;
                }
                float col_max_01[3], col_max_12[3];
                #pragma unroll
                for (int i = 0; i < 3; ++i) {
                    col_max_01[i] = max_float(r[0][i], r[1][i]);
                    col_max_12[i] = max_float(r[1][i], r[2][i]);
                }
                max_vals[0] = max_float(col_max_01[0], col_max_01[1]);
                max_vals[1] = max_float(col_max_01[1], col_max_01[2]);
                max_vals[2] = max_float(col_max_12[0], col_max_12[1]);
                max_vals[3] = max_float(col_max_12[1], col_max_12[2]);
            } else if (is_k5s1_case) {
                const int h_in_start = h_out_thread_base * sH - pH;
                const int w_in_start = w_out_thread_base * sW - pW;
                float r[6][6];
                #pragma unroll
                for (int i = 0; i < 6; ++i) {
                    const int g_h = h_in_start + i;
                    const bool h_valid = (g_h >= 0 && g_h < H);
                    const float* g_row_ptr = input_nc + g_h * W + w_in_start;
                    float4 val4; float2 val2;
                    val4.x = (h_valid && w_in_start + 0 >= 0 && w_in_start + 0 < W) ? g_row_ptr[0] : infinity;
                    val4.y = (h_valid && w_in_start + 1 >= 0 && w_in_start + 1 < W) ? g_row_ptr[1] : infinity;
                    val4.z = (h_valid && w_in_start + 2 >= 0 && w_in_start + 2 < W) ? g_row_ptr[2] : infinity;
                    val4.w = (h_valid && w_in_start + 3 >= 0 && w_in_start + 3 < W) ? g_row_ptr[3] : infinity;
                    val2.x = (h_valid && w_in_start + 4 >= 0 && w_in_start + 4 < W) ? g_row_ptr[4] : infinity;
                    val2.y = (h_valid && w_in_start + 5 >= 0 && w_in_start + 5 < W) ? g_row_ptr[5] : infinity;
                    *(reinterpret_cast<float4*>(&r[i][0])) = val4;
                    *(reinterpret_cast<float2*>(&r[i][4])) = val2;
                }
                float col_max_0[6], col_max_1[6];
                #pragma unroll
                for(int i = 0; i < 6; ++i) {
                    col_max_0[i] = max_float(max_float(r[0][i], r[1][i]), max_float(max_float(r[2][i], r[3][i]), r[4][i]));
                    col_max_1[i] = max_float(max_float(r[1][i], r[2][i]), max_float(max_float(r[3][i], r[4][i]), r[5][i]));
                }
                max_vals[0] = max_float(max_float(col_max_0[0], col_max_0[1]), max_float(max_float(col_max_0[2], col_max_0[3]), col_max_0[4]));
                max_vals[1] = max_float(max_float(col_max_0[1], col_max_0[2]), max_float(max_float(col_max_0[3], col_max_0[4]), col_max_0[5]));
                max_vals[2] = max_float(max_float(col_max_1[0], col_max_1[1]), max_float(max_float(col_max_1[2], col_max_1[3]), col_max_1[4]));
                max_vals[3] = max_float(max_float(col_max_1[1], col_max_1[2]), max_float(max_float(col_max_1[3], col_max_1[4]), col_max_1[5]));
            } else {
                const int h_in_base_thread = h_out_thread_base * sH - pH;
                const int w_in_base_thread = w_out_thread_base * sW - pW;
                #pragma unroll
                for (int kh = 0; kh < kH; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < kW; ++kw) {
                        const int h_in_0 = h_in_base_thread + kh * dH;
                        const int w_in_0 = w_in_base_thread + kw * dW;
                        const bool v0 = (h_in_0 >= 0 && h_in_0 < H && w_in_0 >= 0 && w_in_0 < W);
                        max_vals[0] = max_float(max_vals[0], v0 ? input_nc[h_in_0 * W + w_in_0] : infinity);

                        const int w_in_1 = w_in_0 + sW;
                        const bool v1 = (h_in_0 >= 0 && h_in_0 < H && w_in_1 >= 0 && w_in_1 < W);
                        max_vals[1] = max_float(max_vals[1], v1 ? input_nc[h_in_0 * W + w_in_1] : infinity);

                        const int h_in_1 = h_in_0 + sH;
                        const bool v2 = (h_in_1 >= 0 && h_in_1 < H && w_in_0 >= 0 && w_in_0 < W);
                        max_vals[2] = max_float(max_vals[2], v2 ? input_nc[h_in_1 * W + w_in_0] : infinity);

                        const bool v3 = (h_in_1 >= 0 && h_in_1 < H && w_in_1 >= 0 && w_in_1 < W);
                        max_vals[3] = max_float(max_vals[3], v3 ? input_nc[h_in_1 * W + w_in_1] : infinity);
                    }
                }
            }

            const int h_out0 = h_out_thread_base;
            const int w_out0 = w_out_thread_base;
            const int h_out1 = h_out0 + 1;

            if (h_out0 < H_out) {
                if (w_out0 + 1 < W_out) {
                    *(reinterpret_cast<float2*>(output_nc + h_out0 * W_out + w_out0)) = make_float2(max_vals[0], max_vals[1]);
                } else if (w_out0 < W_out) {
                    output_nc[h_out0 * W_out + w_out0] = max_vals[0];
                }
            }
            if (h_out1 < H_out) {
                if (w_out0 + 1 < W_out) {
                    *(reinterpret_cast<float2*>(output_nc + h_out1 * W_out + w_out0)) = make_float2(max_vals[2], max_vals[3]);
                } else if (w_out0 < W_out) {
                    output_nc[h_out1 * W_out + w_out0] = max_vals[2];
                }
            }
        }

        void maxpool2d_forward_launcher(
            const float* input,
            float* output,
            int N, int C, int H, int W,
            int H_out, int W_out,
            int kH, int kW, int sH, int sW,
            int pH, int pW, int dH, int dW)
        {
            dim3 block(BLOCK_W, BLOCK_H);
            dim3 grid((W_out + OUTPUT_TILE_W - 1) / OUTPUT_TILE_W,
                      (H_out + OUTPUT_TILE_H - 1) / OUTPUT_TILE_H,
                      N * C);

            maxpool2d_forward_kernel<<<grid, block, 0>>>(
                input, output, H, W, H_out, W_out,
                kH, kW, sH, sW, pH, pW, dH, dW);

            cudaGetLastError();
        }
        '''

        self.cuda_module = load_inline(
            name='maxpool2d_tree_reduce_unified_d2r',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            verbose=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Applies the custom Max Pooling 2D CUDA kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        '''
        self._load_cuda_module()

        x = x.contiguous()

        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

        output = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

        self.cuda_module.forward(x, output, H, W, H_out, W_out, kH, kW, sH, sW, pH, pW, dH, dW)

        return output