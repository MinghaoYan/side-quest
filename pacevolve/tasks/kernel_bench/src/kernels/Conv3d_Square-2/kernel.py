# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _triple
import math

# Global cache for the compiled CUDA module to avoid recompilation
_cached_winograd_module = None

class ModelNew(nn.Module):
    """
    Implements a 3D convolution using the Winograd F(2x2x2, 3x3x3) algorithm.
    This version enhances performance by offloading the kernel transformation step
    (U = G @ W @ G.T) from PyTorch's `einsum` to a dedicated, fused CUDA kernel.
    This reduces kernel launch overhead and eliminates intermediate tensor allocations
    on the host side, keeping the entire transformation pipeline on the GPU.

    The data transformation kernels remain optimized with intra-tile parallelism,
    where a cooperative thread block (64 threads) processes each 4x4x4 data tile.
    
    This implementation is specialized for 3x3x3 kernels with stride 1, padding 1,
    and dilation 1. For other convolution parameters, it falls back to the
    standard PyTorch `F.conv3d` implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        kernel_size_t = _triple(kernel_size)
        stride_t = _triple(stride)
        padding_t = _triple(padding)
        dilation_t = _triple(dilation)

        self.use_winograd = (
            kernel_size_t == (3, 3, 3) and
            stride_t == (1, 1, 1) and
            padding_t == (1, 1, 1) and
            dilation_t == (1, 1, 1) and
            groups == 1
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size_t
        self.stride = stride_t
        self.padding = padding_t
        self.dilation = dilation_t

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if self.use_winograd:
            self.tile_size = 4
            self.output_tile_size = 2
            self.winograd_op = self._load_cuda_kernel()
            G = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32, device='cuda')
            self.register_buffer('G', G)
            self.register_buffer('U_bmm', None)
        else:
            self.winograd_op = None

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _transform_kernel(self, weight):
        # Offload the transformation to the custom CUDA kernel
        return self.winograd_op.winograd_kernel_transform(weight.contiguous(), self.G)

    def _load_cuda_kernel(self):
        global _cached_winograd_module
        if _cached_winograd_module is not None:
            return _cached_winograd_module

        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <cuda_fp16.h>
        #include <c10/cuda/CUDAStream.h>

        // B.T * d
        __device__ void transform_Bt(const half* d, half* o, int stride) {
            o[0*stride] = d[0*stride] - d[2*stride];
            o[1*stride] = d[1*stride] + d[2*stride];
            o[2*stride] = -d[1*stride] + d[2*stride];
            o[3*stride] = d[1*stride] - d[3*stride];
        }

        // A.T * m
        __device__ void transform_At(const half* m, half* y, int stride) {
            y[0*stride] = m[0*stride] + m[1*stride] + m[2*stride];
            y[1*stride] = m[1*stride] - m[2*stride] - m[3*stride];
        }
        
        __global__ void winograd_kernel_transform_kernel(
            const float* W_in, const float* G_in, half* U_out,
            const int C_out, const int C_in) {
            
            const int c_in_idx = blockIdx.x;
            const int c_out_idx = blockIdx.y;

            const int tid = threadIdx.x;
            
            // Each thread computes one element of the 4x4x4 transformed kernel U
            const int u_k = tid / 16;
            const int u_j = (tid / 4) % 4;
            const int u_i = tid % 4;

            __shared__ float w_tile[3][3][3];
            __shared__ float g_tile[4][3];

            // Load 3x3x3 weight tile into shared memory
            if (tid < 27) {
                int d = tid / 9;
                int e = (tid / 3) % 3;
                int f = tid % 3;
                long w_offset = (long)c_out_idx * C_in * 27 + (long)c_in_idx * 27 + tid;
                w_tile[d][e][f] = W_in[w_offset];
            }

            // Load 4x3 G matrix into shared memory
            if (tid < 12) {
                g_tile[tid / 3][tid % 3] = G_in[tid];
            }
            __syncthreads();

            // Compute U_uji = sum_d,e,f (G_ud * G_je * G_if * W_def)
            float u_val = 0.0f;
            for (int d = 0; d < 3; ++d) {
                for (int e = 0; e < 3; ++e) {
                    for (int f = 0; f < 3; ++f) {
                        u_val += g_tile[u_k][d] * g_tile[u_j][e] * g_tile[u_i][f] * w_tile[d][e][f];
                    }
                }
            }
            
            // Write to output tensor in (alpha, C_out, C_in) layout
            long alpha = u_k * 16 + u_j * 4 + u_i;
            long out_offset = alpha * C_out * C_in + c_out_idx * C_in + c_in_idx;
            U_out[out_offset] = __float2half(u_val);
        }

        __global__ void winograd_input_transform_kernel(
            const half* input, half* output,
            const int C, const int D, const int H, const int W,
            const int num_tiles_d, const int num_tiles_h, const int num_tiles_w) {
            
            const int tile_idx_channel = blockIdx.x;
            const int n = blockIdx.y;
            const int tid = threadIdx.x;

            const int tile_size = 4;
            const int output_tile_size = 2;
            
            const long total_tiles_per_channel = (long)num_tiles_d * num_tiles_h * num_tiles_w;
            const long tile_idx_global = (long)n * C * total_tiles_per_channel + tile_idx_channel;

            const int c = tile_idx_channel / total_tiles_per_channel;
            const int tile_idx_plane = tile_idx_channel % total_tiles_per_channel;
            const int td = tile_idx_plane / (num_tiles_h * num_tiles_w);
            const int th = (tile_idx_plane / num_tiles_w) % num_tiles_h;
            const int tw = tile_idx_plane % num_tiles_w;
            
            const int d_in_start = td * output_tile_size;
            const int h_in_start = th * output_tile_size;
            const int w_in_start = tw * output_tile_size;

            __shared__ half patch[tile_size][tile_size][tile_size];
            
            int k = tid / (tile_size * tile_size);
            int j = (tid / tile_size) % tile_size;
            int i = tid % tile_size;

            long input_offset = (long)n * C * D * H * W + (long)c * D * H * W +
                                (long)(d_in_start + k) * H * W + (long)(h_in_start + j) * W + (w_in_start + i);
            patch[k][j][i] = input[input_offset];
            
            __syncthreads();

            __shared__ half tmp1[tile_size][tile_size][tile_size];
            __shared__ half tmp2[tile_size][tile_size][tile_size];
            
            if (tid < 16) {
                int row_k = tid / 4;
                int row_j = tid % 4;
                transform_Bt(&patch[row_k][row_j][0], &tmp1[row_k][row_j][0], 1);
            }
            __syncthreads();
            if (tid < 16) {
                int col_k = tid / 4;
                int col_i = tid % 4;
                transform_Bt(&tmp1[col_k][0][col_i], &tmp2[col_k][0][col_i], tile_size);
            }
            __syncthreads();
            if (tid < 16) {
                int dcol_j = tid / 4;
                int dcol_i = tid % 4;
                transform_Bt(&tmp2[0][dcol_j][dcol_i], &patch[0][dcol_j][dcol_i], tile_size * tile_size);
            }
            __syncthreads();
                
            long V_idx = (long)k*tile_size*tile_size + j*tile_size + i;
            long output_offset = V_idx * gridDim.x * gridDim.y + tile_idx_global;
            output[output_offset] = patch[k][j][i];
        }

        __global__ void winograd_output_transform_kernel(
            const half* input, half* output,
            const int C_out, const int D, const int H, const int W,
            const int num_tiles_d, const int num_tiles_h, const int num_tiles_w) {

            const int tile_idx_channel = blockIdx.x;
            const int n = blockIdx.y;
            const int tid = threadIdx.x;

            const int tile_size = 4;
            const int output_tile_size = 2;

            const long total_tiles_per_channel = (long)num_tiles_d * num_tiles_h * num_tiles_w;
            const long tile_idx_global = (long)n * C_out * total_tiles_per_channel + tile_idx_channel;

            const int c = tile_idx_channel / total_tiles_per_channel;
            const int tile_idx_plane = tile_idx_channel % total_tiles_per_channel;
            const int td = tile_idx_plane / (num_tiles_h * num_tiles_w);
            const int th = (tile_idx_plane / num_tiles_w) % num_tiles_h;
            const int tw = tile_idx_plane % num_tiles_w;

            __shared__ half M_tile[tile_size][tile_size][tile_size];
            
            int k = tid / (tile_size * tile_size);
            int j = (tid / tile_size) % tile_size;
            int i = tid % tile_size;
            
            long M_idx = (long)k*tile_size*tile_size + j*tile_size + i;
            long input_offset = M_idx * gridDim.x * gridDim.y + tile_idx_global;
            M_tile[k][j][i] = input[input_offset];
            __syncthreads();
            
            __shared__ half tmp1[output_tile_size][tile_size][tile_size];
            __shared__ half tmp2[output_tile_size][output_tile_size][tile_size];
            __shared__ half Y_tile[output_tile_size][output_tile_size][output_tile_size];

            if (tid < 16) {
                int dcol_j = tid / 4;
                int dcol_i = tid % 4;
                transform_At(&M_tile[0][dcol_j][dcol_i], &tmp1[0][dcol_j][dcol_i], tile_size * tile_size);
            }
            __syncthreads();
            if (tid < 8) {
                int col_k = tid / 4; // 0-1
                int col_i = tid % 4; // 0-3
                transform_At(&tmp1[col_k][0][col_i], &tmp2[col_k][0][col_i], tile_size);
            }
            __syncthreads();
            if (tid < 4) {
                int row_k = tid / 2; // 0-1
                int row_j = tid % 2; // 0-1
                transform_At(&tmp2[row_k][row_j][0], &Y_tile[row_k][row_j][0], 1);
            }
            __syncthreads();

            if (tid < 8) {
                int out_k = tid / (output_tile_size * output_tile_size);
                int out_j = (tid / output_tile_size) % output_tile_size;
                int out_i = tid % output_tile_size;
                
                const int d_out_start = td * output_tile_size;
                const int h_out_start = th * output_tile_size;
                const int w_out_start = tw * output_tile_size;
                
                long output_offset = (long)n * C_out * D * H * W + (long)c * D * H * W +
                                     (long)(d_out_start + out_k) * H * W + (long)(h_out_start + out_j) * W + (w_out_start + out_i);
                if ((d_out_start + out_k < D) && (h_out_start + out_j < H) && (w_out_start + out_i < W)) {
                    output[output_offset] = Y_tile[out_k][out_j][out_i];
                }
            }
        }
        
        torch::Tensor winograd_kernel_transform(torch::Tensor W, torch::Tensor G) {
            const int C_out = W.size(0);
            const int C_in = W.size(1);
            const int K = W.size(2);
            const int tile_size = 4;
            auto U_bmm = torch::empty({tile_size*tile_size*tile_size, C_out, C_in}, W.options().dtype(torch::kHalf));
            
            dim3 grid(C_in, C_out);
            dim3 block(tile_size * tile_size * tile_size);

            winograd_kernel_transform_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                W.data_ptr<float>(), G.data_ptr<float>(), (half*)U_bmm.data_ptr<at::Half>(),
                C_out, C_in);
            TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel (kernel_transform) launch failed");
            return U_bmm;
        }

        torch::Tensor winograd_input_transform(torch::Tensor input) {
            const int N = input.size(0); const int C = input.size(1);
            const int D = input.size(2); const int H = input.size(3); const int W = input.size(4);
            const int output_tile_size = 2; const int tile_size = 4;
            const int num_tiles_d = (D - tile_size) / output_tile_size + 1;
            const int num_tiles_h = (H - tile_size) / output_tile_size + 1;
            const int num_tiles_w = (W - tile_size) / output_tile_size + 1;
            const long num_tiles_total = (long)num_tiles_d * num_tiles_h * num_tiles_w;
            if (num_tiles_total == 0) return torch::empty({0}, input.options());

            auto V = torch::empty({tile_size*tile_size*tile_size, N * C * num_tiles_total}, input.options());

            const int threads = 64;
            dim3 grid(C * num_tiles_total, N);

            winograd_input_transform_kernel<<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                (const half*)input.data_ptr<at::Half>(), (half*)V.data_ptr<at::Half>(),
                C, D, H, W, num_tiles_d, num_tiles_h, num_tiles_w);
            TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel (input_transform) launch failed");
            return V;
        }

        torch::Tensor winograd_output_transform(torch::Tensor M, int N, int C_out, int D_out, int H_out, int W_out) {
            const int output_tile_size = 2;
            const int num_tiles_d = D_out / output_tile_size; const int num_tiles_h = H_out / output_tile_size; const int num_tiles_w = W_out / output_tile_size;
            const long num_tiles_total = (long)num_tiles_d * num_tiles_h * num_tiles_w;
            if (num_tiles_total == 0) return torch::empty({N, C_out, D_out, H_out, W_out}, M.options());
            
            auto output = torch::empty({N, C_out, D_out, H_out, W_out}, M.options());

            const int threads = 64;
            dim3 grid(C_out * num_tiles_total, N);

            winograd_output_transform_kernel<<<grid, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                (const half*)M.data_ptr<at::Half>(), (half*)output.data_ptr<at::Half>(),
                C_out, D_out, H_out, W_out, num_tiles_d, num_tiles_h, num_tiles_w);
            TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel (output_transform) launch failed");
            return output;
        }
        """
        cpp_source = """
        torch::Tensor winograd_kernel_transform(torch::Tensor W, torch::Tensor G);
        torch::Tensor winograd_input_transform(torch::Tensor input);
        torch::Tensor winograd_output_transform(torch::Tensor M, int N, int C_out, int D_out, int H_out, int W_out);
        """

        winograd_module = torch.utils.cpp_extension.load_inline(
            name='winograd_3d_cuda_v4_kernel_transform',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['winograd_kernel_transform', 'winograd_input_transform', 'winograd_output_transform'],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
        _cached_winograd_module = winograd_module
        return _cached_winograd_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_winograd:
            return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        N, C, D, H, W = x.shape
        out_d, out_h, out_w = D, H, W
        
        if out_d % self.output_tile_size != 0 or out_h % self.output_tile_size != 0 or out_w % self.output_tile_size != 0:
            return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        if self.U_bmm is None or self.training:
            # The weight parameter is float32 by default
            self.U_bmm = self._transform_kernel(self.weight)
        
        x_padded = F.pad(x.half(), (1, 1, 1, 1, 1, 1))

        # 1. Input transform: V = B.T @ d @ B
        V = self.winograd_op.winograd_input_transform(x_padded)
        if V.numel() == 0:
             output_shape = (N, self.out_channels, out_d, out_h, out_w)
             output = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
             if self.bias is not None:
                output += self.bias.view(1,-1,1,1,1)
             return output

        # Reshape for BMM
        num_tiles_d = out_d // self.output_tile_size
        num_tiles_h = out_h // self.output_tile_size
        num_tiles_w = out_w // self.output_tile_size
        num_tiles_total = num_tiles_d * num_tiles_h * num_tiles_w
        
        V_bmm = V.view(self.tile_size**3, N, C, num_tiles_total).permute(0, 2, 1, 3).reshape(self.tile_size**3, C, N*num_tiles_total)
        
        # 2. Batched matrix multiplication: M = U @ V
        M_bmm = torch.bmm(self.U_bmm, V_bmm)
        
        # Reshape for output transform
        M = M_bmm.view(self.tile_size**3, self.out_channels, N, num_tiles_total).permute(0,2,1,3).reshape(self.tile_size**3, N*self.out_channels*num_tiles_total)

        # 3. Output transform: Y = A.T @ M @ A
        output_half = self.winograd_op.winograd_output_transform(M.contiguous(), N, self.out_channels, out_d, out_h, out_w)

        if self.bias is not None:
            output_half += self.bias.half().view(1, -1, 1, 1, 1)

        return output_half.float()
# RegexTagCustomPruningAlgorithmEnd