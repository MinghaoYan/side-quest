import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

_fused_op_lib = None

def load_fused_op_lib():
    global _fused_op_lib
    if _fused_op_lib is not None:
        return _fused_op_lib

    cuda_source = r'''
#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE_DIM 32
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 8
#define WPT 4          // Work per thread (rows)
#define WPT_X 2        // Work per thread (cols)

__global__ void fused_linear_relu_kernel_T(
    const float* __restrict__ A,   // M x K (input)
    const float* __restrict__ B_T, // K x N (transposed weight)
    const float* __restrict__ bias, // N
    float* __restrict__ C,       // M x N (output)
    int M, int N, int K) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col_base = blockIdx.x * TILE_DIM + tx * WPT_X;
    const int row_base = blockIdx.y * TILE_DIM + ty * WPT;

    __shared__ float As[2][TILE_DIM][TILE_DIM + 1]; // Padded to avoid bank conflicts
    __shared__ float Bs[2][TILE_DIM][TILE_DIM + 1]; // Padded to avoid bank conflicts

    float acc[WPT][WPT_X];
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        #pragma unroll
        for (int j = 0; j < WPT_X; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Correctness fix for K=0
    if (num_tiles == 0) {
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int current_row = row_base + i;
            if (current_row < M) {
                #pragma unroll
                for (int j = 0; j < WPT_X; ++j) {
                    int current_col = col_base + j;
                    if (current_col < N) {
                        float val = bias[current_col];
                        C[current_row * N + current_col] = fmaxf(0.0f, val);
                    }
                }
            }
        }
        return;
    }

    int t = 0;
    // Load first tile
    {
        const int g_col_A_base = t * TILE_DIM;
        const int g_row_A_base = blockIdx.y * TILE_DIM;
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int sm_row = ty * WPT + i;
            int g_row = g_row_A_base + sm_row;
            #pragma unroll
            for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                int sm_col = tx + j * BLOCK_DIM_X;
                int g_col = g_col_A_base + sm_col;
                As[0][sm_row][sm_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
            }
        }

        const int g_col_B_base = blockIdx.x * TILE_DIM;
        const int g_row_B_base = t * TILE_DIM;
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int sm_row = ty * WPT + i;
            int g_row = g_row_B_base + sm_row;
            #pragma unroll
            for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                int sm_col = tx + j * BLOCK_DIM_X;
                int g_col = g_col_B_base + sm_col;
                Bs[0][sm_row][sm_col] = (g_row < K && g_col < N) ? B_T[g_row * N + g_col] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (t = 0; t < num_tiles - 1; ++t) {
        const int current_buf = t % 2;
        const int next_buf = (t + 1) % 2;

        // Pre-fetch next tile
        {
            const int g_col_A_base = (t + 1) * TILE_DIM;
            const int g_row_A_base = blockIdx.y * TILE_DIM;
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                int sm_row = ty * WPT + i;
                int g_row = g_row_A_base + sm_row;
                #pragma unroll
                for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                    int sm_col = tx + j * BLOCK_DIM_X;
                    int g_col = g_col_A_base + sm_col;
                    As[next_buf][sm_row][sm_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
                }
            }
        }
        {
            const int g_col_B_base = blockIdx.x * TILE_DIM;
            const int g_row_B_base = (t + 1) * TILE_DIM;
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                int sm_row = ty * WPT + i;
                int g_row = g_row_B_base + sm_row;
                #pragma unroll
                for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                    int sm_col = tx + j * BLOCK_DIM_X;
                    int g_col = g_col_B_base + sm_col;
                    Bs[next_buf][sm_row][sm_col] = (g_row < K && g_col < N) ? B_T[g_row * N + g_col] : 0.0f;
                }
            }
        }

        // Compute with current tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float b_vals[WPT_X];
            #pragma unroll
            for(int j=0; j<WPT_X; ++j) b_vals[j] = Bs[current_buf][k][tx * WPT_X + j];
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                const float a_val = As[current_buf][ty * WPT + i][k];
                #pragma unroll
                for (int j = 0; j < WPT_X; ++j) acc[i][j] += a_val * b_vals[j];
            }
        }
        __syncthreads();
    }

    // Compute with last tile
    int last_buf = (num_tiles - 1) % 2;
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        float b_vals[WPT_X];
        #pragma unroll
        for(int j=0; j<WPT_X; ++j) b_vals[j] = Bs[last_buf][k][tx * WPT_X + j];
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            const float a_val = As[last_buf][ty * WPT + i][k];
            #pragma unroll
            for (int j = 0; j < WPT_X; ++j) acc[i][j] += a_val * b_vals[j];
        }
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        int current_row = row_base + i;
        if (current_row < M) {
            #pragma unroll
            for (int j = 0; j < WPT_X; ++j) {
                int current_col = col_base + j;
                if (current_col < N) {
                    float val = acc[i][j] + bias[current_col];
                    C[current_row * N + current_col] = fmaxf(0.0f, val);
                }
            }
        }
    }
}

__global__ void linear_kernel_T(
    const float* __restrict__ A,   // M x K (input)
    const float* __restrict__ B_T, // K x N (transposed weight)
    const float* __restrict__ bias, // N
    float* __restrict__ C,       // M x N (output)
    int M, int N, int K) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col_base = blockIdx.x * TILE_DIM + tx * WPT_X;
    const int row_base = blockIdx.y * TILE_DIM + ty * WPT;

    __shared__ float As[2][TILE_DIM][TILE_DIM + 1];
    __shared__ float Bs[2][TILE_DIM][TILE_DIM + 1];

    float acc[WPT][WPT_X];
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        #pragma unroll
        for (int j = 0; j < WPT_X; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Correctness fix for K=0
    if (num_tiles == 0) {
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int current_row = row_base + i;
            if (current_row < M) {
                #pragma unroll
                for (int j = 0; j < WPT_X; ++j) {
                    int current_col = col_base + j;
                    if (current_col < N) {
                        C[current_row * N + current_col] = bias[current_col];
                    }
                }
            }
        }
        return;
    }

    int t = 0;
    // Load first tile
    {
        const int g_col_A_base = t * TILE_DIM;
        const int g_row_A_base = blockIdx.y * TILE_DIM;
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int sm_row = ty * WPT + i;
            int g_row = g_row_A_base + sm_row;
            #pragma unroll
            for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                int sm_col = tx + j * BLOCK_DIM_X;
                int g_col = g_col_A_base + sm_col;
                As[0][sm_row][sm_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
            }
        }

        const int g_col_B_base = blockIdx.x * TILE_DIM;
        const int g_row_B_base = t * TILE_DIM;
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            int sm_row = ty * WPT + i;
            int g_row = g_row_B_base + sm_row;
            #pragma unroll
            for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                int sm_col = tx + j * BLOCK_DIM_X;
                int g_col = g_col_B_base + sm_col;
                Bs[0][sm_row][sm_col] = (g_row < K && g_col < N) ? B_T[g_row * N + g_col] : 0.0f;
            }
        }
    }
    __syncthreads();

    for (t = 0; t < num_tiles - 1; ++t) {
        const int current_buf = t % 2;
        const int next_buf = (t + 1) % 2;

        // Pre-fetch next tile
        {
            const int g_col_A_base = (t + 1) * TILE_DIM;
            const int g_row_A_base = blockIdx.y * TILE_DIM;
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                int sm_row = ty * WPT + i;
                int g_row = g_row_A_base + sm_row;
                #pragma unroll
                for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                    int sm_col = tx + j * BLOCK_DIM_X;
                    int g_col = g_col_A_base + sm_col;
                    As[next_buf][sm_row][sm_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
                }
            }
        }
        {
            const int g_col_B_base = blockIdx.x * TILE_DIM;
            const int g_row_B_base = (t + 1) * TILE_DIM;
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                int sm_row = ty * WPT + i;
                int g_row = g_row_B_base + sm_row;
                #pragma unroll
                for (int j = 0; j < (TILE_DIM / BLOCK_DIM_X); ++j) {
                    int sm_col = tx + j * BLOCK_DIM_X;
                    int g_col = g_col_B_base + sm_col;
                    Bs[next_buf][sm_row][sm_col] = (g_row < K && g_col < N) ? B_T[g_row * N + g_col] : 0.0f;
                }
            }
        }

        // Compute with current tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float b_vals[WPT_X];
            #pragma unroll
            for(int j=0; j<WPT_X; ++j) b_vals[j] = Bs[current_buf][k][tx * WPT_X + j];
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                const float a_val = As[current_buf][ty * WPT + i][k];
                #pragma unroll
                for (int j = 0; j < WPT_X; ++j) acc[i][j] += a_val * b_vals[j];
            }
        }
        __syncthreads();
    }

    // Compute with last tile
    int last_buf = (num_tiles - 1) % 2;
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        float b_vals[WPT_X];
        #pragma unroll
        for(int j=0; j<WPT_X; ++j) b_vals[j] = Bs[last_buf][k][tx * WPT_X + j];
        #pragma unroll
        for (int i = 0; i < WPT; ++i) {
            const float a_val = As[last_buf][ty * WPT + i][k];
            #pragma unroll
            for (int j = 0; j < WPT_X; ++j) acc[i][j] += a_val * b_vals[j];
        }
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        int current_row = row_base + i;
        if (current_row < M) {
            #pragma unroll
            for (int j = 0; j < WPT_X; ++j) {
                int current_col = col_base + j;
                if (current_col < N) {
                    C[current_row * N + current_col] = acc[i][j] + bias[current_col];
                }
            }
        }
    }
}

void fused_linear_relu_forward_cuda(torch::Tensor A, torch::Tensor B_T, torch::Tensor bias, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_T.size(1);
    const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    const dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    fused_linear_relu_kernel_T<<<blocks, threads>>>(A.data_ptr<float>(), B_T.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void linear_forward_cuda(torch::Tensor A, torch::Tensor B_T, torch::Tensor bias, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_T.size(1);
    const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    const dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    linear_kernel_T<<<blocks, threads>>>(A.data_ptr<float>(), B_T.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
'''

    cpp_source = r'''
#include <torch/extension.h>
void fused_linear_relu_forward_cuda(torch::Tensor A, torch::Tensor B_T, torch::Tensor bias, torch::Tensor C);
void linear_forward_cuda(torch::Tensor A, torch::Tensor B_T, torch::Tensor bias, torch::Tensor C);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_linear_relu_forward(torch::Tensor A, torch::Tensor B_T, torch::Tensor bias) {
    CHECK_INPUT(A); CHECK_INPUT(B_T); CHECK_INPUT(bias);
    TORCH_CHECK(A.dim() == 2 && B_T.dim() == 2, "Inputs must be 2D");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_T.size(1);
    TORCH_CHECK(K == B_T.size(0), "Input features and weight features do not match");
    auto C = torch::empty({M, N}, A.options());
    fused_linear_relu_forward_cuda(A, B_T, bias, C);
    return C;
}

torch::Tensor linear_forward(torch::Tensor A, torch::Tensor B_T, torch::Tensor bias) {
    CHECK_INPUT(A); CHECK_INPUT(B_T); CHECK_INPUT(bias);
    TORCH_CHECK(A.dim() == 2 && B_T.dim() == 2, "Inputs must be 2D");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_T.size(1);
    TORCH_CHECK(K == B_T.size(0), "Input features and weight features do not match");
    auto C = torch::empty({M, N}, A.options());
    linear_forward_cuda(A, B_T, bias, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_relu", &fused_linear_relu_forward);
    m.def("linear", &linear_forward);
}
'''
    _fused_op_lib = load_inline(
        name=f"fused_op_lib_{os.getpid()}",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        verbose=False,
    )
    return _fused_op_lib

class FusedLinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        _linear = nn.Linear(in_features, out_features)
        # Transpose weight for coalesced memory access and better shared memory access pattern
        self.weight = nn.Parameter(_linear.weight.T.contiguous())
        self.bias = _linear.bias
        self.fused_op_lib = load_fused_op_lib()

    def forward(self, x):
        return self.fused_op_lib.fused_linear_relu(x, self.weight, self.bias)

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        _linear = nn.Linear(in_features, out_features)
        # Transpose weight for coalesced memory access and better shared memory access pattern
        self.weight = nn.Parameter(_linear.weight.T.contiguous())
        self.bias = _linear.bias
        self.fused_op_lib = load_fused_op_lib()

    def forward(self, x):
        return self.fused_op_lib.linear(x, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()

        layers = []
        current_input_size = input_size

        for layer_size in layer_sizes:
            layers.append(FusedLinearReLU(current_input_size, layer_size))
            current_input_size = layer_size

        layers.append(CustomLinear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)