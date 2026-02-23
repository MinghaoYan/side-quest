import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# JIT compilation of the CUDA kernel. This is done once when the module is imported.
# The kernel implements a tiled matrix multiplication algorithm for high performance.
try:
    cuda_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <string>

// --- Kernel Configuration ---
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_COLS 8
#define THREAD_WORK_ROWS 4
#define THREAD_WORK_COLS 4

// Simplified CUDA kernel for matrix multiplication without software pipelining.
// It uses vectorized memory access and register tiling for high performance,
// with robust boundary handling.
__global__ void matmul_kernel_simplified_vectorized(const float* A, const float* B, float* C, int M, int K, int N) {
    // Single-buffered shared memory.
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1]; // Padded

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_COLS + tx;
    const int threads_per_block = BLOCK_ROWS * BLOCK_COLS;

    const int base_row_c = blockIdx.y * TILE_DIM + ty * THREAD_WORK_ROWS;
    const int base_col_c = blockIdx.x * TILE_DIM + tx * THREAD_WORK_COLS;

    float c_vals[THREAD_WORK_ROWS][THREAD_WORK_COLS] = {{0.0f}};

    for (int p = 0; p < K; p += TILE_DIM) {
        // --- Vectorized Load from Global to Shared Memory ---
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            int idx = tid + i * threads_per_block;
            // Load A
            int l_row_A = (idx * 4) / TILE_DIM;
            int l_col_A = (idx * 4) % TILE_DIM;
            int g_row_A = blockIdx.y * TILE_DIM + l_row_A;
            int g_col_A = p + l_col_A;
            if (g_row_A < M && g_col_A + 3 < K) {
                *(float4*)&As[l_row_A][l_col_A] = *(const float4*)&A[g_row_A * K + g_col_A];
            } else {
                #pragma unroll
                for(int k_off=0; k_off<4; ++k_off) {
                    if (g_row_A < M && g_col_A + k_off < K) As[l_row_A][l_col_A + k_off] = A[g_row_A * K + g_col_A + k_off];
                    else As[l_row_A][l_col_A + k_off] = 0.0f;
                }
            }

            // Load B (transposed)
            int row_in_B_tile = idx / (TILE_DIM / 4);
            int col_in_B_tile_f4 = idx % (TILE_DIM / 4);
            int g_row_B = p + row_in_B_tile;
            int g_col_B = blockIdx.x * TILE_DIM + col_in_B_tile_f4 * 4;
            if (g_row_B < K && g_col_B + 3 < N) {
                float4 b_read = *(const float4*)&B[g_row_B * N + g_col_B];
                Bs[col_in_B_tile_f4 * 4 + 0][row_in_B_tile] = b_read.x;
                Bs[col_in_B_tile_f4 * 4 + 1][row_in_B_tile] = b_read.y;
                Bs[col_in_B_tile_f4 * 4 + 2][row_in_B_tile] = b_read.z;
                Bs[col_in_B_tile_f4 * 4 + 3][row_in_B_tile] = b_read.w;
            } else {
                #pragma unroll
                for (int j=0; j<4; ++j) {
                    if(g_row_B < K && g_col_B + j < N) Bs[col_in_B_tile_f4 * 4 + j][row_in_B_tile] = B[g_row_B * N + g_col_B + j];
                    else Bs[col_in_B_tile_f4 * 4 + j][row_in_B_tile] = 0.0f;
                }
            }
        }
        __syncthreads();

        // --- Computation ---
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            float a_regs[THREAD_WORK_ROWS];
            float b_regs[THREAD_WORK_COLS];
            #pragma unroll
            for(int i = 0; i < THREAD_WORK_ROWS; ++i) a_regs[i] = As[ty*THREAD_WORK_ROWS+i][k];
            #pragma unroll
            for(int j = 0; j < THREAD_WORK_COLS; ++j) b_regs[j] = Bs[tx*THREAD_WORK_COLS+j][k];
            #pragma unroll
            for (int i = 0; i < THREAD_WORK_ROWS; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_WORK_COLS; ++j) {
                    c_vals[i][j] += a_regs[i] * b_regs[j];
                }
            }
        }
        __syncthreads();
    }

    // --- Store results to C ---
    #pragma unroll
    for (int i = 0; i < THREAD_WORK_ROWS; ++i) {
        int g_row = base_row_c + i;
        if (g_row < M && base_col_c < N) {
             if (base_col_c + THREAD_WORK_COLS <= N) {
                 *((float4*)&C[g_row * N + base_col_c]) = make_float4(c_vals[i][0], c_vals[i][1], c_vals[i][2], c_vals[i][3]);
             } else {
                #pragma unroll
                for (int j = 0; j < THREAD_WORK_COLS; ++j) {
                    if (base_col_c + j < N) C[g_row * N + base_col_c + j] = c_vals[i][j];
                }
            }
        }
    }
}


// C++ wrapper function that interfaces with PyTorch.
torch::Tensor matmul_cuda_wrapper(const torch::Tensor& A, const torch::Tensor& B) {
    // Input validation checks.
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch for matmul");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on a CUDA device");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Tensors must be of type float32");

    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();

    const int M = A_cont.size(0);
    const int K = A_cont.size(1);
    const int N = B_cont.size(1);

    // Vectorized kernel requires K and N to be multiples of 4 for vectorized access.
    TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4 for vectorized kernel");
    TORCH_CHECK(N % 4 == 0, "N must be a multiple of 4 for vectorized kernel");

    auto C = torch::empty({M, N}, A_cont.options());

    // Configure kernel launch parameters.
    const int TILE_DIM = 32;
    // A block uses 8x8 threads.
    const dim3 threadsPerBlock(BLOCK_COLS, BLOCK_ROWS, 1);
    const dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM, 1);

    // Launch the CUDA kernel.
    matmul_kernel_simplified_vectorized<<<numBlocks, threadsPerBlock>>>(
        A_cont.data_ptr<float>(),
        B_cont.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    return C;
}
'''

    cpp_src = "torch::Tensor matmul_cuda_wrapper(const torch::Tensor& A, const torch::Tensor& B);"

    # Use load_inline to compile the CUDA C++ code into a loadable module.
    matmul_cuda_module = load_inline(
        name='matmul_cuda_module_db_vec',
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=['matmul_cuda_wrapper'],
        verbose=False,
    )
except Exception as e:
    # If compilation fails, fall back to the original torch.matmul to ensure functionality.
    print(f"CUDA kernel compilation failed: {e}. Falling back to torch.matmul.")
    matmul_cuda_module = None


class ModelNew(nn.Module):
    '''
    Optimized model that performs a single matrix multiplication (C = A * B)
    using a custom CUDA kernel with shared memory tiling. If the kernel fails
    to compile, it gracefully falls back to the standard torch.matmul.
    '''
    def __init__(self):
        super(ModelNew, self).__init__()
        self.use_custom_kernel = matmul_cuda_module is not None

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        '''
        Performs matrix multiplication of A and B. It attempts to use the custom
        CUDA kernel for performance, otherwise falls back to torch.matmul.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        '''
        # Use the custom kernel only if it's compiled, inputs are on CUDA, and dimensions are compatible for vectorization.
        # This provides a graceful fallback to the default PyTorch implementation.
        if (self.use_custom_kernel and
            A.is_cuda and B.is_cuda and
            A.shape[1] % 4 == 0 and B.shape[1] % 4 == 0):
            return matmul_cuda_module.matmul_cuda_wrapper(A, B)
        else:
            return torch.matmul(A, B)