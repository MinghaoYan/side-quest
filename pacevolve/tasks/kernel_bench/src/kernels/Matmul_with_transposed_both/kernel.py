# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define M, K, N for context, though the kernel is generic.
M = 1024
K = 4096
N = 2048

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> // For WMMA

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == torch::kHalf, #x " must be a half tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x)

// --- WMMA Configuration ---
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// --- Kernel Configuration ---
// Block tile size
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32

// Threads per block.
// Each block has (BLOCK_M/WMMA_M) * (BLOCK_N/WMMA_N) warps = (64/16)*(64/16) = 4*4 = 16 warps.
// 16 warps * 32 threads/warp = 512 threads.
// We arrange threads in 16 rows and 32 columns.
#define THREADS_X 32
#define THREADS_Y 16

using namespace nvcuda;

__global__ void wmma_matmul_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // --- Block and Warp Identification ---
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int warpId = threadIdx.y * (THREADS_X / 32) + threadIdx.x / 32;

    // --- Shared Memory Allocation ---
    __shared__ __half As[BLOCK_M][BLOCK_K];
    __shared__ __half Bs[BLOCK_K][BLOCK_N];

    // --- WMMA Fragment Initialization ---
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // --- Main Loop over K dimension ---
    for (int k_base = 0; k_base < K; k_base += BLOCK_K) {
        // --- Load Global to Shared Memory ---
        // Each of the 512 threads loads 4 halfs (one float-worth) for A and B tiles.
        const int thread_id = threadIdx.y * THREADS_X + threadIdx.x;
        const int items_per_thread = 4; // 4 halfs == sizeof(float)

        // Load A tile (BLOCK_M x BLOCK_K = 64x32)
        int a_load_offset = thread_id * items_per_thread;
        int a_row = a_load_offset / BLOCK_K;
        int a_col = a_load_offset % BLOCK_K;
        int g_a_row = block_row * BLOCK_M + a_row;
        int g_a_col = k_base + a_col;

        if (g_a_row < M) {
             *reinterpret_cast<float*>(&As[a_row][a_col]) = *reinterpret_cast<const float*>(&A[g_a_row * K + g_a_col]);
        } else {
             // Pad with zeros if out of M bounds
             *reinterpret_cast<float*>(&As[a_row][a_col]) = 0.0f;
        }

        // Load B tile (BLOCK_K x BLOCK_N = 32x64)
        int b_load_offset = thread_id * items_per_thread;
        int b_row = b_load_offset / BLOCK_N;
        int b_col = b_load_offset % BLOCK_N;
        int g_b_row = k_base + b_row;
        int g_b_col = block_col * BLOCK_N + b_col;

        if (g_b_col + items_per_thread <= N) {
            *reinterpret_cast<float*>(&Bs[b_row][b_col]) = *reinterpret_cast<const float*>(&B[g_b_row * N + g_b_col]);
        } else {
            // Slower path for boundary conditions along N dimension
            for(int i=0; i < items_per_thread; ++i) {
                if (g_b_col + i < N) {
                    (reinterpret_cast<__half*>(&Bs[b_row][b_col]))[i] = (reinterpret_cast<const __half*>(&B[g_b_row*N + g_b_col]))[i];
                } else {
                    (reinterpret_cast<__half*>(&Bs[b_row][b_col]))[i] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();

        // --- Inner Loop for WMMA computation ---
        #pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner += WMMA_K) {
            const int warp_row_in_block = warpId / (BLOCK_N / WMMA_N);
            const int warp_col_in_block = warpId % (BLOCK_N / WMMA_N);
            const int a_row_start = warp_row_in_block * WMMA_M;
            const int b_col_start = warp_col_in_block * WMMA_N;

            wmma::load_matrix_sync(a_frag, &As[a_row_start][k_inner], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &Bs[k_inner][b_col_start], BLOCK_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    // --- Store Result from Fragments to Global Memory ---
    const int warp_row_in_block = warpId / (BLOCK_N / WMMA_N);
    const int warp_col_in_block = warpId % (BLOCK_N / WMMA_N);

    const int c_row_start = block_row * BLOCK_M + warp_row_in_block * WMMA_M;
    const int c_col_start = block_col * BLOCK_N + warp_col_in_block * WMMA_N;

    // Boundary check before storing the entire 16x16 fragment
    if (c_row_start < M && c_col_start < N) {
        wmma::store_matrix_sync(&C[c_row_start * N + c_col_start], acc_frag, N, wmma::mem_row_major);
    }
}


torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);

    TORCH_CHECK(K == K_B, "Matrices have incompatible dimensions");
    TORCH_CHECK(K % WMMA_K == 0, "K dimension must be a multiple of 16 for this WMMA kernel");

    // Output is float32, as accumulated in the kernel
    auto options = torch::TensorOptions().device(A.device()).dtype(torch::kFloat);
    auto C = torch::zeros({M, N}, options);

    dim3 dimBlock(THREADS_X, THREADS_Y);
    dim3 dimGrid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    wmma_matmul_kernel<<<dimGrid, dimBlock>>>(
        (const __half*)A.data_ptr<at::Half>(),
        (const __half*)B.data_ptr<at::Half>(),
        C.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return C;
}

"""

CPP_SOURCE = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Load the inline extension
matmul_cuda_module = load_inline(
    name='wmma_matmul_cuda_module',
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=['matmul_cuda'],
    verbose=True,
    extra_cuda_cflags=["-arch=sm_80"] # Required for wmma on A100
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 0):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda_module.matmul_cuda

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Convert inputs to FP16 for the WMMA Tensor Core kernel
        A_t = x.T.contiguous().half()
        B_t = y.T.contiguous().half()

        # The kernel computes using FP16 inputs, accumulates in FP32, and returns an FP32 tensor
        C = self.matmul_cuda(A_t, B_t)

        # Return the FP32 result
        return C
# RegexTagCustomPruningAlgorithmEnd