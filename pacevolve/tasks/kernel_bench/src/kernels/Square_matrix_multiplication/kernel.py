# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

wmma_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h> // For WMMA intrinsics

using namespace nvcuda;

// WMMA fragment dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Tile dimensions matching the experiment description
#define M_TILE 64
#define N_TILE 64
#define K_TILE 16

// Threadblock dimensions
#define BLOCK_DIM_M 64
#define BLOCK_DIM_N 64
#define THREADS_PER_BLOCK 128 // 4 warps

__global__ void wmma_matmul_padded_kernel(const __half *A, const __half *B, float *C, int P) {
    // Shared memory for tiles of A and B
    __shared__ __half As[M_TILE][K_TILE];
    __shared__ __half Bs[K_TILE][N_TILE];

    // Get block and thread indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;

    // Each warp computes a 32x32 sub-tile of the output C tile.
    // There are 4 warps in a 2x2 grid within the thread block.
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    // Declare accumulator fragments for each warp. Each warp handles a 2x2 grid of 16x16 fragments.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag[2][2];

    // Initialize accumulator fragments to zero
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(C_frag[i][j], 0.0f);
        }
    }

    // Loop over tiles of the K-dimension
    for (int k_tile_idx = 0; k_tile_idx < P / K_TILE; ++k_tile_idx) {
        // --- Load one tile of A and B from global to shared memory ---
        // Cooperative loading by all threads in the block.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int linear_idx = thread_id + i * THREADS_PER_BLOCK;
            
            // Load A tile (M_TILE x K_TILE -> 64x16)
            int row_a = linear_idx / K_TILE;
            int col_a = linear_idx % K_TILE;
            if (row_a < M_TILE) {
                As[row_a][col_a] = A[(block_row * M_TILE + row_a) * P + (k_tile_idx * K_TILE + col_a)];
            }

            // Load B tile (K_TILE x N_TILE -> 16x64)
            int row_b = linear_idx / N_TILE;
            int col_b = linear_idx % N_TILE;
            if (row_b < K_TILE) {
                Bs[row_b][col_b] = B[(k_tile_idx * K_TILE + row_b) * P + (block_col * N_TILE + col_b)];
            }
        }
        __syncthreads();

        // --- Compute using the tile in shared memory ---
        // Declare fragments for A and B.
        // **FIX**: A_frag layout changed from col_major to row_major to match memory layout in As.
        // This prevents an implicit transpose during load, which was causing incorrect results.
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> A_frag[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> B_frag[2];

        // Each warp computes its 32x32 C sub-tile by iterating 2x2 times over fragments
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            // Load A fragments for the warp's row
            int a_row_start = warp_row * 32 + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &As[a_row_start][0], K_TILE);

            #pragma unroll
            for (int j = 0; j < 2; j++) {
                // Load B fragments for the warp's column
                int b_col_start = warp_col * 32 + j * WMMA_N;
                wmma::load_matrix_sync(B_frag[j], &Bs[0][b_col_start], N_TILE);

                // Perform matrix multiply-accumulate
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        __syncthreads();
    }

    // --- Write accumulated results to global memory (C matrix) ---
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int C_row = block_row * M_TILE + warp_row * 32 + i * WMMA_M;
            int C_col = block_col * N_TILE + warp_col * 32 + j * WMMA_N;
            wmma::store_matrix_sync(&C[C_row * P + C_col], C_frag[i][j], P, wmma::mem_row_major);
        }
    }
}

torch::Tensor wmma_matmul_padded_cuda(torch::Tensor A_padded, torch::Tensor B_padded) {
    TORCH_CHECK(A_padded.dim() == 2, "A must be 2D");
    TORCH_CHECK(B_padded.dim() == 2, "B must be 2D");
    TORCH_CHECK(A_padded.size(0) == B_padded.size(0), "Padded dimensions must be equal");
    TORCH_CHECK(A_padded.size(0) % M_TILE == 0, "Padded dimension must be a multiple of M_TILE");
    TORCH_CHECK(A_padded.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B_padded.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A_padded.dtype() == torch::kFloat16, "A must be a Half tensor");
    TORCH_CHECK(B_padded.dtype() == torch::kFloat16, "B must be a Half tensor");

    const int P = A_padded.size(0);
    auto C_padded = torch::zeros({P, P}, torch::dtype(torch::kFloat32).device(A_padded.device()));

    // Kernel launch configuration
    dim3 threads(THREADS_PER_BLOCK, 1, 1);
    dim3 blocks(P / N_TILE, P / M_TILE, 1);

    wmma_matmul_padded_kernel<<<blocks, threads>>>(
        (const __half*)A_padded.data_ptr<at::Half>(),
        (const __half*)B_padded.data_ptr<at::Half>(),
        C_padded.data_ptr<float>(),
        P
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C_padded;
}
"""

wmma_cpp_source = """
torch::Tensor wmma_matmul_padded_cuda(torch::Tensor A_padded, torch::Tensor B_padded);
"""

# JIT compilation of the CUDA kernel
# Requires compute capability 8.0+ (Ampere)
wmma_matmul_cuda = load_inline(
    name='wmma_matmul_cuda_padded_fixed',
    cpp_sources=wmma_cpp_source,
    cuda_sources=wmma_cuda_source,
    functions=['wmma_matmul_padded_cuda'],
    verbose=True,
    extra_cuda_cflags=['-gencode=arch=compute_80,code=sm_80']
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int = -1):
        super(ModelNew, self).__init__()
        self.matmul_cuda = wmma_matmul_cuda
        # Hard-code tile dimensions, must match #define in CUDA source
        self.tile_dim = 64 # Used for padding calculation

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a custom CUDA kernel that leverages
        WMMA intrinsics (Tensor Cores) on padded matrices.
        Args:
            A (torch.Tensor): The first input matrix (N x N), float32.
            B (torch.Tensor): The second input matrix (N x N), float32.
        Returns:
            torch.Tensor: The result of A @ B, float32.
        """
        N = A.size(0)

        # Calculate padded dimension P to be a multiple of the tile dimension
        P = ((N + self.tile_dim - 1) // self.tile_dim) * self.tile_dim

        # WMMA operates on half precision, so we convert inputs.
        # Create padded tensors directly with the target dtype.
        A_padded = torch.zeros((P, P), device=A.device, dtype=torch.float16)
        B_padded = torch.zeros((P, P), device=B.device, dtype=torch.float16)
        
        # Copy original data into the padded tensors. This also handles the type conversion.
        A_padded[:N, :N] = A
        B_padded[:N, :N] = B

        # Call the WMMA kernel
        C_padded = self.matmul_cuda.wmma_matmul_padded_cuda(A_padded, B_padded)

        # Return the unpadded slice of the result. C_padded is float32.
        return C_padded[:N, :N]
# RegexTagCustomPruningAlgorithmEnd