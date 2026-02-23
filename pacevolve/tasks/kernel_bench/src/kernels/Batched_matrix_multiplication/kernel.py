# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 0):
        super().__init__()
        
        cpp_source = '''
        #include <torch/extension.h>
        torch::Tensor batched_matmul_wmma_cuda(torch::Tensor A, torch::Tensor B);
        '''

        cuda_source = '''
        #include <torch/extension.h>
        #include <cuda_fp16.h>
        #include <mma.h>
        #include <vector>

        // WMMA (Tensor Core) operations operate on 16x16x16 matrices
        #define WMMA_M 16
        #define WMMA_N 16
        #define WMMA_K 16

        // Block-level tile dimensions: 128x64x16 (M x N x K)
        // Each block computes a 128x64 tile of the output matrix C.
        #define BLOCK_M 128
        #define BLOCK_N 64
        #define BLOCK_K 16

        // Threads per block. 8 warps (256/32=8)
        #define THREADS_PER_BLOCK 256
        
        // Arrangement of warps in the block (8x1 grid of warps)
        #define WARPS_PER_BLOCK_M 8
        #define WARPS_PER_BLOCK_N 1

        // Padding for shared memory to avoid bank conflicts
        #define SHMEM_B_PADDING 8
        
        // Number of `half` elements per vectorized load
        #define VEC_SIZE 8

        __device__ __forceinline__ void load_a_tile_vectorized(
            const half* A_batch, 
            half* sh_A, 
            const int M, const int K,
            const int block_row_warp,
            const int k_start_idx) {
            
            // Total data to load: BLOCK_M * BLOCK_K = 128 * 16 = 2048 halfs
            // Number of vectors: 2048 / 8 = 256 vectors. Each thread loads one vector.
            const int vec_idx = threadIdx.x;
            const int linear_idx = vec_idx * VEC_SIZE;
            const int row_in_tile = linear_idx / BLOCK_K;
            const int col_in_tile = linear_idx % BLOCK_K;

            const int gmem_row = block_row_warp * BLOCK_M + row_in_tile;
            const int gmem_col = k_start_idx + col_in_tile;

            if (gmem_row < M && gmem_col + (VEC_SIZE - 1) < K) {
                // Fast path: vector is fully within bounds
                *(reinterpret_cast<float4*>(&sh_A[linear_idx])) = 
                    *(reinterpret_cast<const float4*>(&A_batch[gmem_row * K + gmem_col]));
            } else {
                // Slow path: handle boundaries element by element
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    if (gmem_row < M && (gmem_col + i) < K) {
                        sh_A[linear_idx + i] = A_batch[gmem_row * K + (gmem_col + i)];
                    } else {
                        sh_A[linear_idx + i] = __float2half(0.0f);
                    }
                }
            }
        }
        
        __device__ __forceinline__ void load_b_tile_vectorized(
            const half* B_batch,
            half* sh_B,
            const int K, const int N,
            const int block_col_warp,
            const int k_start_idx) {

            // Total data to load: BLOCK_K * BLOCK_N = 16 * 64 = 1024 halfs
            // Number of vectors: 1024 / 8 = 128 vectors. First 128 threads load.
            if (threadIdx.x < (BLOCK_K * BLOCK_N / VEC_SIZE)) {
                const int vec_idx = threadIdx.x;
                const int linear_idx = vec_idx * VEC_SIZE;
                const int row_in_tile = linear_idx / BLOCK_N;
                const int col_in_tile = linear_idx % BLOCK_N;
                
                const int gmem_row = k_start_idx + row_in_tile;
                const int gmem_col = block_col_warp * BLOCK_N + col_in_tile;

                if (gmem_row < K && gmem_col + (VEC_SIZE - 1) < N) {
                    // Fast path: vector is fully within bounds
                    *(reinterpret_cast<float4*>(&sh_B[row_in_tile * (BLOCK_N + SHMEM_B_PADDING) + col_in_tile])) =
                        *(reinterpret_cast<const float4*>(&B_batch[gmem_row * N + gmem_col]));
                } else {
                    // Slow path: handle boundaries element by element
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        if (gmem_row < K && (gmem_col + i) < N) {
                            sh_B[row_in_tile * (BLOCK_N + SHMEM_B_PADDING) + col_in_tile + i] = B_batch[gmem_row * N + (gmem_col + i)];
                        } else {
                            sh_B[row_in_tile * (BLOCK_N + SHMEM_B_PADDING) + col_in_tile + i] = __float2half(0.0f);
                        }
                    }
                }
            }
        }


        __global__ void batched_mm_wmma_kernel(
            const half *__restrict__ A,
            const half *__restrict__ B,
            half *__restrict__ C,
            const int M, const int K, const int N) {

            // --- Batch, Block, and Warp Indexing ---
            const int batch_idx = blockIdx.z;
            const int block_row_warp = blockIdx.y; 
            const int block_col_warp = blockIdx.x; 
            
            const int warp_id = threadIdx.x / 32;

            // --- Pointers for Current Batch ---
            const half *A_batch = A + batch_idx * M * K;
            const half *B_batch = B + batch_idx * K * N;
            half *C_batch = C + batch_idx * M * N;

            // --- Shared Memory Allocation (Double Buffered) ---
            __shared__ half sh_A[2][BLOCK_M * BLOCK_K];
            __shared__ half sh_B[2][BLOCK_K * (BLOCK_N + SHMEM_B_PADDING)];

            // --- WMMA Fragment Initialization ---
            // Each warp computes a 16x64 tile, which is four adjacent 16x16 fragments in the N dimension
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag[4];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                nvcuda::wmma::fill_fragment(acc_frag[i], 0.0f);
            }

            // --- Warp coordinate calculation (moved outside loop to fix scope issues) ---
            // Map 8 warps to an 8x1 grid, each computing a 16x64 output tile
            const int warp_row_in_block = (warp_id / WARPS_PER_BLOCK_N) * WMMA_M;
            const int warp_col_in_block = (warp_id % WARPS_PER_BLOCK_N) * (WMMA_N * 4); // This will be 0

            // --- Load first tile into buffer 0 ---
            load_a_tile_vectorized(A_batch, sh_A[0], M, K, block_row_warp, 0);
            load_b_tile_vectorized(B_batch, sh_B[0], K, N, block_col_warp, 0);
            __syncthreads();

            const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

            // --- Main Loop: Prefetch next tile, compute on current ---
            for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
                const int current_buffer = k_tile_idx % 2;
                const int next_buffer = (k_tile_idx + 1) % 2;
                
                // --- Prefetch Next Tile ---
                if (k_tile_idx + 1 < num_k_tiles) {
                    const int next_k_start = (k_tile_idx + 1) * BLOCK_K;
                    load_a_tile_vectorized(A_batch, sh_A[next_buffer], M, K, block_row_warp, next_k_start);
                    load_b_tile_vectorized(B_batch, sh_B[next_buffer], K, N, block_col_warp, next_k_start);
                }

                // --- Compute on Current Tile ---
                #pragma unroll
                for (int k_step = 0; k_step < BLOCK_K; k_step += WMMA_K) {
                    const half* sh_A_ptr = &sh_A[current_buffer][warp_row_in_block * BLOCK_K + k_step];
                    const half* sh_B_base_ptr = &sh_B[current_buffer][k_step * (BLOCK_N + SHMEM_B_PADDING) + warp_col_in_block];

                    nvcuda::wmma::load_matrix_sync(a_frag, sh_A_ptr, BLOCK_K);
                    
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const half* sh_B_ptr = sh_B_base_ptr + i * WMMA_N;
                        nvcuda::wmma::load_matrix_sync(b_frag[i], sh_B_ptr, BLOCK_N + SHMEM_B_PADDING);
                        nvcuda::wmma::mma_sync(acc_frag[i], a_frag, b_frag[i], acc_frag[i]);
                    }
                }
                __syncthreads();
            }

            // --- Store Result to Global Memory ---
            const int gmem_c_row_start = block_row_warp * BLOCK_M + warp_row_in_block;
            const int gmem_c_col_start = block_col_warp * BLOCK_N + warp_col_in_block;
            
            half* C_ptr = &C_batch[gmem_c_row_start * N + gmem_c_col_start];

            if (gmem_c_row_start < M) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    // store_matrix_sync handles boundary checks for the N dimension internally
                    if (gmem_c_col_start + i * WMMA_N < N) {
                         nvcuda::wmma::store_matrix_sync(C_ptr + i * WMMA_N, acc_frag[i], N, nvcuda::wmma::mem_row_major);
                    }
                }
            }
        }

        torch::Tensor batched_matmul_wmma_cuda(torch::Tensor A, torch::Tensor B) {
            TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be on CUDA devices.");
            TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "Input tensors must be of Float16 type.");
            TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Input tensors must be 3-dimensional");

            A = A.contiguous();
            B = B.contiguous();

            const int batch_size = A.size(0);
            const int M = A.size(1);
            const int K = A.size(2);
            const int N = B.size(2);

            TORCH_CHECK(B.size(0) == batch_size && B.size(1) == K, "Matrix dimensions are not compatible for multiplication.");

            auto options = torch::TensorOptions().dtype(torch::kFloat16).device(A.device());
            torch::Tensor C = torch::empty({batch_size, M, N}, options);

            dim3 threads(THREADS_PER_BLOCK, 1, 1);
            dim3 blocks((N + BLOCK_N - 1) / BLOCK_N, 
                        (M + BLOCK_M - 1) / BLOCK_M, 
                        batch_size);

            batched_mm_wmma_kernel<<<blocks, threads>>>(
                (const half*)A.data_ptr<at::Half>(),
                (const half*)B.data_ptr<at::Half>(),
                (half*)C.data_ptr<at::Half>(),
                M, K, N);
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
            }

            return C;
        }
        '''
        
        self.batched_matmul_wmma = None
        try:
            # Using a unique name to avoid conflicts
            self.batched_matmul_wmma = load_inline(
                name='batched_matmul_wmma_128x64x16',
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=['batched_matmul_wmma_cuda'],
                extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_80'],
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load custom WMMA CUDA kernel: {e}")
            print("Falling back to torch.bmm. The kernel will not be optimized.")


    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self.batched_matmul_wmma:
            A_fp16 = A.to(torch.float16)
            B_fp16 = B.to(torch.float16)
            
            C_fp16 = self.batched_matmul_wmma.batched_matmul_wmma_cuda(A_fp16, B_fp16)
            
            return C_fp16.to(torch.float32)
        else:
            return torch.bmm(A, B)
# RegexTagCustomPruningAlgorithmEnd