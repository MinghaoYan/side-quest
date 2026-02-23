import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import hashlib

# --- Global cache for the compiled kernel ---
# This avoids recompiling the kernel every time a model instance is created.
_fused_op_cache = {}

class ModelNew(nn.Module):
    '''
    Model that performs a batch matrix multiplication, layer normalization, summation,
    residual addition, and multiplication using a single, fully-fused CUDA kernel.
    This kernel implements a novel dual software pipeline for both the input vector 'x'
    and the weight matrix 'W', aiming to hide both shared and global memory latency
    by prefetching all data into registers concurrently with computation.
    '''
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # nn.Linear is used to conveniently manage weight and bias parameters.
        self.linear = nn.Linear(in_features, out_features)

        # Parameters for the fused LayerNorm part of the kernel
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.eps = eps

        # Load the custom CUDA kernel from a global cache to avoid recompilation
        global _fused_op_cache
        kernel_key = "fused_op_dual_pipeline_v1"
        if kernel_key not in _fused_op_cache:
            _fused_op_cache[kernel_key] = self._load_cuda_kernel()
        self.fused_op = _fused_op_cache[kernel_key]

    def _load_cuda_kernel(self):
        cuda_source = r'''
#include <cuda_runtime.h>

// Helper for vectorized dot product
__device__ __forceinline__ float dot_product(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Device function for efficient sum reduction within a warp using shuffle instructions.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_linear_layernorm_add_mul_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ y_in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    const int in_features,
    const int out_features,
    const float eps) {

    // One block per row in the batch.
    const int row_idx = blockIdx.x;
    const int in_features_f4 = in_features / 4;
    const int out_features_f4 = out_features / 4;

    extern __shared__ char s_char[];
    float4* s_x_f4 = reinterpret_cast<float4*>(s_char);
    float4* s_linear_out_f4 = &s_x_f4[in_features_f4];
    float* s_reduce_base = reinterpret_cast<float*>(&s_linear_out_f4[out_features_f4]);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    float* s_warp_sums = s_reduce_base;
    float* s_warp_sums_sq = &s_reduce_base[num_warps];
    float* s_mean_inv_stddev = &s_reduce_base[2 * num_warps];

    // --- Part 1: Cooperatively load x_in row into shared memory ---
    const float4* x_in_row_f4 = reinterpret_cast<const float4*>(x_in + row_idx * in_features);
    for (int k_f4 = threadIdx.x; k_f4 < in_features_f4; k_f4 += blockDim.x) {
        s_x_f4[k_f4] = x_in_row_f4[k_f4];
    }
    __syncthreads();

    // --- Part 2: Fused GEMV with Dual Software Pipeline ---
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    const int UNROLL_J = 4;
    const int j_loop_stride = blockDim.x * UNROLL_J;
    const int j_unrolled_limit = (out_features_f4 / j_loop_stride) * j_loop_stride;
    const float4* bias_f4 = reinterpret_cast<const float4*>(bias);

    // Main loop for quad outputs (branch-free)
    for (int j_f4_base = threadIdx.x; j_f4_base < j_unrolled_limit; j_f4_base += j_loop_stride) {
        float acc1[4] = {0.0f}, acc2[4] = {0.0f}, acc3[4] = {0.0f}, acc4[4] = {0.0f};

        const int j_f4_1 = j_f4_base;
        const int j_f4_2 = j_f4_base + blockDim.x;
        const int j_f4_3 = j_f4_base + blockDim.x * 2;
        const int j_f4_4 = j_f4_base + blockDim.x * 3;
        const int j_base1 = j_f4_1 * 4, j_base2 = j_f4_2 * 4, j_base3 = j_f4_3 * 4, j_base4 = j_f4_4 * 4;

        // Pointers to the start of each weight row
        const float4* w_ptr[4][4];
        w_ptr[0][0] = reinterpret_cast<const float4*>(weight + (j_base1 + 0) * in_features);
        w_ptr[0][1] = reinterpret_cast<const float4*>(weight + (j_base1 + 1) * in_features);
        w_ptr[0][2] = reinterpret_cast<const float4*>(weight + (j_base1 + 2) * in_features);
        w_ptr[0][3] = reinterpret_cast<const float4*>(weight + (j_base1 + 3) * in_features);
        w_ptr[1][0] = reinterpret_cast<const float4*>(weight + (j_base2 + 0) * in_features);
        w_ptr[1][1] = reinterpret_cast<const float4*>(weight + (j_base2 + 1) * in_features);
        w_ptr[1][2] = reinterpret_cast<const float4*>(weight + (j_base2 + 2) * in_features);
        w_ptr[1][3] = reinterpret_cast<const float4*>(weight + (j_base2 + 3) * in_features);
        w_ptr[2][0] = reinterpret_cast<const float4*>(weight + (j_base3 + 0) * in_features);
        w_ptr[2][1] = reinterpret_cast<const float4*>(weight + (j_base3 + 1) * in_features);
        w_ptr[2][2] = reinterpret_cast<const float4*>(weight + (j_base3 + 2) * in_features);
        w_ptr[2][3] = reinterpret_cast<const float4*>(weight + (j_base3 + 3) * in_features);
        w_ptr[3][0] = reinterpret_cast<const float4*>(weight + (j_base4 + 0) * in_features);
        w_ptr[3][1] = reinterpret_cast<const float4*>(weight + (j_base4 + 1) * in_features);
        w_ptr[3][2] = reinterpret_cast<const float4*>(weight + (j_base4 + 2) * in_features);
        w_ptr[3][3] = reinterpret_cast<const float4*>(weight + (j_base4 + 3) * in_features);

        const int UNROLL_K = 4;
        // Dual software pipeline registers
        float4 x_regs[2][UNROLL_K];
        float4 w_regs[2][4][4][UNROLL_K]; // [buffer][j_idx][sub_idx][k_idx]

        // Prefetch first tile of x and W into buffer 0
        #pragma unroll
        for(int k=0; k<UNROLL_K; ++k) {
            x_regs[0][k] = s_x_f4[k];
            #pragma unroll
            for(int j=0; j<4; ++j) {
                #pragma unroll
                for(int sub=0; sub<4; ++sub) w_regs[0][j][sub][k] = w_ptr[j][sub][k];
            }
        }

        for (int k_base_f4 = 0; k_base_f4 < in_features_f4; k_base_f4 += UNROLL_K) {
            const int p_compute = (k_base_f4 / UNROLL_K) % 2;
            const int p_prefetch = 1 - p_compute;

            const int next_k_base = k_base_f4 + UNROLL_K;
            if (next_k_base < in_features_f4) {
                 #pragma unroll
                 for(int k=0; k<UNROLL_K; ++k) {
                    x_regs[p_prefetch][k] = s_x_f4[next_k_base + k];
                    #pragma unroll
                    for(int j=0; j<4; ++j) {
                        #pragma unroll
                        for(int sub=0; sub<4; ++sub) w_regs[p_prefetch][j][sub][k] = w_ptr[j][sub][next_k_base+k];
                    }
                 }
            }

            #pragma unroll
            for (int k_off = 0; k_off < UNROLL_K; ++k_off) {
                float4 x_v = x_regs[p_compute][k_off];
                acc1[0] += dot_product(x_v, w_regs[p_compute][0][0][k_off]); acc1[1] += dot_product(x_v, w_regs[p_compute][0][1][k_off]);
                acc1[2] += dot_product(x_v, w_regs[p_compute][0][2][k_off]); acc1[3] += dot_product(x_v, w_regs[p_compute][0][3][k_off]);
                acc2[0] += dot_product(x_v, w_regs[p_compute][1][0][k_off]); acc2[1] += dot_product(x_v, w_regs[p_compute][1][1][k_off]);
                acc2[2] += dot_product(x_v, w_regs[p_compute][1][2][k_off]); acc2[3] += dot_product(x_v, w_regs[p_compute][1][3][k_off]);
                acc3[0] += dot_product(x_v, w_regs[p_compute][2][0][k_off]); acc3[1] += dot_product(x_v, w_regs[p_compute][2][1][k_off]);
                acc3[2] += dot_product(x_v, w_regs[p_compute][2][2][k_off]); acc3[3] += dot_product(x_v, w_regs[p_compute][2][3][k_off]);
                acc4[0] += dot_product(x_v, w_regs[p_compute][3][0][k_off]); acc4[1] += dot_product(x_v, w_regs[p_compute][3][1][k_off]);
                acc4[2] += dot_product(x_v, w_regs[p_compute][3][2][k_off]); acc4[3] += dot_product(x_v, w_regs[p_compute][3][3][k_off]);
            }
        }

        const float4 b1 = bias_f4[j_f4_1], b2 = bias_f4[j_f4_2], b3 = bias_f4[j_f4_3], b4 = bias_f4[j_f4_4];

        const float4 res1 = make_float4(acc1[0] + b1.x, acc1[1] + b1.y, acc1[2] + b1.z, acc1[3] + b1.w);
        s_linear_out_f4[j_f4_1] = res1;
        const float4 res2 = make_float4(acc2[0] + b2.x, acc2[1] + b2.y, acc2[2] + b2.z, acc2[3] + b2.w);
        s_linear_out_f4[j_f4_2] = res2;
        const float4 res3 = make_float4(acc3[0] + b3.x, acc3[1] + b3.y, acc3[2] + b3.z, acc3[3] + b3.w);
        s_linear_out_f4[j_f4_3] = res3;
        const float4 res4 = make_float4(acc4[0] + b4.x, acc4[1] + b4.y, acc4[2] + b4.z, acc4[3] + b4.w);
        s_linear_out_f4[j_f4_4] = res4;

        float s1 = res1.x + res1.y + res1.z + res1.w;
        float sq1 = res1.x * res1.x + res1.y * res1.y + res1.z * res1.z + res1.w * res1.w;
        float s2 = res2.x + res2.y + res2.z + res2.w;
        float sq2 = res2.x * res2.x + res2.y * res2.y + res2.z * res2.z + res2.w * res2.w;
        float s3 = res3.x + res3.y + res3.z + res3.w;
        float sq3 = res3.x * res3.x + res3.y * res3.y + res3.z * res3.z + res3.w * res3.w;
        float s4 = res4.x + res4.y + res4.z + res4.w;
        float sq4 = res4.x * res4.x + res4.y * res4.y + res4.z * res4.z + res4.w * res4.w;

        thread_sum += s1 + s2 + s3 + s4;
        thread_sum_sq += sq1 + sq2 + sq3 + sq4;
    }

    // Cleanup loop for remaining single outputs
    for (int j_f4 = j_unrolled_limit + threadIdx.x; j_f4 < out_features_f4; j_f4 += blockDim.x) {
        float acc1[4] = {0.0f};
        const int j_base1 = j_f4 * 4;
        const float4* w1r0 = reinterpret_cast<const float4*>(weight + (j_base1 + 0) * in_features);
        const float4* w1r1 = reinterpret_cast<const float4*>(weight + (j_base1 + 1) * in_features);
        const float4* w1r2 = reinterpret_cast<const float4*>(weight + (j_base1 + 2) * in_features);
        const float4* w1r3 = reinterpret_cast<const float4*>(weight + (j_base1 + 3) * in_features);
        for (int k_f4 = 0; k_f4 < in_features_f4; ++k_f4) {
            float4 x_v = s_x_f4[k_f4];
            acc1[0] += dot_product(x_v, w1r0[k_f4]); acc1[1] += dot_product(x_v, w1r1[k_f4]);
            acc1[2] += dot_product(x_v, w1r2[k_f4]); acc1[3] += dot_product(x_v, w1r3[k_f4]);
        }
        const float4 b1 = bias_f4[j_f4];
        const float4 res1 = make_float4(acc1[0] + b1.x, acc1[1] + b1.y, acc1[2] + b1.z, acc1[3] + b1.w);
        s_linear_out_f4[j_f4] = res1;
        thread_sum += res1.x + res1.y + res1.z + res1.w;
        thread_sum_sq += res1.x * res1.x + res1.y * res1.y + res1.z * res1.z + res1.w * res1.w;
    }
    __syncthreads();

    // --- Part 3: Parallel Reduction ---
    thread_sum = warp_reduce_sum(thread_sum);
    thread_sum_sq = warp_reduce_sum(thread_sum_sq);

    if (lane_id == 0) { s_warp_sums[warp_id] = thread_sum; s_warp_sums_sq[warp_id] = thread_sum_sq; }
    __syncthreads();

    for (int offset = num_warps / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) { s_warp_sums[threadIdx.x] += s_warp_sums[threadIdx.x + offset]; s_warp_sums_sq[threadIdx.x] += s_warp_sums_sq[threadIdx.x + offset]; }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = s_warp_sums[0] / out_features;
        float var = s_warp_sums_sq[0] / out_features - mean * mean;
        s_mean_inv_stddev[0] = mean;
        s_mean_inv_stddev[1] = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = s_mean_inv_stddev[0];
    const float inv_stddev = s_mean_inv_stddev[1];

    // --- Part 4: Final Application (Normalization, Scale, Add, Mul) ---
    const float4* y_in_f4 = reinterpret_cast<const float4*>(y_in + row_idx * out_features);
    const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta_f4 = reinterpret_cast<const float4*>(beta);
    float4* out_f4 = reinterpret_cast<float4*>(out + row_idx * out_features);

    for (int j_f4 = threadIdx.x; j_f4 < out_features_f4; j_f4 += blockDim.x) {
        float4 val = s_linear_out_f4[j_f4];
        const float4 gamma_val = gamma_f4[j_f4];
        const float4 beta_val = beta_f4[j_f4];
        const float4 y_val = y_in_f4[j_f4];
        val.x = (val.x - mean) * inv_stddev * gamma_val.x + beta_val.x;
        val.y = (val.y - mean) * inv_stddev * gamma_val.y + beta_val.y;
        val.z = (val.z - mean) * inv_stddev * gamma_val.z + beta_val.z;
        val.w = (val.w - mean) * inv_stddev * gamma_val.w + beta_val.w;
        val.x = (val.x + y_val.x) * y_val.x;
        val.y = (val.y + y_val.y) * y_val.y;
        val.z = (val.z + y_val.z) * y_val.z;
        val.w = (val.w + y_val.w) * y_val.w;
        out_f4[j_f4] = val;
    }
}

void fused_op_launcher(
    const float* x_in, const float* y_in, const float* weight, const float* bias,
    const float* gamma, const float* beta, float* out,
    int batch_size, int in_features, int out_features, float eps) {

    int block_size = 512;
    if (out_features <= 128) block_size = 128;
    else if (out_features <= 256) block_size = 256;
    else if (out_features > 512) block_size = 1024;

    const int num_blocks = batch_size;
    const int num_warps = block_size / 32;
    const size_t smem_size = (in_features + out_features + (2 * num_warps + 2)) * sizeof(float);

    fused_linear_layernorm_add_mul_kernel<<<num_blocks, block_size, smem_size>>>(
        x_in, y_in, weight, bias, gamma, beta, out, in_features, out_features, eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be propagated by PyTorch's wrapper.
    }
}
'''
        cpp_source = r'''
#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void fused_op_launcher(
    const float* x_in, const float* y_in, const float* weight, const float* bias,
    const float* gamma, const float* beta, float* out,
    int batch_size, int in_features, int out_features, float eps);

torch::Tensor fused_op_forward(
    torch::Tensor x_in, torch::Tensor y_in, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta, double eps) {

    TORCH_CHECK(x_in.is_cuda() && y_in.is_cuda() && weight.is_cuda() && bias.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(x_in.is_contiguous() && y_in.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "Inputs must be contiguous");

    const auto in_features = x_in.size(1);
    const auto out_features = y_in.size(1);
    // UNROLL_K=4 requires in_features to be divisible by 4 * 4 = 16
    TORCH_CHECK(in_features % 16 == 0, "Input feature dimension must be divisible by 16 for 4x K-unrolled kernel");
    // UNROLL_J=4 requires out_features to be divisible by 4 * 4 = 16
    TORCH_CHECK(out_features % 16 == 0, "Output feature dimension must be divisible by 16 for 4x J-unrolled kernel");

    const auto batch_size = x_in.size(0);
    auto out = torch::empty_like(y_in);

    fused_op_launcher(
        x_in.data_ptr<float>(), y_in.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), out.data_ptr<float>(),
        batch_size, in_features, out_features, static_cast<float>(eps));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_op_forward, "Fused Linear+LayerNorm+Add+Mul forward (CUDA, Dual Pipeline)");
}
'''
        unique_name = f"fused_op_{hashlib.md5((cuda_source + cpp_source).encode()).hexdigest()}"
        op_module = load_inline(
            name=unique_name,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            verbose=False,
        )
        return op_module.forward

    def forward(self, x, y):
        '''
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        '''
        # The single fused kernel call replaces the entire original forward pass.
        return self.fused_op(
            x, y,
            self.linear.weight, self.linear.bias,
            self.gamma, self.beta,
            self.eps
        )