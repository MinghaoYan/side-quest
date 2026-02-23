# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os
import math

# Set CUDA architecture for A100-SXM4-40GB.
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# Define the custom CUDA kernel for fused Bias Add + ReLU + Dropout using half precision.
# This kernel integrates a philox-based PRNG for stateless dropout, which is fused
# with the bias addition and ReLU operations from the state-of-the-art kernel.
# Vectorization with __half2 is maintained to process 4 elements per thread.
cublas_fused_dropout_source_fp16 = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <thrust/random.h> // This header contains the philox_engine definition

// Helper device function to generate four random numbers using Philox,
// create a dropout mask for two __half2 vectors, and apply the inverted dropout.
__device__ inline void generate_dropout_mask_and_apply(
    __half2 &vec1,
    __half2 &vec2,
    unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset,
    const float p,
    const __half2 scale_vec)
{
    // Philox generates four 32-bit uints at a time, perfect for two __half2 vectors.
    thrust::philox_engine<4, 32, 10> rng(seed, subsequence, offset);
    const uint4 r = rng();

    // Convert uints to floats in [0.0, 1.0)
    const float rand1 = __uint2float_rn(r.x) / 4294967296.0f;
    const float rand2 = __uint2float_rn(r.y) / 4294967296.0f;
    const float rand3 = __uint2float_rn(r.z) / 4294967296.0f;
    const float rand4 = __uint2float_rn(r.w) / 4294967296.0f;

    // Create dropout masks. 0.0f if dropped, 1.0f if kept.
    const __half mask1 = (rand1 < p) ? __float2half(0.0f) : __float2half(1.0f);
    const __half mask2 = (rand2 < p) ? __float2half(0.0f) : __float2half(1.0f);
    const __half mask3 = (rand3 < p) ? __float2half(0.0f) : __float2half(1.0f);
    const __half mask4 = (rand4 < p) ? __float2half(0.0f) : __float2half(1.0f);

    const __half2 mask_vec1 = __halves2half2(mask1, mask2);
    const __half2 mask_vec2 = __halves2half2(mask3, mask4);

    // Apply mask and scale the kept elements (inverted dropout)
    vec1 = __hmul2(__hmul2(vec1, mask_vec1), scale_vec);
    vec2 = __hmul2(__hmul2(vec2, mask_vec2), scale_vec);
}


__global__ void add_bias_relu_dropout_kernel_fp16_vectorized(
    const __half* __restrict__ input,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int N,
    const int num_elements,
    const float p,
    const unsigned long long seed,
    const unsigned long long offset_base)
{
    const int num_vec_chunks = num_elements / 4;
    const __half zero_h = __float2half(0.0f);
    const __half2 zero_vec = __halves2half2(zero_h, zero_h);

    // Pre-calculate scaling factor for inverted dropout
    const __half scale_h = (p > 0.0f) ? __float2half(1.0f / (1.0f - p)) : __float2half(1.0f);
    const __half2 scale_vec = __halves2half2(scale_h, scale_h);

    // Grid-stride loop to process 4-element chunks
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vec_chunks;
         i += gridDim.x * blockDim.x)
    {
        const int col_start = (i * 4) % N;

        const __half2 input_vec1 = reinterpret_cast<const __half2*>(input)[i * 2];
        const __half2 input_vec2 = reinterpret_cast<const __half2*>(input)[i * 2 + 1];
        const __half2 bias_vec1 = *reinterpret_cast<const __half2*>(&bias[col_start]);
        const __half2 bias_vec2 = *reinterpret_cast<const __half2*>(&bias[col_start + 2]);

        // Fused Add + ReLU
        __half2 output_vec1 = __hmax2(__hadd2(input_vec1, bias_vec1), zero_vec);
        __half2 output_vec2 = __hmax2(__hadd2(input_vec2, bias_vec2), zero_vec);

        // Fused Dropout: only apply if p > 0
        if (p > 0.0f) {
            // 'i' provides a unique subsequence ID for each 4-element chunk
            generate_dropout_mask_and_apply(output_vec1, output_vec2, seed, i, offset_base, p, scale_vec);
        }

        reinterpret_cast<__half2*>(output)[i * 2] = output_vec1;
        reinterpret_cast<__half2*>(output)[i * 2 + 1] = output_vec2;
    }
}

torch::Tensor add_bias_relu_dropout_forward_fp16(
    torch::Tensor input,
    torch::Tensor bias,
    const float p,
    const unsigned long long seed,
    const unsigned long long offset)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Input must be a Half tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "Bias must be a Half tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input's second dimension must match bias size");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.size(1) % 4 == 0, "Vectorized kernel requires the second dimension to be a multiple of 4");

    const int M = input.size(0);
    const int N = input.size(1);
    const int num_elements = M * N;

    auto output = torch::empty_like(input);
    if (num_elements == 0) return output;

    const int block_size = 256;
    const int num_vec_chunks = num_elements / 4;
    const int grid_size = std::min((num_vec_chunks + block_size - 1) / block_size, 4096);

    add_bias_relu_dropout_kernel_fp16_vectorized<<<grid_size, block_size>>>(
        (const __half*)input.data_ptr<at::Half>(),
        (const __half*)bias.data_ptr<at::Half>(),
        (__half*)output.data_ptr<at::Half>(),
        N,
        num_elements,
        p,
        seed,
        offset
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &add_bias_relu_dropout_forward_fp16, "Fused Add Bias, ReLU, and Dropout forward (CUDA Half FP16)");
}
"""

# Compile the inline CUDA code
cublas_fused_dropout_module_fp16 = load_inline(
    name='cublas_fused_relu_dropout_fp16',
    cpp_sources=[],
    cuda_sources=[cublas_fused_dropout_source_fp16],
    verbose=True
)

# Define a custom module that fuses Linear, ReLU, and Dropout
class LinearReLUDropout(nn.Module):
    def __init__(self, in_features, out_features, p=0.5):
        super(LinearReLUDropout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        # State for the stateless PRNG
        if self.p > 0:
            self.seed = torch.randint(2**32, (1,)).item()
            self.offset = 0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        matmul_result = F.linear(input, self.weight)
        
        p = self.p if self.training else 0.0
        
        # The kernel handles p=0 efficiently (becomes a no-op)
        # We only need to provide state when dropout is active
        seed_val = 0
        offset_val = 0
        if p > 0.0:
            seed_val = self.seed
            offset_val = self.offset
            # Increment offset for the next training iteration
            self.offset += 1
            
        return cublas_fused_dropout_module_fp16.forward(
            matmul_result, self.bias, p, seed_val, offset_val
        )

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        classifier_in_features = 512 * 7 * 7
        classifier_hidden_features = 4096
        # Hardcode dropout probability for this experiment.
        self.dropout_p = 0.5

        # Classifier with custom fused Linear-ReLU-Dropout layers
        self.classifier = nn.Sequential(
            LinearReLUDropout(classifier_in_features, classifier_hidden_features, p=self.dropout_p),
            LinearReLUDropout(classifier_hidden_features, classifier_hidden_features, p=self.dropout_p),
            nn.Linear(classifier_hidden_features, num_features)
        )
        
        self.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.float()

# RegexTagCustomPruningAlgorithmEnd