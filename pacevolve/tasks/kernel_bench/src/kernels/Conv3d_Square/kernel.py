# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        # Adhering to the specified __init__ signature by hardcoding parameters.
        # `num_features` is interpreted as both in_channels and out_channels.
        self.in_channels = num_features
        self.out_channels = num_features
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = 1
        self.bias_enabled = False # Hardcoding bias to False for simplicity

        # Kernel shape: (out_channels, in_channels/groups, kD, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(
                self.out_channels,
                self.in_channels // self.groups,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size
            )
        )
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.randn(self.out_channels))
        else:
            self.register_parameter('bias', None)

        # JIT compilation of the CUDA kernel
        self.conv3d_cuda_op = self._load_cuda_kernel()

    def _load_cuda_kernel(self):
        cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Macro for checking CUDA errors
#define CUDA_CHECK(expr)                                                       \\
  do {                                                                         \\
    cudaError_t err = (expr);                                                  \\
    if (err != cudaSuccess) {                                                  \\
      const char* err_str = cudaGetErrorString(err);                           \\
      throw std::runtime_error(std::string("CUDA Error: ") + err_str);         \\
    }                                                                          \\
  } while (0)

__global__ void conv3d_direct_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int KD, int KH, int KW,
    int SD, int SH, int SW,
    int PD, int PH, int PW,
    int DD, int DH, int DW,
    int G) {

    // Calculate the total number of output elements
    long long output_size = (long long)N * C_out * D_out * H_out * W_out;
    // Calculate the global thread index
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx >= output_size) {
        return;
    }

    // Decompose 1D index `idx` to 5D output coordinates (n, c_out, d_out, h_out, w_out)
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (W_out * H_out * D_out * C_out);

    // Group convolution parameters
    int C_in_per_group = C_in / G;
    int C_out_per_group = C_out / G;
    int group_idx = c_out / C_out_per_group;
    int c_in_start = group_idx * C_in_per_group;

    // Initialize accumulator
    float acc = 0.0f;
    if (bias != nullptr) {
        acc = bias[c_out];
    }

    // Perform convolution
    for (int c_in_g = 0; c_in_g < C_in_per_group; ++c_in_g) {
        int c_in = c_in_start + c_in_g;
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    // Calculate input coordinates
                    int d_in = d_out * SD - PD + kd * DD;
                    int h_in = h_out * SH - PH + kh * DH;
                    int w_in = w_out * SW - PW + kw * DW;

                    // Boundary check for input
                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // Calculate flat indices for input and weight tensors
                        long long input_idx = (long long)n * C_in * D_in * H_in * W_in +
                                              (long long)c_in * D_in * H_in * W_in +
                                              (long long)d_in * H_in * W_in +
                                              (long long)h_in * W_in +
                                              w_in;

                        long long weight_idx = (long long)c_out * C_in_per_group * KD * KH * KW +
                                               (long long)c_in_g * KD * KH * KW +
                                               (long long)kd * KH * KW +
                                               (long long)kh * KW +
                                               kw;

                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Store the result
    output[idx] = acc;
}


torch::Tensor conv3d_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on a CUDA device");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on a CUDA device");
        TORCH_CHECK(bias.is_contiguous(), "Bias tensor must be contiguous");
    }

    // Get tensor dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(0);
    const int KD = weight.size(2);
    const int KH = weight.size(3);
    const int KW = weight.size(4);

    // Get convolution parameters
    const int SD = stride, SH = stride, SW = stride;
    const int PD = padding, PH = padding, PW = padding;
    const int DD = dilation, DH = dilation, DW = dilation;
    const int G = groups;

    // Calculate output dimensions
    const int D_out = (D_in + 2 * PD - DD * (KD - 1) - 1) / SD + 1;
    const int H_out = (H_in + 2 * PH - DH * (KH - 1) - 1) / SH + 1;
    const int W_out = (W_in + 2 * PW - DW * (KW - 1) - 1) / SW + 1;

    // Create output tensor
    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    long long output_size = (long long)N * C_out * D_out * H_out * W_out;
    if (output_size == 0) {
        return output;
    }

    // Configure and launch the kernel
    const int threads_per_block = 256;
    const int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv3d_direct_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        DD, DH, DW,
        G
    );

    CUDA_CHECK(cudaGetLastError());

    return output;
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv3d_forward_cuda, "3D Convolution forward (CUDA)");
}
"""
        # Set a unique build directory to avoid conflicts if multiple runs happen
        build_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions", f"conv3d_cuda_{os.getpid()}")
        os.makedirs(build_dir, exist_ok=True)
        
        op_module = load_inline(
            name='conv3d_cuda_op',
            cpp_sources='',
            cuda_sources=cuda_source,
            functions=['forward'],
            with_cuda=True,
            extra_cuda_cflags=['-O3'],
            build_directory=build_dir,
            verbose=False 
        )
        return op_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom CUDA kernel.
        """
        return self.conv3d_cuda_op.forward(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
# RegexTagCustomPruningAlgorithmEnd