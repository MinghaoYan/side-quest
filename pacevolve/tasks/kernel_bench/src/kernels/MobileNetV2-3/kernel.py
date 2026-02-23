# RegexTagCustomPruningAlgorithmStart
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os
import sys
import warnings

# Suppress warnings and verbose output from the JIT compiler for a clean output
os.environ['TORCH_JIT_LOG_LEVEL'] = 'CRITICAL'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Use a temporary directory for compilation artifacts
build_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions_mobilenet")
os.makedirs(build_dir, exist_ok=True)

# Redirect stdout to suppress compiler messages
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Define the C++ and CUDA source code for the custom kernel
cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA kernel launcher
void fused_conv_bn_relu6_launcher(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

// C++ wrapper to handle logic and launch the kernel
torch::Tensor fused_conv_bn_relu6_forward(
    const torch::Tensor& input,
    const torch::Tensor& conv_w,
    const torch::Tensor& bn_w,
    const torch::Tensor& bn_b,
    const torch::Tensor& bn_mean,
    const torch::Tensor& bn_var,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    double bn_eps)
{
    // Fuse BatchNorm parameters into Conv weights and bias on the GPU
    // Ensure all calculations are done with float to match PyTorch's default precision
    auto scale = bn_w / torch::sqrt(bn_var + static_cast<float>(bn_eps));
    auto fused_w = conv_w * scale.reshape({-1, 1, 1, 1});
    auto fused_b = bn_b - bn_mean * scale;

    // Calculate output dimensions
    const int N = input.size(0);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = conv_w.size(0);
    const int K_h = conv_w.size(2);
    const int K_w = conv_w.size(3);
    const int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    // Create the output tensor on the same device as the input
    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    // Launch the CUDA kernel
    fused_conv_bn_relu6_launcher(input, fused_w, fused_b, output, stride_h, stride_w, pad_h, pad_w);
    
    return output;
}

// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_conv_bn_relu6_forward, "Fused Conv-BN-ReLU6 forward (CUDA)");
}
"""

cuda_source = """
#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel for Fused Conv2D + Bias + ReLU6
__global__ void fused_conv_bias_relu6_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int H_out, const int W_out)
{
    // Use a grid-stride loop to ensure all elements are processed
    for (long long index = blockIdx.x * blockDim.x + threadIdx.x;
         index < (long long)N * C_out * H_out * W_out;
         index += (long long)blockDim.x * gridDim.x)
    {
        // De-flatten the 1D index to 4D coordinates
        const int w_out = index % W_out;
        const int h_out = (index / W_out) % H_out;
        const int c_out = (index / (W_out * H_out)) % C_out;
        const int n = index / (C_out * W_out * H_out);

        float acc = bias[c_out];

        // Perform the convolution
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    const int h_in = -pad_h + h_out * stride_h + kh;
                    const int w_in = -pad_w + w_out * stride_w + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        const long long input_idx = (long long)n * C_in * H_in * W_in +
                                              (long long)c_in * H_in * W_in +
                                              (long long)h_in * W_in + w_in;
                        const long long weight_idx = (long long)c_out * C_in * K_h * K_w +
                                               (long long)c_in * K_h * K_w +
                                               (long long)kh * K_w + kw;
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        // Apply ReLU6 activation and store the result
        output[index] = fminf(fmaxf(0.0f, acc), 6.0f);
    }
}

// Launcher function to set up grid and block dimensions and call the kernel
void fused_conv_bn_relu6_launcher(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    // Get tensor dimensions for the kernel
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);
    const int H_out = output.size(2);
    const int W_out = output.size(3);
    
    const long long total_elements = output.numel();
    if (total_elements == 0) return;

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int num_blocks = std::min((int)((total_elements + threads_per_block - 1) / threads_per_block), 65535);
    
    fused_conv_bias_relu6_kernel<<<num_blocks, threads_per_block>>>(
        input.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        H_out, W_out);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
"""

# JIT compile the CUDA extension
try:
    fused_conv_op = load_inline(
        name="fused_conv_op_mobilenet_v2",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cflags=['-O3'],
        # Removed --use_fast_math to ensure numerical correctness
        extra_cuda_cflags=['-O3'],
        build_directory=build_dir,
        verbose=False
    )
except Exception as e:
    fused_conv_op = None

# Restore stdout
sys.stdout.close()
sys.stdout = original_stdout

class ModelNew(nn.Module):
    def __init__(self, num_features: int = 1000):
        super(ModelNew, self).__init__()

        if fused_conv_op is None:
            raise RuntimeError("CUDA extension for FusedConvBnReLU6 failed to load.")

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
            [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ]

        # First layer is separated to be replaced by the custom fused kernel
        self.first_conv = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
        self.first_bn = nn.BatchNorm2d(input_channel)
        
        features = []
        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                hidden_dim = int(input_channel * t)
                block_layers = []
                if t != 1:
                    block_layers.extend([
                        nn.Conv2d(input_channel, hidden_dim, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True)
                    ])
                block_layers.extend([
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(output_channel),
                ])
                features.append(nn.Sequential(*block_layers))
                input_channel = output_channel

        # Build last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_features),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom fused kernel for the first layer
        x = fused_conv_op.forward(
            x, self.first_conv.weight,
            self.first_bn.weight, self.first_bn.bias,
            self.first_bn.running_mean, self.first_bn.running_var,
            self.first_conv.stride[0], self.first_conv.stride[1],
            self.first_conv.padding[0], self.first_conv.padding[1],
            self.first_bn.eps
        )
        
        # Pass through the rest of the model
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
# RegexTagCustomPruningAlgorithmEnd