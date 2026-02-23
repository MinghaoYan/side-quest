import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Set CUDA architecture for A100 GPU (Compute Capability 8.0)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

# CUDA and C++ source code for the fused Pointwise Convolution -> BatchNorm -> ReLU6 kernel
fused_conv_bn_relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <string>

// CUDA Kernel for Fused PointwiseConv + BatchNorm + ReLU6 with batch support
__global__ void fused_pointwise_conv_bn_relu6_kernel(
    const float* input, 
    const float* weight, 
    const float* bn_weight, // gamma
    const float* bn_bias,   // beta
    const float* bn_mean,
    const float* bn_var,
    float* output, 
    int batch_size,
    int in_channels, 
    int out_channels, 
    int height, 
    int width,
    float bn_eps) 
{
    // Linear index for the output tensor element (n, oc, h, w)
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total_elements = (long long)batch_size * out_channels * height * width;

    if (idx < total_elements) {
        // Decompose linear index to multi-dimensional index
        const int h_w = height * width;
        const int oc_h_w = out_channels * h_w;

        const int n = idx / oc_h_w;
        const int rem = idx % oc_h_w;
        const int oc = rem / h_w;
        const int hw = rem % h_w;
        const int h = hw / width;
        const int w = hw % width;

        // --- Step 1: Pointwise Convolution ---
        float sum = 0.0f;
        const float* input_n_slice = input + n * in_channels * h_w;
        const float* weight_oc_slice = weight + oc * in_channels;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            // Index into input tensor: input[n][ic][h][w]
            int input_idx = ic * h_w + h * width + w;
            sum += input_n_slice[input_idx] * weight_oc_slice[ic];
        }
        
        // --- Step 2: BatchNorm ---
        // Fuse BatchNorm parameters: y = gamma * (x - mean) / sqrt(var + eps) + beta
        // This is equivalent to: y = x * scale + bias
        const float scale = bn_weight[oc] * rsqrtf(bn_var[oc] + bn_eps);
        const float bias = bn_bias[oc] - bn_mean[oc] * scale;
        const float bn_out = sum * scale + bias;
        
        // --- Step 3: ReLU6 ---
        // Equivalent to min(max(0, x), 6)
        output[idx] = fminf(fmaxf(0.0f, bn_out), 6.0f);
    }
}

// C++ wrapper function to be called from Python
torch::Tensor fused_conv_bn_relu6_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bn_weight, 
    torch::Tensor bn_bias, 
    torch::Tensor bn_mean, 
    torch::Tensor bn_var,
    double bn_eps) 
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");

    // Get tensor dimensions
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_channels = weight.size(0);

    // Create the output tensor
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    // Configure kernel launch parameters
    const long long total_elements = (long long)batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Launch the CUDA kernel
    fused_pointwise_conv_bn_relu6_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        static_cast<float>(bn_eps)
    );
    
    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    return output;
}
"""

# Define the C++ source for the function signature
fused_conv_bn_relu6_cpp_source = """
torch::Tensor fused_conv_bn_relu6_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bn_weight, 
    torch::Tensor bn_bias, 
    torch::Tensor bn_mean, 
    torch::Tensor bn_var,
    double bn_eps);
"""

# Use torch.utils.cpp_extension.load_inline to compile the CUDA code
fused_conv_op = load_inline(
    name='fused_conv_op_v2',
    cpp_sources=fused_conv_bn_relu6_cpp_source,
    cuda_sources=fused_conv_bn_relu6_source,
    functions=['fused_conv_bn_relu6_cuda'],
    verbose=True
)


class FusedPointwiseConvBNReLU6(nn.Module):
    """
    Custom module that replaces Conv2d(1x1) + BatchNorm2d + ReLU6 with a fused kernel.
    Assumes inference mode.
    """
    def __init__(self, inp, oup):
        super(FusedPointwiseConvBNReLU6, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_conv_op.fused_conv_bn_relu6_cuda(
            x,
            self.conv.weight,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )


class ModelNew(nn.Module):
    def __init__(self, num_features: int = 1000):
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(round(inp * expand_ratio))
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Replace the standard Conv-BN-ReLU6 sequence with our fused operator
                layers.append(FusedPointwiseConvBNReLU6(inp, hidden_dim))
            
            # Depthwise and projection layers
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            # Return a tuple to match the baseline's signature
            return nn.Sequential(*layers), use_res_connect

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                # The [0] is crucial to match the baseline's behavior of not using the residual connection
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)[0])
                input_channel = output_channel

        # Building last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.0), # Matched to baseline
            nn.Linear(last_channel, num_features),
        )

        # Weight initialization
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
        x = self.features(x)
        x = x.view(x.size(0), -1) # Matched to baseline
        x = self.classifier(x)
        return x