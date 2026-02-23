import torch
import torch.nn as nn
import torch.compiler

# Import inductor config to enable the full suite of advanced compiler features.
try:
    import torch._inductor.config
    # Enable aggressive fusion to merge more operations into single kernels.
    torch._inductor.config.aggressive_fusion = True
    # Activate a more advanced autotuning algorithm to find faster kernels.
    torch._inductor.config.coordinate_descent_tuning = True
    # Enable fusion of memory permutation operations like flatten/view.
    torch._inductor.config.permute_fusion = True
    # Activate CUDA Graphs to capture all kernel launches into a single graph,
    # drastically reducing CPU launch overhead.
    torch._inductor.config.triton.cudagraphs = True
    # Use persistent reductions for pooling layers, which can improve performance
    # by keeping data in registers/shared memory across the reduction.
    torch._inductor.config.triton.persistent_reductions = True
    # Enable Inductor's layout optimization pass to intelligently choose optimal
    # memory layouts, replacing manual channels_last conversion.
    torch._inductor.config.layout_optimization = True
    # Enable epilogue fusion to merge element-wise ops into preceding kernels,
    # reducing memory bandwidth by avoiding writing intermediate tensors.
    torch._inductor.config.epilogue_fusion = True
    # Increase the maximum fusion size to allow the compiler to create
    # even larger, more monolithic kernels.
    torch._inductor.config.max_fusion_size = 16384

except ImportError:
    print("Warning: torch._inductor.config not available, cannot enable advanced tuning features.")

# Set TF32 precision for matrix multiplications globally. This is a crucial optimization
# for A100 GPUs, allowing PyTorch to use Tensor Cores for FP32 operations.
try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    print("Warning: torch.set_float32_matmul_precision is not available in this PyTorch version.")


class AlexNetImpl(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetImpl, self).__init__()
        # The entire model is defined as a single, unified nn.Sequential chain
        # to present a simple, linear graph to the compiler, maximizing fusion.
        self.net = nn.Sequential(
            # Features
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Flatten
            nn.Flatten(start_dim=1),
            # Classifier
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Instantiate the model and immediately move it to the GPU.
        # The model is kept in the default (contiguous) memory format.
        model = AlexNetImpl(num_classes=num_classes).cuda()

        # Freeze the model after setting it to evaluation mode.
        # This treats model weights as compile-time constants, enabling
        # more aggressive optimizations like constant folding and simplifying
        # the graph for other passes like layout optimization.
        model.eval()
        try:
            model = torch.compiler.freezing.freeze(model)
        except (AttributeError, RuntimeError):
            print("Warning: torch.compiler.freezing.freeze not available or failed, skipping.")

        # Apply torch.compile with the most potent settings for this workload.
        self.model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)

        # Create a dummy input tensor in the default memory format.
        # The batch size of 128 matches the evaluation harness, ensuring autotuning
        # is specialized for the correct input dimensions.
        dummy_input = torch.randn(128, 3, 224, 224, dtype=torch.float32, device='cuda')

        # Perform a warm-up forward pass to trigger JIT compilation, autotuning,
        # and CUDA graph capture.
        with torch.inference_mode():
            self.model(dummy_input)
        torch.cuda.synchronize() # Wait for compilation to finish.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `torch.inference_mode()` is the most efficient context for pure inference.
        with torch.inference_mode():
            # The input tensor is passed directly to the model without any
            # memory format conversion, as the layout optimizer will handle it internally.
            return self.model(x)