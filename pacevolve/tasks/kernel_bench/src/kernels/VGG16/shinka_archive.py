import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        '''
        Initialize the VGG16 model with a synergistic optimization strategy,
        combining a clean, fusion-friendly graph with aggressive autotuning.
        '''
        super(ModelNew, self).__init__()

        # --- Optimizations Applied at Initialization ---
        # Opt 1: Enable cuDNN auto-tuner to find the fastest convolution algorithms.
        torch.backends.cudnn.benchmark = True

        # --- Model Architecture ---
        # A standard nn.Sequential structure is used for a clean, compiler-friendly graph.
        # Opt 2: ReLU(inplace=True) is used to reduce memory overhead.
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        # Opt 3: Use channels_last memory format for convolutions. This is generally
        # faster on NVIDIA hardware with Tensor Core support.
        self.to(memory_format=torch.channels_last)

        # Opt 4: Statically cast the compute-intensive features module to float16
        # to leverage A100's Tensor Cores for maximum throughput.
        self.features.to(dtype=torch.float16)

        # Opt 5: Holistically compile the forward pass. `max-autotune` is selected
        # as the simplified graph structure allows it to effectively find the
        # fastest kernels without being hindered by framework overhead.
        self.forward = torch.compile(self.forward, mode="max-autotune", fullgraph=True)

    def forward(self, x):
        '''
        A clean forward pass with an explicit, static mixed-precision flow.
        '''
        # --- Feature Extraction (float16, channels_last) ---
        # Input is cast to float16 and channels_last just-in-time.
        x = self.features(x.to(memory_format=torch.channels_last, dtype=torch.float16))

        # --- Classifier (float32, contiguous) ---
        # The output of flatten on a channels_last tensor is not contiguous.
        # A .contiguous() call is critical for performance before the linear layers.
        # This is combined with the cast back to float32 for numerical stability.
        x = self.classifier(x.to(torch.float32).contiguous())

        return x