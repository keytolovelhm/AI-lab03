import torch
from torch import nn


def conv_bn(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super().__init__()
        self.transition = nn.Sequential(conv_bn(in_channels, out_channels, 1), nn.ReLU())
        self.branch = nn.Sequential(
            conv_bn(out_channels, out_channels, 3),
            nn.ReLU(),
            conv_bn(out_channels, out_channels, 3),
        )
        self.residual = residual

    def forward(self, images):
        shortcut = self.transition(images)
        features = self.branch(shortcut)
        if self.residual:
            features = features + shortcut
        return torch.relu(features)


class CifarCNN(nn.Module):
    def __init__(self, widths, dropout, residual):
        super().__init__()
        # Move each 2x2 patch into channels before expensive spatial convolution.
        self.features = nn.Sequential(
            nn.PixelUnshuffle(2),
            ConvStage(12, widths[0], residual),
            nn.MaxPool2d(2),
            ConvStage(widths[0], widths[1], residual),
            nn.MaxPool2d(2),
            ConvStage(widths[1], widths[2], residual),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[2] * 4, 10)
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(config, run, device):
    model = CifarCNN(config["widths"], config["dropout"], run["residual"])
    if device.type == "cuda":
        # Fixed 4x4 to 2x2 pooling avoids nondeterministic adaptive pooling backward on CUDA.
        model.classifier[0] = nn.AvgPool2d(2)
    return model.to(device=device, memory_format=torch.channels_last)
