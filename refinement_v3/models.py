"""Stage-3 model zoo: attention variants, stems, and MobileNetV3-Small transfer.

The base architecture must stay bit-identical to the frozen stage-1/2 ``src/models.py``
so that the ``none`` configuration reproduces the reference parameter count 228,634 and
the same-seed initialisation. ``tests/test_modules.py`` asserts both.
"""

from __future__ import annotations

import torch
from torch import nn

REFERENCE_PARAMS = 228_634


def conv_bn(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class SE(nn.Module):
    """Squeeze-and-Excitation (Hu et al. 2018): GAP -> FC(r) -> ReLU -> FC(c) -> sigmoid."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        descriptor = self.pool(x).view(b, c)
        weight = self.gate(self.fc2(self.act(self.fc1(descriptor)))).view(b, c, 1, 1)
        return x * weight


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018): channel then spatial."""

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        average = self.fc2(self.act(self.fc1(x.mean(dim=(2, 3)))))
        peak = self.fc2(self.act(self.fc1(x.amax(dim=(2, 3)))))
        return torch.sigmoid(average + peak).view(b, c, 1, 1)

    def spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.spatial(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


ATTENTIONS = {"none": lambda c: nn.Identity(), "se": SE, "cbam": CBAM}


class ConvStage(nn.Module):
    """Conv transition + two 3x3 convs, with optional identity residual and attention."""

    def __init__(self, in_channels: int, out_channels: int, residual: bool, attention: str = "none"):
        super().__init__()
        self.transition = nn.Sequential(conv_bn(in_channels, out_channels, 1), nn.ReLU(inplace=True))
        self.branch = nn.Sequential(
            conv_bn(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            conv_bn(out_channels, out_channels, 3),
        )
        self.attention = ATTENTIONS[attention](out_channels)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.transition(x)
        features = self.attention(self.branch(shortcut))
        if self.residual:
            features = features + shortcut
        return torch.relu(features)


class StageNet(nn.Module):
    """Stage-1/2 CNN with configurable stem and attention.

    stem="pixel_unshuffle": identical to src.models.CifarCNN (bit-exact parameter set).
    stem="conv": 3x3 stride-2 conv + BN + ReLU producing 16x16x12, same spatial size and
    channel count as PixelUnshuffle, so the downstream stack is unchanged.
    """

    def __init__(self, widths=(24, 48, 96), dropout=0.15, residual=False, attention="none", stem="pixel_unshuffle"):
        super().__init__()
        if stem == "pixel_unshuffle":
            # Kept inside ``features`` exactly like src.models.CifarCNN so the parameter keys are
            # identical to the frozen stage-1/2 checkpoints (features.0 = PixelUnshuffle(2)).
            self.features = nn.Sequential(nn.PixelUnshuffle(2))
            stem_channels = 12
        elif stem == "conv":
            self.features = nn.Sequential()
            stem_channels = 12
        else:
            raise ValueError(f"unknown stem {stem!r}")
        self.stem_name = stem
        self.features.extend([
            *( [nn.Sequential(nn.Conv2d(3, 12, 3, stride=2, padding=1, bias=False),
                              nn.BatchNorm2d(12), nn.ReLU(inplace=True))] if stem == "conv" else [] ),
            ConvStage(stem_channels, widths[0], residual, attention),
            nn.MaxPool2d(2),
            ConvStage(widths[0], widths[1], residual, attention),
            nn.MaxPool2d(2),
            ConvStage(widths[1], widths[2], residual, attention),
        ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[2] * 4, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def remap_legacy_state_dict(state: dict) -> dict:
    """Accept the two refactor-era layouts and map them onto the current definition.

    layout A (stage blocks at features.0/2/4, PixelUnshuffle as a list entry) -> shift to 1/3/5
    layout B (conv stem under ``stem.*``, stage blocks still at features.0/2/4) -> fold the stem
        into features.0 and shift the blocks to 1/3/5
    Trained weights are never rewritten; only the keys are renamed in memory.
    """
    if any(key.startswith("stem.") for key in state):
        mapping = {"stem": "features.0", "features.0": "features.1", "features.2": "features.3",
                   "features.4": "features.5"}
    elif "features.0.transition.0.0.weight" in state:
        mapping = {"features.0": "features.1", "features.2": "features.3", "features.4": "features.5"}
    else:
        return state
    remapped = {}
    for key, value in state.items():
        for prefix in sorted(mapping, key=len, reverse=True):
            if key.startswith(prefix + "."):
                key = mapping[prefix] + key[len(prefix):]
                break
        if key in remapped:
            raise ValueError(f"Legacy key collision: {key}")
        remapped[key] = value
    return remapped


def load_compatible(model: nn.Module, state: dict) -> nn.Module:
    model.load_state_dict(remap_legacy_state_dict(state), strict=True)
    return model


def build_stage_model(config: dict, run: dict) -> nn.Module:
    return StageNet(widths=tuple(config["widths"]), dropout=config["dropout"],
                    residual=run.get("residual", False), attention=run.get("attention", "none"),
                    stem=run.get("stem", "pixel_unshuffle"))


class MobileNetV3SmallCifar(nn.Module):
    """torchvision MobileNetV3-Small retargeted to 10 classes for 96x96 CIFAR input."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        backbone = mobilenet_v3_small(weights=None, width_mult=width_mult)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(backbone.classifier[0].in_features, 1024, bias=True),
                                        nn.Hardswish(inplace=True), nn.Dropout(0.2, inplace=True),
                                        nn.Linear(1024, num_classes, bias=True))
        self.feature_dim = backbone.classifier[0].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


def load_pretrained_backbone(model: MobileNetV3SmallCifar, state_dict: dict) -> dict:
    """Load ImageNet weights into features; everything else stays randomly initialised."""
    missing, unexpected = model.features.load_state_dict(state_dict, strict=True)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def set_trainable(model: MobileNetV3SmallCifar, mode: str) -> dict:
    """mode: 'frozen' (features strictly frozen, BN kept in eval), 'full', 'last2', 'last4'.

    For fine-tune modes the unfrozen trailing blocks are trained *including* their BatchNorm,
    which keeps the parameter ordering frozen < last2 < last4 < full unambiguous.
    """
    spans = {"frozen": len(model.features), "last2": 11, "last4": 9, "full": 0}
    if mode not in spans:
        raise ValueError(f"unknown trainable mode {mode!r}")
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in list(model.features)[spans[mode]:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return {
        "mode": mode,
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
        "unfrozen_blocks": [i for i, b in enumerate(model.features)
                            if any(p.requires_grad for p in b.parameters())],
    }


def freeze_bn_for_mode(model: MobileNetV3SmallCifar, mode: str) -> None:
    """'frozen' keeps BN in eval mode; fine-tuning modes leave BN in train mode."""
    if mode != "frozen":
        return
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
