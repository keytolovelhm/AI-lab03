"""Implement the fixed perturbation protocol (see perturbations.PROTOCOL).

Operates on *normalised* image tensors (B,3,32,32).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .perturbations import PROTOCOL


def gaussian_noise(images: torch.Tensor, level: int, generator: torch.Generator) -> torch.Tensor:
    sigma = PROTOCOL["perturbations"]["gaussian_noise"]["sigma"][level - 1]
    return images + torch.randn(images.shape, generator=generator, dtype=images.dtype) * sigma


def gaussian_blur(images: torch.Tensor, level: int) -> torch.Tensor:
    spec = PROTOCOL["perturbations"]["gaussian_blur"]
    kernel, sigma = spec["kernel"][level - 1], spec["sigma"][level - 1]
    radius = (kernel - 1) / 2
    coordinates = torch.arange(kernel, dtype=torch.float32) - radius
    line = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    line = line / line.sum()
    kernel_2d = torch.outer(line, line).view(1, 1, kernel, kernel)
    weight = kernel_2d.repeat(images.shape[1], 1, 1, 1)
    return F.conv2d(images, weight, padding=kernel // 2, groups=images.shape[1])


def brightness(images: torch.Tensor, level: int) -> torch.Tensor:
    delta = PROTOCOL["perturbations"]["brightness"]["delta"][level - 1]
    return images + delta


def contrast(images: torch.Tensor, level: int) -> torch.Tensor:
    factor = PROTOCOL["perturbations"]["contrast"]["factor"][level - 1]
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return (images - mean) * factor + mean


def occlusion(images: torch.Tensor, level: int) -> torch.Tensor:
    size = PROTOCOL["perturbations"]["occlusion"]["size"][level - 1]
    height, width = images.shape[2], images.shape[3]
    top = (height - size) // 2
    left = (width - size) // 2
    fill = PROTOCOL["control"]["value"]
    out = images.clone()
    out[:, :, top:top + size, left:left + size] = fill
    return out


def uniform_grey(images: torch.Tensor, level: int) -> torch.Tensor:
    return torch.full_like(images, PROTOCOL["control"]["value"])


def apply(images: torch.Tensor, name: str, level: int, generator: torch.Generator | None = None) -> torch.Tensor:
    if name == "clean":
        return images
    if name == "uniform_grey":
        return uniform_grey(images, level)
    if name == "gaussian_noise":
        if generator is None:
            raise ValueError("gaussian noise requires a generator")
        return gaussian_noise(images, level, generator)
    if name == "gaussian_blur":
        return gaussian_blur(images, level)
    if name == "brightness":
        return brightness(images, level)
    if name == "contrast":
        return contrast(images, level)
    if name == "occlusion":
        return occlusion(images, level)
    raise ValueError(f"unknown perturbation {name!r}")


def summarise() -> dict:
    """Describe every perturbation/level pair for reporting."""
    rows = []
    for name, spec in PROTOCOL["perturbations"].items():
        for level in range(1, 6):
            key = next(k for k in spec if k not in ("kind",))
            rows.append({"perturbation": name, "kind": spec["kind"], "level": level,
                         "parameter": spec[key][level - 1]})
    return {"benchmark": PROTOCOL["benchmark"], "rows": rows}
