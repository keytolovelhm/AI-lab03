"""MixUp and CutMix with soft-target losses whose weights match the realised mixing area."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def beta_sample(alpha: float, generator: torch.Generator) -> float:
    # Beta.sample has no generator argument. Preserve Dropout's global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        value = float(torch.distributions.Beta(alpha, alpha).sample())
        generator.set_state(torch.get_rng_state())
    return value


def mixup_batch(images: torch.Tensor, alpha: float, generator: torch.Generator):
    """Returns (mixed_images, permutation, lam). lam is the weight of the ORIGINAL label."""
    lam = beta_sample(alpha, generator)
    permutation = torch.randperm(images.shape[0], generator=generator)
    mixed = lam * images + (1.0 - lam) * images[permutation]
    return mixed, permutation, lam


def cutmix_batch(images: torch.Tensor, alpha: float, generator: torch.Generator):
    """CutMix with lam recomputed from the realised patch area (so labels match pixels)."""
    height, width = images.shape[2], images.shape[3]
    lam = beta_sample(alpha, generator)
    ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h = int(round(height * ratio))
    cut_w = int(round(width * ratio))
    cut_h = min(cut_h, height - 1)
    cut_w = min(cut_w, width - 1)
    top = int(torch.randint(0, height - cut_h + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - cut_w + 1, (1,), generator=generator).item())
    permutation = torch.randperm(images.shape[0], generator=generator)
    mixed = images.clone()
    mixed[:, :, top:top + cut_h, left:left + cut_w] = images[permutation, :, top:top + cut_h, left:left + cut_w]
    realised_lam = 1.0 - (cut_h * cut_w) / (height * width)
    return mixed, permutation, realised_lam, {"top": top, "left": left, "height": cut_h, "width": cut_w}


def apply_mixing(images: torch.Tensor, kind: str, alpha: float, generator: torch.Generator | None):
    """Uniform interface used by the trainer. Returns (images, permutation|None, lam, meta)."""
    if kind == "none" or generator is None:
        return images, None, 1.0, {}
    if kind == "mixup":
        mixed, permutation, lam = mixup_batch(images, alpha, generator)
        return mixed, permutation, lam, {"alpha": alpha}
    if kind == "cutmix":
        mixed, permutation, lam, box = cutmix_batch(images, alpha, generator)
        return mixed, permutation, lam, {"alpha": alpha, **box}
    raise ValueError(f"unknown mixing kind {kind!r}")


def soft_target_loss(logits: torch.Tensor, targets: torch.Tensor, permutation, lam: float,
                     label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross entropy on mixup/cutmix batches with optional label smoothing on both parts."""
    loss = lam * F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    if permutation is None:
        return loss
    return loss + (1.0 - lam) * F.cross_entropy(logits, targets[permutation], label_smoothing=label_smoothing)
