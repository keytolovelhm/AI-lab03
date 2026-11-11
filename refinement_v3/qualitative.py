"""Qualitative analysis: Grad-CAM at 4x4 / 8x8, SE channel gates, CBAM attention maps."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .calibration import calibration_metrics
from .common import RESULTS, SEEDS, read_csv, read_json, write_csv, write_json
from .config import BASE
from .data import load_dataset
from .train import build, configure, logits_for, run_dir


def _layers(model):
    """Return {name: (module, feature_hw)} for the stage-3 CNN (4x4 = last, 8x8 = middle)."""
    stages = [layer for layer in model.features if hasattr(layer, "transition")]
    return {"4x4": stages[2], "8x8": stages[1]}


def grad_cam_maps(model, layer, inputs, targets, image_size: int):
    """Positive Grad-CAM on an explicit layer, using a manual forward to keep grads on the activation."""
    activations = {}

    def save(module, inputs_, output):
        activations["out"] = output

    handle = layer.register_forward_hook(save)
    output = model(inputs)
    handle.remove()
    acts = activations["out"]
    logits = output.gather(1, targets[:, None]).sum()
    grad, = torch.autograd.grad(logits, acts)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    maps = (acts * weights).sum(1).relu().detach()
    maps = F.interpolate(maps[:, None], size=(image_size, image_size), mode="bilinear",
                         align_corners=False, antialias=True)[:, 0].clamp_min(0)
    maximum = maps.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return (maps / maximum).numpy()


def select_samples(predictions: list[dict]) -> list[dict]:
    correct = [r for r in predictions if r["true_class"] == r["predicted_class"]]
    errors = [r for r in predictions if r["true_class"] != r["predicted_class"]]
    high_correct = sorted(correct, key=lambda r: -float(r["confidence"]))[:2]
    low_correct = sorted(correct, key=lambda r: float(r["confidence"]))[:2]
    high_error = sorted(errors, key=lambda r: -float(r["confidence"]))[:2]
    return ([("correct_high", r) for r in high_correct] + [("correct_low", r) for r in low_correct]
            + [("error_high", r) for r in high_error])


def analyse(run_name: str, seed: int = SEEDS[0]):
    """Return verified current CAM material. Recompute all assets with evaluate --out NEW."""
    from analysis_code.final_acceptance import DEFAULT,verify_outputs
    verify_outputs(DEFAULT)
    folder=DEFAULT/'stage3'/run_name/str(seed)
    path=folder/'gradcam_4x4_vs_8x8.png'
    if not path.exists():raise FileNotFoundError(f'No predeclared qualitative panel: {path}')
    return {'figure':str(path),'samples':str(folder/'gradcam_samples.csv')}
