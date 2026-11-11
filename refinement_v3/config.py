"""Stage-3 experiment matrix. Every formal run is (configuration, seed)."""

from __future__ import annotations

import math

SEEDS = (20261105, 20261106, 20261107)

BASE = {
    "split_seed": 3407,
    "split_sha256": "1d5cb77add4233a578b7a044c08ccc5cb5cf39c8177c5dd815ace9817b8f4dde",
    "device": "cpu",
    "precision": "bf16",
    "cpu_threads": 8,
    "batch_size": 256,
    "evaluation_batch_size": 512,
    "epochs": 60,
    "extension_epochs": 80,
    "patience": 12,
    "loss_trend_window": 6,
    "loss_trend_threshold": 0.005,
    "learning_rate": 0.002,
    "min_learning_rate": 0.00005,
    "weight_decay": 0.001,
    "dropout": 0.15,
    "widths": [24, 48, 96],
}

# ---------------------------------------------------------------- configurations
# reference: the frozen stage-2 baseline (stage 3 never retrains it)
REFERENCE = {
    "name": "reference_both",
    "origin": "stage2:baseline_both",
    "residual": False, "attention": "none", "stem": "pixel_unshuffle",
    "crop": True, "flip": True, "mixing": "none", "label_smoothing": 0.0,
}

P0 = [
    {"name": "se_both", "component": "C1 SE attention", "residual": False, "attention": "se",
     "stem": "pixel_unshuffle", "crop": True, "flip": True, "mixing": "none", "label_smoothing": 0.0},
    {"name": "cbam_both", "component": "C2 CBAM attention", "residual": False, "attention": "cbam",
     "stem": "pixel_unshuffle", "crop": True, "flip": True, "mixing": "none", "label_smoothing": 0.0},
    {"name": "ls_both", "component": "C3 label smoothing 0.05", "residual": False, "attention": "none",
     "stem": "pixel_unshuffle", "crop": True, "flip": True, "mixing": "none", "label_smoothing": 0.05},
    {"name": "convstem_both", "component": "C4 conv stem + augmentation", "residual": False, "attention": "none",
     "stem": "conv", "crop": True, "flip": True, "mixing": "none", "label_smoothing": 0.0},
    {"name": "convstem_plain", "component": "C5 conv stem, no augmentation", "residual": False, "attention": "none",
     "stem": "conv", "crop": False, "flip": False, "mixing": "none", "label_smoothing": 0.0},
]

P1 = [
    {"name": "mixup_both", "component": "C6 MixUp alpha=0.2", "residual": False, "attention": "none",
     "stem": "pixel_unshuffle", "crop": True, "flip": True, "mixing": "mixup", "mix_alpha": 0.2,
     "label_smoothing": 0.0},
    {"name": "cutmix_both", "component": "C7 CutMix alpha=1.0", "residual": False, "attention": "none",
     "stem": "pixel_unshuffle", "crop": True, "flip": True, "mixing": "cutmix", "mix_alpha": 1.0,
     "label_smoothing": 0.0},
]

MOBILE = {
    "family": "mobilenet_v3_small",
    "input_size": 96,
    "normalization": "imagenet",
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 60,
    "extension_epochs": 80,
    "patience": 12,
    "weight_decay": 0.001,
    "official_weights_sha256": None,  # experimental branch requires a verified external checksum
}


def contrasts():
    """Named paired contrasts (weights over config names; ``stage2:`` keys are frozen stage-2 runs)."""
    return {
        "se_minus_reference": {"se_both": 1, "stage2:baseline_both": -1},
        "cbam_minus_reference": {"cbam_both": 1, "stage2:baseline_both": -1},
        "cbam_minus_se": {"cbam_both": 1, "se_both": -1},
        "ls_minus_reference": {"ls_both": 1, "stage2:baseline_both": -1},
        "convstem_minus_reference": {"convstem_both": 1, "stage2:baseline_both": -1},
        "convstem_plain_minus_stage2_plain": {"convstem_plain": 1, "stage2:baseline_plain": -1},
        "mixup_minus_reference": {"mixup_both": 1, "stage2:baseline_both": -1},
        "cutmix_minus_reference": {"cutmix_both": 1, "stage2:baseline_both": -1},
        "stem_x_augmentation_interaction": {"convstem_both": 1, "convstem_plain": -1,
                                            "stage2:baseline_both": -1, "stage2:baseline_plain": 1},
    }


def sign_test(differences):
    signed = [d for d in differences if d != 0]
    n = len(signed)
    if n == 0:
        return 1.0, 0
    positives = sum(1 for d in signed if d > 0)
    tail = sum(math.comb(n, k) for k in range(min(positives, n - positives) + 1)) / 2 ** n
    return min(1.0, 2 * tail), n
