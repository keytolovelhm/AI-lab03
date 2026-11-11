"""Fixed, pre-registered perturbation protocol for the synthetic robustness study.

These parameter tables were written BEFORE any robustness measurement and must not be
changed afterwards. The study is a *synthetic* robustness test generated from CIFAR-10;
it is explicitly NOT the CIFAR-10-C benchmark.
"""

from __future__ import annotations

BENCHMARK_LABEL = "synthetic CIFAR-10 perturbations (workbook protocol), NOT CIFAR-10-C"

LEVELS = (1, 2, 3, 4, 5)

PROTOCOL = {
    "benchmark": BENCHMARK_LABEL,
    "noise_seeds": (1001, 1002, 1003),
    "batch_size": 512,
    "protocol_version": ("v2 (2026-11-10): occlusion fill switched to grey (normalised 0.0) to avoid an "
                         "edge-removal confound, and a uniform-colour control was added; the noise/blur/"
                         "brightness/contrast levels of v1 were never reported and stay unchanged."),
    # Control condition: a uniform grey image must score ~10 % on every model.
    "control": {"kind": "uniform_grey_constant_image", "value": 0.0},
    "perturbations": {
        "gaussian_noise": {
            "kind": "additive_white_gaussian_in_normalized_space",
            "sigma": [0.02, 0.05, 0.10, 0.20, 0.35],
        },
        "gaussian_blur": {
            "kind": "gaussian_filter_antialiased",
            "kernel": [3, 5, 7, 9, 11],
            "sigma": [0.5, 1.0, 1.5, 2.0, 2.5],
        },
        "brightness": {
            "kind": "additive_brightness_shift_in_normalized_space",
            "delta": [-0.10, -0.20, -0.30, 0.20, 0.35],
        },
        "contrast": {
            "kind": "blend_with_per_image_channel_mean",
            "factor": [0.85, 0.65, 0.45, 1.25, 1.60],
        },
        "occlusion": {
            "kind": "centered_square_occlusion_filled_with_grey_0_0",
            "size": [4, 8, 12, 16, 24],
        },
    },
}


def protocol_sha_ready() -> dict:
    return PROTOCOL
