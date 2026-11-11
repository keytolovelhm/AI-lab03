"""Representative-model robustness study for stage 3 (NEW protocol: pixel-space perturbations).

This supersedes, but does NOT replace, the retired ``PROTOCOL v2`` (which perturbed tensors after
normalisation). Numbers from the two protocols are not comparable and must not be tabulated
together.

Protocol ``pixel_v1``
---------------------
1. Start from the raw CIFAR-10 test pixels in [0, 1] (B, 3, 32, 32).
2. Apply the perturbation in pixel space.
3. Clamp back into [0, 1].
4. Normalise with the frozen stage-1 train mean/std (results/split.json).
5. Feed to the model in eval() mode, CPU, float32.

This is a synthetic perturbation study built on CIFAR-10. It is NOT CIFAR-10-C.

Models: stage-2 baseline_both / baseline_plain, stage-3 ls_both / mixup_both / cutmix_both /
convstem_both, all three seeds.

Writes refinement_v3/results/corrected/robustness/.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from refinement_v3.common import SEEDS, read_json, write_csv, write_json
from refinement_v3.config import BASE, P0, P1
from refinement_v3.data import _load_cifar_batch
from refinement_v3.models import build_stage_model
from refinement_v3.state_loader import load_by_shape
from refinement_v3.common import DATA_DIR

OUT = ROOT / "refinement_v3/results/final_verified/v1/robustness"
S3 = ROOT / "refinement_v3/runs/suite_corrected/experiments"
S2 = ROOT / "refinement/runs/suite/experiments"

LEVELS = (1, 2, 3, 4, 5)
NOISE_SEEDS = (1001, 1002, 1003)

# --------------------------------------------------------------------------- protocol
PROTOCOL = {
    "name": "pixel_v1",
    "benchmark": "synthetic CIFAR-10 perturbations, pixel space, NOT CIFAR-10-C",
    "space": "raw pixels in [0,1], perturbed then clamped to [0,1] then normalised",
    "normalisation": "frozen stage-1 train mean/std from results/split.json",
    "noise_seeds": list(NOISE_SEEDS),
    "batch_size": 512,
    "perturbations": {
        # additive white Gaussian noise, sigma in pixel units
        "gaussian_noise": {"param": "sigma_pixel", "values": [0.02, 0.05, 0.10, 0.20, 0.30],
                           "kind": "additive_white_gaussian_noise"},
        # isotropic Gaussian blur, sigma in pixels
        "gaussian_blur": {"param": "sigma_pixel", "values": [0.5, 1.0, 1.5, 2.0, 3.0],
                          "kind": "gaussian_blur_gaussian_kernel"},
        # additive brightness shift in pixel units
        "brightness": {"param": "delta_pixel", "values": [-0.10, -0.20, -0.30, 0.20, 0.35],
                       "kind": "additive_brightness_shift"},
        # contrast around the per-image mean, factor <1 flattens
        "contrast": {"param": "factor", "values": [0.85, 0.65, 0.45, 1.25, 1.60],
                     "kind": "contrast_about_image_mean"},
        # centred square occlusion filled with mid grey (0.5)
        "occlusion": {"param": "side_pixels", "values": [4, 8, 12, 16, 24],
                      "kind": "centred_square_occlusion_mid_grey"},
    },
}

CONFIGS = {run["name"]: run for run in P0 + P1}
STAGE2_RUN = {"name": "stage2_baseline", "residual": False, "attention": "none", "stem": "pixel_unshuffle",
              "crop": True, "flip": True, "mixing": "none", "label_smoothing": 0.0}
MODELS = [
    ("stage2_baseline_both", "stage2", "baseline_both", "Stage-2 crop+flip baseline"),
    ("stage2_baseline_plain", "stage2", "baseline_plain", "Stage-2 no-augmentation baseline"),
    ("ls_both", "stage3", "ls_both", "Stage-3 label smoothing 0.05"),
    ("mixup_both", "stage3", "mixup_both", "Stage-3 MixUp a=0.2"),
    ("cutmix_both", "stage3", "cutmix_both", "Stage-3 CutMix a=1.0"),
    ("convstem_both", "stage3", "convstem_both", "Stage-3 conv stem + augmentation"),
]


# --------------------------------------------------------------------------- perturbations
def gaussian_noise(x, level, generator):
    sigma = PROTOCOL["perturbations"]["gaussian_noise"]["values"][level - 1]
    return x + torch.randn(x.shape, generator=generator, dtype=x.dtype) * sigma


def gaussian_blur(x, level):
    sigma = PROTOCOL["perturbations"]["gaussian_blur"]["values"][level - 1]
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    line = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    line = line / line.sum()
    kernel = torch.outer(line, line).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    weight = kernel.repeat(x.shape[1], 1, 1, 1)
    pad = radius
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, weight, groups=x.shape[1])


def brightness(x, level):
    delta = PROTOCOL["perturbations"]["brightness"]["values"][level - 1]
    return x + delta


def contrast(x, level):
    factor = PROTOCOL["perturbations"]["contrast"]["values"][level - 1]
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * factor + m


def occlusion(x, level):
    side = PROTOCOL["perturbations"]["occlusion"]["values"][level - 1]
    h, w = x.shape[2], x.shape[3]
    top, left = (h - side) // 2, (w - side) // 2
    out = x.clone()
    out[:, :, top:top + side, left:left + side] = 0.5
    return out


def perturb(x, name, level, generator=None):
    if name == "clean":
        return x
    if name == "gaussian_noise":
        return gaussian_noise(x, level, generator)
    if name == "gaussian_blur":
        return gaussian_blur(x, level)
    if name == "brightness":
        return brightness(x, level)
    if name == "contrast":
        return contrast(x, level)
    if name == "occlusion":
        return occlusion(x, level)
    raise ValueError(name)


# --------------------------------------------------------------------------- data & models
def raw_test_pixels() -> torch.Tensor:
    """Untouched CIFAR-10 test pixels as float in [0, 1], shape (N, 3, 32, 32)."""
    data, labels = _load_cifar_batch(DATA_DIR / "cifar-10-batches-py" / "test_batch")
    x = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return x, torch.from_numpy(labels)


def normalise(x):
    meta = read_json(ROOT / "results/split.json")
    mean = torch.tensor(meta["train_mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(meta["train_std"], dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def load_any(kind: str, folder_name: str, seed: int):
    folder = (S2 if kind == "stage2" else S3) / folder_name / str(seed)
    from analysis_code.canonical import load_checkpoint
    model,ck=load_checkpoint(folder/'best.pt')
    return model


@torch.inference_mode()
def accuracy(model, pixels, labels, name, level, seed):
    generator = torch.Generator().manual_seed(seed) if name == "gaussian_noise" else None
    correct = 0
    bs = PROTOCOL["batch_size"]
    for start in range(0, len(labels), bs):
        sl = slice(start, start + bs)
        batch = perturb(pixels[sl], name, level, generator).clamp_(0.0, 1.0)
        batch = normalise(batch).to(memory_format=torch.channels_last)
        correct += int((model(batch).argmax(1) == labels[sl]).sum())
    return correct / len(labels)


def main() -> None:
    torch.set_num_threads(BASE["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    from analysis_code.canonical import output_directory
    output_directory(OUT)
    write_json(OUT / "robustness_protocol_pixel_v1.json", PROTOCOL)

    pixels, labels = raw_test_pixels()
    print(f"test pixels: {tuple(pixels.shape)} labels: {tuple(labels.shape)}")

    rows = []
    for label, kind, folder_name, description in MODELS:
        for seed in SEEDS:
            model = load_any(kind, folder_name, seed)
            clean = accuracy(model, pixels, labels, "clean", 1, seed)
            for name in PROTOCOL["perturbations"]:
                levels = [1, 2, 3, 4, 5]
                for level in levels:
                    if name == "gaussian_noise":
                        values = [accuracy(model, pixels, labels, name, level, s) for s in NOISE_SEEDS]
                        acc, sd = mean(values), stdev(values)
                        n_seeds = len(values)
                    else:
                        acc, sd, n_seeds = accuracy(model, pixels, labels, name, level, seed), 0.0, 1
                    rows.append({
                        "model": label, "description": description, "seed": seed,
                        "perturbation": name, "level": level,
                        "parameter": PROTOCOL["perturbations"][name]["values"][level - 1],
                        "clean_accuracy": clean, "perturbed_accuracy": acc,
                        "drop_pp": (clean - acc) * 100, "noise_seed_sd": sd,
                        "n_noise_seeds": n_seeds, "protocol": PROTOCOL["name"],
                    })
            print(f"  {label:<24}{seed} clean={clean:.4f} done")

    write_csv(OUT / "robustness_per_seed.csv", rows)

    # ---- grouped across seeds: mean +- SD ---------------------------------------
    grouped = []
    for label, _, _, description in MODELS:
        for name in PROTOCOL["perturbations"]:
            for level in LEVELS:
                block = [r for r in rows if r["model"] == label and r["perturbation"] == name
                         and r["level"] == level]
                pert = [r["perturbed_accuracy"] for r in block]
                drops = [r["drop_pp"] for r in block]
                grouped.append({
                    "model": label, "description": description, "perturbation": name, "level": level,
                    "parameter": PROTOCOL["perturbations"][name]["values"][level - 1],
                    "n_seeds": len(block),
                    "clean_accuracy_mean": mean(r["clean_accuracy"] for r in block),
                    "clean_accuracy_sd": stdev(r["clean_accuracy"] for r in block),
                    "perturbed_accuracy_mean": mean(pert),
                    "perturbed_accuracy_sd": stdev(pert),
                    "drop_pp_mean": mean(drops), "drop_pp_sd": stdev(drops),
                })
    write_csv(OUT / "robustness_grouped.csv", grouped)

    # ---- per-perturbation severity summary (mean drop over levels 1-5) ---------
    summary = []
    for label, _, _, description in MODELS:
        block = [r for r in grouped if r["model"] == label]
        row = {"model": label, "description": description,
               "clean_accuracy_mean": mean(r["clean_accuracy_mean"] for r in block)}
        for name in PROTOCOL["perturbations"]:
            vals = [r["drop_pp_mean"] for r in block if r["perturbation"] == name]
            absolute=[r['perturbed_accuracy_mean'] for r in block if r['perturbation']==name]
            row[f"{name}_absolute_accuracy_mean"] = mean(absolute)
            row[f"{name}_mean_drop_pp"] = mean(vals)
            row[f"{name}_worst_drop_pp"] = max(vals)
        row["mean_drop_all_pp"] = mean([r["drop_pp_mean"] for r in block])
        summary.append(row)
    write_csv(OUT / "robustness_summary.csv", summary)

    write_json(OUT / "robustness_readme.json", {
        "protocol": PROTOCOL["name"],
        "benchmark": PROTOCOL["benchmark"],
        "not_comparable_to": "retired PROTOCOL v2 (perturbations applied after normalisation); "
                             "stage-3 legacy robustness numbers were deleted with the old suite",
        "models": [m[0] for m in MODELS],
        "seeds": list(SEEDS),
        "cost_note": "all six models x three seeds completed; nothing was downscoped",
    })
    from analysis_code.canonical import sha
    sources={}
    for label,kind,name,_ in MODELS:
        for seed in SEEDS:
            path=(S2 if kind=='stage2' else S3)/name/str(seed)/'best.pt'
            sources[str(path.relative_to(ROOT))]=sha(path)
    write_json(OUT/'checkpoint_sha256.json',sources)
    write_json(OUT/'completion.json',{'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}})
    print("\nwrote", OUT)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out:OUT=args.out
    main()
