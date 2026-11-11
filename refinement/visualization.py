from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.evaluate import sample_row
from src.plots import plt, save_figure
from src.runtime import write_csv, write_json


SAMPLE_FIELDS = ("selection", "test_index", "true_class", "predicted_class", "confidence")


def distinct_classes(rows, limit):
    result, classes = [], set()
    for row in rows:
        if row[1] not in classes:
            classes.add(row[1])
            result.append(row)
            if len(result) == limit:
                break
    return result


def cam_examples(samples):
    by_high = sorted(samples, key=lambda row: (-row[3], row[0]))
    correct = distinct_classes((row for row in by_high if row[1] == row[2]), 2)
    high_error = distinct_classes((row for row in by_high if row[1] != row[2]), 2)
    used = {row[0] for row in correct + high_error}
    by_low = sorted((row for row in samples if row[0] not in used), key=lambda row: (row[3], row[0]))
    low = distinct_classes(by_low, 2)
    return [(kind, row) for kind, group in (("Correct", correct), ("Low confidence", low),
                                            ("High confidence error", high_error)) for row in group]


def gradcam_maps(model, images, target_classes, layer_index, device):
    """Return native positive Grad-CAM maps for explicit target class IDs."""
    model.eval()
    activations = images.to(device, memory_format=torch.channels_last)
    with torch.no_grad():
        for layer in model.features[:layer_index]:
            activations = layer(activations)
        activations = model.features[layer_index](activations)
    activations = activations.detach().requires_grad_()
    features = model.features[layer_index + 1:](activations)
    logits = model.classifier(features)
    target = logits.gather(1, target_classes.to(device).unsqueeze(1)).sum()
    gradient, = torch.autograd.grad(target, activations)
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    maps = (activations * weights).sum(dim=1).relu().detach().cpu()
    if not torch.isfinite(maps).all():
        raise ValueError("Nonfinite Grad-CAM activation map")
    return maps


def upsample_maps(native):
    maps = F.interpolate(native[:, None], size=(128, 128), mode="bicubic", align_corners=False)[:, 0].clamp_min(0)
    return (maps / maps.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)).numpy()


def show_overlay(ax, image, heatmap):
    extent = (0, 1, 1, 0)
    ax.imshow(image, interpolation="nearest", extent=extent)
    if heatmap is not None:
        ax.imshow(heatmap, cmap="magma", interpolation="nearest", extent=extent, vmin=0, vmax=1,
                  alpha=0.68 * heatmap**1.35)
    ax.set_axis_off()


def save_gradcam(model, dataset, images, samples, run_name, device, folder):
    examples = cam_examples(samples)
    if not examples:
        raise ValueError("No Grad-CAM samples")
    indices = [row[0] for _, row in examples]
    truths = torch.tensor([row[1] for _, row in examples])
    predictions = torch.tensor([row[2] for _, row in examples])
    inputs = images[indices].permute(0, 3, 1, 2)
    native = {}
    for size, layer in ((4, 5), (8, 3)):
        for label, classes in (("predicted", predictions), ("true", truths)):
            maps = gradcam_maps(model, inputs, classes, layer, device)
            if maps.shape != (len(indices), size, size):
                raise ValueError(f"Expected {size}x{size} activations at feature layer {layer}")
            native[f"{label}_{size}x{size}"] = maps
    np.savez_compressed(folder / "gradcam_maps.npz", test_index=np.asarray(indices),
                        **{key: value.numpy() for key, value in native.items()})
    write_csv(folder / "gradcam_samples.csv", [
        {"selection": kind, **sample_row(row, dataset.classes)} for kind, row in examples
    ], SAMPLE_FIELDS)

    overlay = {key: upsample_maps(value) for key, value in native.items()}
    fig, axes = plt.subplots(len(examples), 5, figsize=(10.6, 2 * len(examples) + 0.9),
                             layout="constrained", squeeze=False)
    for ax, title in zip(axes[0], ("Original", "Predicted · 4×4", "Predicted · 8×8", "True · 4×4", "True · 8×8")):
        ax.set_title(title, fontsize=11, pad=14)
    for i, (kind, (index, truth, pred, confidence)) in enumerate(examples):
        show_overlay(axes[i, 0], dataset.data[index], None)
        axes[i, 0].text(0, -0.03, f"{kind} · #{index}\n{dataset.classes[truth]} → {dataset.classes[pred]} · {confidence:.2%}",
                        transform=axes[i, 0].transAxes, va="top", fontsize=7.7)
        keys = ("predicted_4x4", "predicted_8x8", "true_4x4", "true_8x8")
        for column, key in enumerate(keys, 1):
            show_overlay(axes[i, column], dataset.data[index], overlay[key][i])
    fig.suptitle(f"{run_name} | Grad-CAM at two feature resolutions\n"
                 "FP32 gradients · bicubic interpolation · per-map positive normalization", fontsize=13)
    save_figure(fig, folder / "gradcam_4x4_vs_8x8.png")


def error_examples(samples, seed):
    errors = [row for row in samples if row[1] != row[2]]
    high = sorted(errors, key=lambda row: (-row[3], row[0]))[:8]
    high_ids = {row[0] for row in high}
    pool = [row for row in errors if row[0] not in high_ids]
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(pool), min(len(pool), 8), replace=False) if pool else []
    return [("High confidence", row) for row in high] + [("Random", pool[position]) for position in positions]


def save_errors(dataset, samples, run_name, seed, folder):
    examples = error_examples(samples, seed)
    raw = folder / "error_images"
    raw.mkdir(exist_ok=True)
    for _, row in examples:
        Image.fromarray(dataset.data[row[0]]).save(raw / f"{row[0]:05d}.png")
    write_csv(folder / "error_samples.csv", [
        {"selection": kind, **sample_row(row, dataset.classes)} for kind, row in examples
    ], SAMPLE_FIELDS)
    write_json(folder / "error_sampling.json", {"seed": seed, "random_pool": "all errors except the high-confidence panel",
                                                "random_draw": "uniform without replacement"})
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6), layout="constrained")
    by_selection = defaultdict(list)
    for kind, row in examples:
        by_selection[kind].append(row)
    for kind, panels in (("High confidence", axes[:2]), ("Random", axes[2:])):
        for i, ax in enumerate(panels.flat):
            ax.set_axis_off()
            if i < len(by_selection[kind]):
                index, truth, pred, confidence = by_selection[kind][i]
                ax.imshow(dataset.data[index], interpolation="nearest")
                ax.set_title(f"{kind} · #{index} · {confidence:.2%}\n"
                             f"{dataset.classes[truth]} → {dataset.classes[pred]}", fontsize=8)
    fig.suptitle(f"{run_name} | High-confidence and randomly sampled test errors", fontsize=13)
    save_figure(fig, folder / "test_errors.png")


def paired_plot(runs, groups, config, path):
    lookup = {(row["run"], row["seed"]): row["test_accuracy"] * 100 for row in runs}
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    labels = [row["run"] for row in groups]
    positions = np.arange(len(labels))
    for seed, marker, color in zip(config["seeds"], ("o", "s", "^"), ("#0072B2", "#B34B00", "#007A58")):
        ax.plot(positions, [lookup[name, seed] for name in labels], marker=marker, color=color,
                label=str(seed), linewidth=1, alpha=0.72)
    ax.errorbar(positions, [100 * row["test_mean"] for row in groups],
                yerr=[100 * row["test_sample_sd"] for row in groups], fmt="D", color="#222222",
                capsize=4, label="Mean ± sample SD")
    ax.set_xticks(positions, labels, rotation=17, ha="right")
    ax.set(ylabel="Test accuracy (%)", title="CIFAR-10 | Three paired training seeds, shared test set")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    save_figure(fig, path)
