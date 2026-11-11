"""Figures for stage 3 (matplotlib Agg only)."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def cam_grid(images, samples, kinds, maps, name, path):
    columns = ("Original", "Predicted 4x4", "Predicted 8x8", "True 4x4", "True 8x8")
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(10.5, 2.05 * len(samples) + 0.8),
                             layout="constrained", squeeze=False)
    for ax, title in zip(axes[0], columns):
        ax.set_title(title, fontsize=10)
    for row, (kind, sample) in enumerate(zip(kinds, samples)):
        index = int(sample["test_index"])
        axes[row, 0].imshow(_raw_image(images[index]))
        axes[row, 0].set_axis_off()
        axes[row, 0].text(0, -0.04, f"{kind} #{index}\n{sample['true_class']} -> {sample['predicted_class']} "
                                    f"{float(sample['confidence']):.2%}", transform=axes[row, 0].transAxes,
                          fontsize=7.2, va="top")
        for column, key in enumerate(("predicted_4x4", "predicted_8x8", "true_4x4", "true_8x8"), start=1):
            heat = maps[key][row]
            axes[row, column].imshow(_raw_image(images[index]))
            axes[row, column].imshow(heat, cmap="magma", alpha=0.65 * heat ** 1.3, vmin=0, vmax=1)
            axes[row, column].set_axis_off()
    fig.suptitle(f"{name} | positive Grad-CAM at 4x4 (last) and 8x8 (middle) feature maps", fontsize=12)
    _save(fig, path)


def attention_grid(model, images, samples, path):
    import torch

    layers = [layer for layer in model.features if hasattr(layer, "attention")
              and not isinstance(layer.attention, torch.nn.Identity)]
    if not layers:
        return
    fig, axes = plt.subplots(len(samples), len(layers) + 1, figsize=(2.3 * (len(layers) + 1), 2.3 * len(samples) + 0.6),
                             layout="constrained", squeeze=False)
    inputs = torch.from_numpy(images.numpy()).permute(0, 3, 1, 2).clone()
    for column in range(len(layers) + 1):
        axes[0, column].set_title("image" if column == 0 else f"stage {column}", fontsize=9)
    for row, sample in enumerate(samples):
        axes[row, 0].imshow(_raw_image(images[row]))
        axes[row, 0].set_axis_off()
        for column, layer in enumerate(layers, start=1):
            attention = layer.attention
            gathered = {}
            handle = attention.register_forward_hook(lambda m, i, o: gathered.update(inp=i[0]))
            model(inputs)
            handle.remove()
            source = gathered["inp"][row:row + 1].detach()
            if hasattr(attention, "spatial"):
                with torch.no_grad():
                    channel = attention.channel_attention(source)
                    spatial = attention.spatial_attention(source * channel)[0, 0]
                axes[row, column].imshow(_raw_image(images[row]))
                axes[row, column].imshow(spatial.detach().numpy(), cmap="viridis", alpha=0.55)
            else:
                with torch.no_grad():
                    descriptor = source.mean(dim=(2, 3))
                    gate = torch.sigmoid(attention.fc2(attention.act(attention.fc1(descriptor))))[0]
                axes[row, column].bar(range(len(gate)), gate.detach().numpy(), color="#0072B2")
                axes[row, column].set_ylim(0, 1)
                axes[row, column].set_xticks([])
            axes[row, column].set_axis_off()
    fig.suptitle("attention maps: CBAM spatial attention or SE channel gates", fontsize=11)
    _save(fig, path)


def reliability(points, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, log in zip(axes, (False, True)):
        if log:
            ax.plot(points["confidence"], points["accuracy"], "o-", color="#B34B00", label="empirical")
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.set_yscale("symlog", linthresh=1e-3)
        else:
            ax.plot([0, 1], [0, 1], "--", color="#555555", label="perfect")
            ax.plot(points["confidence"], points["accuracy"], "o-", color="#B34B00", label="empirical")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set(xlabel="mean confidence", ylabel="empirical accuracy", title="log-log" if log else "linear")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(title)
    _save(fig, path)


def robustness_curves(rows, title, path):
    models = sorted({r["model"] for r in rows})
    perturbations = [p for p in dict.fromkeys(r["perturbation"] for r in rows) if p != "clean"]
    fig, axes = plt.subplots(1, len(perturbations), figsize=(3.3 * len(perturbations), 3.6), layout="constrained",
                             squeeze=False)
    colours = {m: plt.cm.tab10(i % 10) for i, m in enumerate(models)}
    for ax, perturbation in zip(axes[0], perturbations):
        for model in models:
            block = sorted((r for r in rows if r["model"] == model and r["perturbation"] == perturbation),
                           key=lambda r: int(r["level"]))
            ax.plot([int(r["level"]) for r in block], [float(r["accuracy_mean"]) * 100 for r in block],
                    "o-", color=colours[model], label=model, markersize=3)
        ax.set_title(perturbation, fontsize=9)
        ax.set_xlabel("level")
        ax.grid(alpha=0.25)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[0][0].set_ylabel("test accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title)
    _save(fig, path)


def paired_seeds(groups, lookup, seeds, path):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    names = [g["run"] for g in groups]
    positions = np.arange(len(names))
    for seed, marker in zip(seeds, ("o", "s", "^")):
        ax.plot(positions, [lookup[name, seed] * 100 for name in names], marker=marker, linewidth=1,
                alpha=0.7, label=str(seed))
    ax.errorbar(positions, [g["test_mean"] * 100 for g in groups], yerr=[g["test_sd"] * 100 for g in groups],
                fmt="D", color="#222222", capsize=4, label="Mean 卤 SD")
    ax.set_xticks(positions, names, rotation=20, ha="right")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    ax.set_title("stage-3 configurations, three paired seeds, shared frozen test set")
    _save(fig, path)


def learning_curves(history, name, best_epoch, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    epochs = [int(float(r["epoch"])) for r in history]
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(epochs, [float(r[f"train_{metric}"]) for r in history], label="train (eval mode)")
        ax.plot(epochs, [float(r[f"val_{metric}"]) for r in history], "--", label="validation")
        ax.axvline(best_epoch, color="#555555", ls=":", label="selected epoch")
        ax.set(xlabel="epoch", ylabel=metric)
        ax.grid(alpha=0.25)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False)
    fig.suptitle(name)
    _save(fig, path)



def _raw_image(image):
    from .common import ROOT,read_json
    meta=read_json(ROOT/'results/split.json')
    return (image.detach().cpu().numpy()*np.array(meta['train_std'])+np.array(meta['train_mean'])).clip(0,1)
