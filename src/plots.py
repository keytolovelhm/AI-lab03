import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


def save_figure(fig, path):
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def learning_curves(history, summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    epochs = [row["epoch"] for row in history]
    for ax, metric, label in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], color="#0072B2", label="Train")
        ax.plot(epochs, [row[f"val_{metric}"] for row in history], "--", color="#B34B00", label="Validation")
        ax.axvline(summary["best_epoch"], color="#555555", ls=":", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False)
    fig.suptitle(summary["run"]["name"])
    save_figure(fig, path)


def confusion_plot(matrix, classes, name, path):
    fig, ax = plt.subplots(figsize=(7.7, 6.7), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1000, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Images")
    ax.set_xticks(range(10), classes, rotation=38, ha="right")
    ax.set_yticks(range(10), classes)
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"{name} | CIFAR-10 test confusion matrix")
    for (row, column), count in np.ndenumerate(matrix):
        ax.text(column, row, str(count), ha="center", va="center", fontsize=8,
                color="white" if count >= 550 else "#102333")
    save_figure(fig, path)


def error_grid(raw, samples, classes, name, path):
    fig, axes = plt.subplots(4, 5, figsize=(11, 8.8), layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (index, truth, prediction, confidence) in zip(axes.flat, samples):
        ax.imshow(raw[index], interpolation="nearest")
        ax.set_title(f"#{index}  {confidence:.1%}\n{classes[truth]} → {classes[prediction]}", fontsize=9)
    fig.suptitle(f"{name} | Up to two highest-confidence errors per true class")
    save_figure(fig, path)


def gradcam_grid(raw, samples, cams, classes, name, path):
    cols = len(samples)
    fig, axes = plt.subplots(2, cols, figsize=(2 * cols, 5), squeeze=False, layout="constrained")
    for col, ((index, truth, prediction, confidence), cam) in enumerate(zip(samples, cams)):
        for ax in axes[:, col]:
            ax.imshow(raw[index], interpolation="nearest")
            ax.set_axis_off()
        axes[0, col].set_title(f"#{index}  {confidence:.1%}\n{classes[truth]} → {classes[prediction]}", fontsize=9)
        axes[1, col].imshow(cam, cmap="magma", alpha=0.55 * cam, vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].set_title("Correct" if truth == prediction else "Error", fontsize=9)
    fig.suptitle(f"{name} | Grad-CAM for predicted class\nFinal 4×4 feature map, normalized per image and interpolated")
    save_figure(fig, path)


def comparison_plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    labels = [row["run"] for row in rows]
    for ax, values, xlabel in (
        (axes[0], [row["test_accuracy"] for row in rows], "Test accuracy"),
        (axes[1], [row["training_seconds"] / 60 for row in rows], "Training + validation (min)"),
    ):
        bars = ax.barh(labels, values, color="#0072B2")
        ax.bar_label(bars, labels=[f"{value:.2%}" if ax is axes[0] else f"{value:.1f}" for value in values], padding=4)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_xlim(0, 1)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlim(0, max(row["training_seconds"] for row in rows) / 60 * 1.22)
    fig.suptitle("CIFAR-10 | Checkpoints selected by validation accuracy")
    save_figure(fig, path)
