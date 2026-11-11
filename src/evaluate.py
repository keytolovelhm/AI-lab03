import csv
import json

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import TorontoCIFAR10, evaluation_batches, normalize_images
from .models import build_model
from .plots import comparison_plot, confusion_plot, error_grid, gradcam_grid, learning_curves
from .runtime import precision_context, write_csv, write_json


def load_model(config, run, device, output):
    name = run["name"]
    folder = output / "results" / name
    summary = json.loads((folder / "training.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output / "checkpoints" / name / "best.pt", map_location="cpu", weights_only=True)
    for key, expected in {"run": run, "config": config, "split_sha256": summary["split_sha256"],
                          "best_epoch": summary["best_epoch"]}.items():
        if checkpoint[key] != expected or summary[key] != expected:
            raise ValueError(f"Inconsistent checkpoint and training metadata for {name}: {key}")
    model = build_model(config, run, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, summary, checkpoint["normalization"]


@torch.inference_mode()
def predict(model, images, labels, config, device):
    scores = []
    for batch, _ in evaluation_batches(images, labels, torch.arange(len(labels)), config["batch_size"]):
        with precision_context(config, device):
            scores.append(model(batch.to(device, memory_format=torch.channels_last)).float().cpu())
    logits = torch.cat(scores)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite test logits")
    confidence, prediction = logits.softmax(dim=1).max(dim=1)
    return logits, prediction, confidence


def sample_row(sample, classes):
    index, truth, prediction, confidence = sample
    return {"test_index": index, "true_class": classes[truth], "predicted_class": classes[prediction],
            "confidence": confidence}


def evaluate_model(model, summary, config, dataset, images, labels, device, folder):
    logits, prediction, confidence = predict(model, images, labels, config, device)
    classes = dataset.classes
    matrix = torch.bincount(labels * 10 + prediction, minlength=100).reshape(10, 10).numpy()
    per_class = [
        {"class": label, "correct": int(row[i]), "total": int(row.sum()), "accuracy": float(row[i] / row.sum())}
        for i, (label, row) in enumerate(zip(classes, matrix))
    ]
    samples = [(i, y, p, c) for i, (y, p, c) in enumerate(zip(labels.tolist(), prediction.tolist(), confidence.tolist()))]
    errors = sorted((row for row in samples if row[1] != row[2]), key=lambda row: (-row[3], row[0]))
    selected = []
    for label in range(len(classes)):
        selected.extend([row for row in errors if row[1] == label][:2])
    name = summary["run"]["name"]
    test = {"run": name, "test_images": len(labels), "test_correct": int(matrix.trace()),
            "test_accuracy": float(matrix.trace() / matrix.sum()), "test_loss": F.cross_entropy(logits, labels).item(),
            "test_errors": len(errors), "classes": classes,
            "confusion_matrix_rows_true_columns_predicted": matrix.tolist(), "per_class": per_class}
    write_json(folder / "test.json", test)
    write_csv(folder / "per_class.csv", per_class)
    write_csv(folder / "predictions.csv", [sample_row(row, classes) for row in samples])
    write_csv(folder / "confusion_matrix.csv", [
        {"true_class": label, **dict(zip(classes, matrix[i].tolist()))} for i, label in enumerate(classes)
    ])
    confusion_plot(matrix, classes, name, folder / "confusion_matrix.png")
    error_grid(dataset.data, selected, classes, name, folder / "misclassified.png")
    if selected:
        write_csv(folder / "misclassified.csv", [sample_row(row, classes) for row in selected])
        raw_folder = folder / "misclassified_images"
        raw_folder.mkdir(exist_ok=True)
        for index, _, _, _ in selected:
            Image.fromarray(dataset.data[index]).save(raw_folder / f"{index:05d}.png")
    with (folder / "history.csv").open(newline="", encoding="utf-8") as stream:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    learning_curves(history, summary, folder / "learning_curves.png")
    print(f"{name} | test {test['test_accuracy']:.2%} ({test['test_correct']}/{len(labels)})", flush=True)
    return test, samples, errors


def gradcam(model, images, prediction, config, device):
    images = images.to(device, memory_format=torch.channels_last)
    with precision_context(config, device):
        features = model.features(images)
        logits = model.classifier(features)
    if not torch.equal(logits.argmax(dim=1).cpu(), prediction):
        raise ValueError("Grad-CAM prediction differs from the test prediction")
    target = logits.gather(1, prediction.to(device)[:, None]).sum()
    gradient, = torch.autograd.grad(target, features)
    weights = gradient.float().mean(dim=(2, 3), keepdim=True)
    maps = (features.float() * weights).sum(dim=1, keepdim=True).relu()
    maps = F.interpolate(maps, size=(32, 32), mode="bilinear", align_corners=False)
    maximum = maps.amax(dim=(2, 3), keepdim=True)
    return (maps / maximum.clamp_min(1e-12))[:, 0].detach().cpu().numpy()


def evaluate_suite(config, runs, device, data_dir, output):
    loaded = [load_model(config, run, device, output) for run in runs]
    metadata = loaded[0][2]
    if any(item[2] != metadata for item in loaded):
        raise ValueError("All evaluated checkpoints must use the same training split and normalization")
    winner = max(loaded, key=lambda item: (item[1]["best_val_accuracy"], -item[1]["best_val_loss"]))[1]["run"]["name"]

    # The official test images and labels are loaded only after all training is complete.
    dataset = TorontoCIFAR10(data_dir, train=False, download=True)
    labels = torch.tensor(dataset.targets)
    if dataset.classes != metadata["classes"] or torch.bincount(labels).tolist() != [1000] * 10:
        raise ValueError("Unexpected CIFAR-10 test classes or counts")
    images = normalize_images(dataset.data, metadata)
    rows = []
    for model, summary, _ in loaded:
        run = summary["run"]
        folder = output / "results" / run["name"]
        test, samples, errors = evaluate_model(model, summary, config, dataset, images, labels, device, folder)
        rows.append({"run": run["name"], "residual": run["residual"], "augment": run["augment"],
                     "parameters": summary["parameters"], "epochs_completed": summary["epochs_completed"],
                     "best_epoch": summary["best_epoch"], "best_val_accuracy": summary["best_val_accuracy"],
                     "test_accuracy": test["test_accuracy"], "test_correct": test["test_correct"],
                     "test_images": test["test_images"], "training_seconds": summary["training_seconds"],
                     "validation_selected": run["name"] == winner})
        if run["name"] == winner:
            correct = sorted((row for row in samples if row[1] == row[2]), key=lambda row: (-row[3], row[0]))
            cam_samples = diverse_samples(correct) + diverse_samples(errors)
            indices = [row[0] for row in cam_samples]
            targets = torch.tensor([row[2] for row in cam_samples])
            cams = gradcam(model, images[indices].permute(0, 3, 1, 2), targets, config, device)
            gradcam_grid(dataset.data, cam_samples, cams, dataset.classes, winner, folder / "gradcam.png")
            write_csv(folder / "gradcam_samples.csv", [sample_row(row, dataset.classes) for row in cam_samples])
    result_dir = output / "results"
    write_csv(result_dir / "comparison.csv", rows)
    write_json(result_dir / "comparison.json", {"validation_selected": winner, "runs": rows})
    comparison_plot(rows, result_dir / "comparison.png")


def diverse_samples(samples, count=4):
    selected, classes = [], set()
    for row in samples:
        if row[1] not in classes:
            selected.append(row)
            classes.add(row[1])
            if len(selected) == count:
                break
    return selected
