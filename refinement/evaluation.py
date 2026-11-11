import torch
from torch.nn import functional as F

from src.data import TorontoCIFAR10, normalize_images
from src.evaluate import predict, sample_row
from src.models import build_model
from src.plots import confusion_plot, learning_curves
from src.runtime import ROOT, write_csv, write_json

from .core import file_digest, frozen_json, read_csv, read_json, run_folder, specs
from .summary import summarize
from .training import signature
from .visualization import paired_plot, save_errors, save_gradcam


def read_history(folder):
    return [{key: int(value) if key == "epoch" else float(value) for key, value in row.items()}
            for row in read_csv(folder / "history.csv")]


def selected_checkpoints(config, output):
    expected = specs(config)
    index = read_csv(output / "training_index.csv")
    if [(row["run"], int(row["seed"])) for row in index] != [(run["name"], seed) for run, seed in expected]:
        raise ValueError("All 18 training runs must finish before test evaluation")
    budget = read_json(output / "budget_decision.json")["shared_cap"]
    selected = []
    for run, seed in expected:
        folder = run_folder(output, run, seed)
        result = read_json(folder / "training.json")
        if any(result[key] != value for key, value in signature(config, run, seed).items()):
            raise ValueError(f"Training configuration differs: {folder}")
        history = read_history(folder)
        if [item["epoch"] for item in history] != list(range(1, result["epochs_completed"] + 1)):
            raise ValueError(f"Incomplete history: {folder}")
        best = max(history, key=lambda row: (row["val_accuracy"], -row["val_loss"]))
        if (best["epoch"], best["val_accuracy"], best["val_loss"], budget) != (
            result["best_epoch"], result["best_val_accuracy"], result["best_val_loss"], result["shared_epoch_cap"]
        ):
            raise ValueError(f"Invalid checkpoint selection: {folder}")
        selected.append((run, seed, folder, result))
    return selected


def load_selected(config, run, seed, device, folder, result, split):
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    identity = {**signature(config, run, seed), "normalization": split, "best_epoch": result["best_epoch"]}
    if any(checkpoint[key] != value for key, value in identity.items()):
        raise ValueError(f"Invalid checkpoint metadata: {folder}")
    model = build_model(config, run, device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def official_test(split):
    dataset = TorontoCIFAR10(ROOT / "data", train=False, download=False)
    labels = torch.tensor(dataset.targets)
    if dataset.classes != split["classes"] or torch.bincount(labels).tolist() != [1000] * 10:
        raise ValueError("Unexpected official CIFAR-10 test partition")
    return dataset, normalize_images(dataset.data, split), labels


def predict_test(model, images, labels, config, device):
    return predict(model, images, labels, {**config, "batch_size": config["evaluation_batch_size"]}, device)


def evaluate_run(model, result, config, dataset, images, labels, device, folder, visualize):
    final = folder / "evaluation"
    if final.exists():
        return read_json(final / "test.json")
    partial = folder / ".evaluation_pending"
    partial.mkdir(exist_ok=True)
    logits, predictions, confidence = predict_test(model, images, labels, config, device)
    classes = dataset.classes
    matrix = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10).numpy()
    samples = [(index, truth, pred, score) for index, (truth, pred, score) in enumerate(
        zip(labels.tolist(), predictions.tolist(), confidence.tolist())
    )]
    test = {"test_images": len(labels), "test_correct": int(matrix.trace()),
            "test_accuracy": float(matrix.trace() / len(labels)), "test_loss": F.cross_entropy(logits, labels).item(),
            "checkpoint_sha256": file_digest(folder / "best.pt"),
            "classes": classes, "confusion_matrix_rows_true_columns_predicted": matrix.tolist()}
    write_csv(partial / "predictions.csv", [sample_row(row, classes) for row in samples])
    write_csv(partial / "confusion_matrix.csv", [
        {"true_class": label, **dict(zip(classes, matrix[i].tolist()))} for i, label in enumerate(classes)
    ])
    write_csv(partial / "per_class.csv", [
        {"class": label, "correct": int(matrix[i, i]), "total": int(matrix[i].sum()),
         "accuracy": float(matrix[i, i] / matrix[i].sum())} for i, label in enumerate(classes)
    ])
    name = f"{result['run']['name']}/{result['seed']}"
    learning_curves(read_history(folder), {**result, "run": {"name": name}}, partial / "learning_curves.png")
    if visualize:
        confusion_plot(matrix, classes, name, partial / "confusion_matrix.png")
        save_gradcam(model, dataset, images, samples, name, device, partial)
        save_errors(dataset, samples, name, result["seed"] + 40_000, partial)
    write_json(partial / "test.json", test)
    partial.replace(final)
    return test


def evaluate_suite(config, device, output):
    selected = selected_checkpoints(config, output)
    split = read_json(output / "split.json")
    visual_seed = config["seeds"][0]
    manifest = {
        "selection": "validation accuracy, then lower validation loss; all runs sealed before testing",
        "qualitative_selection": {"seed": visual_seed, "crop": True, "flip": True, "architectures": ["baseline", "residual"]},
        "checkpoints": [{"run": run["name"], "seed": seed, "epoch": result["best_epoch"],
                         "sha256": file_digest(folder / "best.pt")} for run, seed, folder, result in selected],
    }
    frozen_json(output / "evaluation_manifest.json", manifest)
    if (output / "evaluation_complete.json").exists():
        print(f"Completed test results preserved in {output}", flush=True)
        return

    # Only this point opens the official test labels and image tensors.
    dataset, images, labels = official_test(split)
    rows = []
    for run, seed, folder, result in selected:
        model = load_selected(config, run, seed, device, folder, result, split)
        visualize = seed == visual_seed and run["crop"] and run["flip"]
        test = evaluate_run(model, result, config, dataset, images, labels, device, folder, visualize)
        fields = ("epochs_completed", "shared_epoch_cap", "stop_reason", "training_seconds", "best_epoch",
                  "best_val_accuracy", "best_val_loss", "parameters", "last_window_val_loss_change")
        rows.append({"run": run["name"], "seed": seed, "residual": run["residual"], "crop": run["crop"],
                     "flip": run["flip"], **{key: result[key] for key in fields},
                     **{key: test[key] for key in ("test_images", "test_correct", "test_accuracy", "test_loss")}})
        print(f"{run['name']}/{seed} | test {test['test_accuracy']:.2%} ({test['test_correct']}/10000)", flush=True)
    summary = summarize(config, rows, output)
    paired_plot(rows, summary["group_summary"], config, output / "seed_comparison.png")
    write_json(output / "evaluation_complete.json", {"run_count": len(rows), "test_images_per_run": len(labels)})
