from itertools import product
from math import cos, isclose, pi
from statistics import mean, stdev

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.runtime import ROOT

from .core import file_digest, read_csv, read_json, verified_training_data
from .evaluation import load_selected, official_test, predict_test, read_history, selected_checkpoints
from .summary import contrast_plan
from .training import clean_metrics
from .visualization import cam_examples, error_examples


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_budget(config, histories, output):
    decision = read_json(output / "budget_decision.json")
    initial = config["epochs"]
    window = config["loss_trend_window"]
    patience = config["patience"]
    eligible = False
    for record in decision["base"]:
        rows = histories[record["run"], record["seed"]]
        before_extension = rows[:initial]
        best = max(before_extension, key=lambda row: (row["val_accuracy"], -row["val_loss"]))
        losses = [row["val_loss"] for row in before_extension]
        delta = mean(losses[-window:]) - mean(losses[-2 * window:-window]) if len(losses) >= 2 * window else None
        same_loss = delta is None and record["last_window_val_loss_change"] is None
        if delta is not None:
            same_loss = isclose(delta, record["last_window_val_loss_change"], abs_tol=1e-12)
        trigger = len(before_extension) == initial and (
            best["epoch"] == initial or delta is not None and delta <= -config["loss_trend_threshold"]
        )
        require(same_loss and trigger == record["trigger"] and best["epoch"] == record["best_epoch"]
                and len(before_extension) == record["epochs"], f"Incorrect validation-only extension: {record}")
        eligible |= trigger
        best_pair = (-1.0, float("-inf"))
        stale = 0
        for row in rows:
            require(stale < patience, f"Training continued past early stopping: {record}")
            current = (row["val_accuracy"], -row["val_loss"])
            if current > best_pair:
                best_pair = current
                stale = 0
            else:
                stale += 1
            step = min(row["epoch"] - 1, initial)
            lr = config["min_learning_rate"] + (config["learning_rate"] - config["min_learning_rate"]) * (
                1 + cos(pi * step / initial)
            ) / 2
            require(isclose(row["lr"], lr, abs_tol=1e-12), f"Learning rate schedule differs: {record}")
        require(len(rows) == decision["shared_cap"] or stale == patience, f"Run ended before its stop condition: {record}")
    require(decision["initial_cap"] == initial and decision["shared_cap"] == (
        config["extension_epochs"] if eligible else initial
    ), "Shared budget differs from preregistered validation rule")


def verify_statistics(config, scores, output):
    results = read_csv(output / "group_summary.csv")
    require(len(results) == len(config["runs"]), "Incomplete group summary")
    for row in results:
        block = [scores[row["run"], seed] / 10000 for seed in config["seeds"]]
        require(int(row["n_seeds"]) == len(block)
                and isclose(float(row["test_mean"]), mean(block), abs_tol=1e-12)
                and isclose(float(row["test_sample_sd"]), stdev(block), abs_tol=1e-12), f"Summary differs: {row}")
    detailed = {(row["contrast"], int(row["seed"])): row for row in read_csv(output / "paired_by_seed.csv")}
    summary = {row["contrast"]: row for row in read_csv(output / "paired_summary.csv")}
    plan = contrast_plan(config)
    require(set(summary) == set(plan) and len(detailed) == len(plan) * len(config["seeds"]), "Incomplete contrast tables")
    for name, weights in plan.items():
        deltas = [sum(weight * scores[run, seed] for run, weight in weights.items()) for seed in config["seeds"]]
        for seed, delta in zip(config["seeds"], deltas):
            require(int(detailed[name, seed]["test_delta_correct"]) == delta
                    and float(detailed[name, seed]["test_delta_pp"]) == delta / 100, f"Paired delta differs: {name}/{seed}")
        signs = [1 if delta > 0 else -1 for delta in deltas if delta != 0]
        null = list(product((-1, 1), repeat=len(signs)))
        p = sum(abs(sum(candidate)) >= abs(sum(signs)) for candidate in null) / len(null)
        row = summary[name]
        require(isclose(float(row["test_mean_delta_pp"]), mean(deltas) / 100, abs_tol=1e-12)
                and isclose(float(row["test_sample_sd_delta_pp"]), stdev(deltas) / 100, abs_tol=1e-12)
                and float(row["two_sided_exact_sign_p_descriptive"]) == p, f"Paired summary differs: {name}")


def verify_visualizations(result, folder, records, dataset, samples):
    cams = read_csv(folder / "gradcam_samples.csv")
    expected_cams = cam_examples(samples)
    require([(item["selection"], int(item["test_index"])) for item in cams] == [
        (kind, row[0]) for kind, row in expected_cams
    ], f"Grad-CAM selection differs: {folder}")
    with np.load(folder / "gradcam_maps.npz", allow_pickle=False) as maps:
        require(maps["test_index"].tolist() == [row[0] for _, row in expected_cams], "Grad-CAM map indices differ")
        for size in (4, 8):
            for target in ("true", "predicted"):
                array = maps[f"{target}_{size}x{size}"]
                require(array.shape == (len(cams), size, size) and np.isfinite(array).all()
                        and np.all(array >= 0), f"Invalid {target} Grad-CAM maps: {folder}")
    errors = read_csv(folder / "error_samples.csv")
    expected_errors = error_examples(samples, result["seed"] + 40_000)
    require([(item["selection"], int(item["test_index"])) for item in errors] == [
        (kind, row[0]) for kind, row in expected_errors
    ], f"Error-sample selection differs: {folder}")
    for sample in cams + errors:
        require({key: value for key, value in sample.items() if key != "selection"} == records[int(sample["test_index"])],
                f"Figure label differs from prediction CSV: {sample}")
    for sample in errors:
        index = int(sample["test_index"])
        with Image.open(folder / "error_images" / f"{index:05d}.png") as original:
            require(np.array_equal(np.asarray(original), dataset.data[index]), "An error image changed pixel content")


def verify_suite(config, device, output):
    selected = selected_checkpoints(config, output)
    require(read_json(output / "evaluation_complete.json") == {"run_count": len(selected), "test_images_per_run": 10000},
            "Evaluation is incomplete")
    require(len(read_json(output / "preflight.json")["runs"]) == len(config["runs"]), "Preflight is incomplete")
    data = verified_training_data(config)
    split = read_json(output / "split.json")
    require(data.metadata == split, "Frozen split differs from original")
    manifest = read_json(output / "evaluation_manifest.json")
    dataset, images, labels = official_test(split)
    scores, histories = {}, {}
    for (run, seed, run_path, result), entry in zip(selected, manifest["checkpoints"], strict=True):
        name = run["name"]
        history = read_history(run_path)
        histories[name, seed] = history
        require(isclose(sum(row["seconds"] for row in history), result["training_seconds"], abs_tol=1e-9),
                f"Training time differs: {run_path}")
        checkpoint_hash = file_digest(run_path / "best.pt")
        require(entry == {"run": name, "seed": seed, "epoch": result["best_epoch"], "sha256": checkpoint_hash},
                f"Checkpoint differs from sealed selection: {run_path}")
        require(not (run_path / "resume.pt").exists(), f"Redundant resume state retained: {run_path}")
        model = load_selected(config, run, seed, device, run_path, result, split)
        loss, accuracy = clean_metrics(model, data, data.val, config, device)
        require(accuracy == result["best_val_accuracy"] and isclose(loss, result["best_val_loss"], abs_tol=1e-7),
                f"Checkpoint cannot reproduce validation metrics: {run_path}")

        logits, predicted, confidence = predict_test(model, images, labels, config, device)
        folder = run_path / "evaluation"
        records = read_csv(folder / "predictions.csv")
        classes = np.asarray(dataset.classes)
        require([int(row["test_index"]) for row in records] == list(range(10000))
                and np.array_equal([row["true_class"] for row in records], classes[labels])
                and np.array_equal([row["predicted_class"] for row in records], classes[predicted])
                and np.array_equal([float(row["confidence"]) for row in records], confidence.numpy()),
                f"Prediction CSV differs on reload: {run_path}")
        matrix = np.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
        test = read_json(folder / "test.json")
        correct = int(np.trace(matrix))
        scores[name, seed] = correct
        require(test["test_correct"] == correct and test["test_accuracy"] == correct / 10000
                and isclose(test["test_loss"], F.cross_entropy(logits, labels).item(), abs_tol=1e-8)
                and matrix.tolist() == test["confusion_matrix_rows_true_columns_predicted"]
                and test["checkpoint_sha256"] == checkpoint_hash, f"Recomputed test result differs: {run_path}")
        for i, (class_row, matrix_row) in enumerate(zip(read_csv(folder / "per_class.csv"),
                                                       read_csv(folder / "confusion_matrix.csv"), strict=True)):
            require(class_row["class"] == matrix_row["true_class"] == classes[i]
                    and int(class_row["total"]) == 1000 and int(class_row["correct"]) == matrix[i, i]
                    and float(class_row["accuracy"]) == matrix[i, i] / 1000
                    and [int(matrix_row[label]) for label in classes] == matrix[i].tolist(), "Class statistics differ")
        if seed == config["seeds"][0] and run["crop"] and run["flip"]:
            samples = [(i, y, p, c) for i, (y, p, c) in enumerate(zip(labels.tolist(), predicted.tolist(), confidence.tolist()))]
            verify_visualizations(result, folder, records, dataset, samples)
        print(f"Verified {name}/{seed} | validation {accuracy:.2%} | test {correct / 10000:.2%}", flush=True)

    verify_budget(config, histories, output)
    verify_statistics(config, scores, output)
    figures = [path for path in output.rglob("*.png") if "error_images" not in path.parts]
    require(len(figures) == len(selected) + 7, "Missing or unexpected result figures")
    for figure in figures:
        with Image.open(figure) as image:
            image.verify()
    print(f"Verified {len(scores)} checkpoints, all paired statistics and {len(figures)} figures.", flush=True)
