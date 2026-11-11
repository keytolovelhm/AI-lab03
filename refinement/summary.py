import math
from statistics import mean, stdev

from src.runtime import write_csv, write_json

from .core import read_csv, run_folder


def contrast_plan(config):
    runs = {(r["residual"], r["crop"], r["flip"]): r["name"] for r in config["runs"]}
    plain = runs[False, False, False]
    crop = runs[False, True, False]
    flip = runs[False, False, True]
    both = runs[False, True, True]
    res_plain = runs[True, False, False]
    res_both = runs[True, True, True]
    return {
        "residual_effect_no_augmentation": {res_plain: 1, plain: -1},
        "residual_effect_with_both": {res_both: 1, both: -1},
        "both_effect_baseline": {both: 1, plain: -1},
        "both_effect_residual": {res_both: 1, res_plain: -1},
        "crop_effect_baseline": {crop: 1, plain: -1},
        "flip_effect_baseline": {flip: 1, plain: -1},
        "flip_added_to_crop": {both: 1, crop: -1},
        "crop_added_to_flip": {both: 1, flip: -1},
        "crop_flip_interaction": {both: 1, crop: -1, flip: -1, plain: 1},
        "residual_augmentation_interaction": {res_both: 1, res_plain: -1, both: -1, plain: 1},
    }


def sign_test(differences):
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(config, runs, output):
    group_rows = []
    for run in config["runs"]:
        block = [row for row in runs if row["run"] == run["name"]]
        if [row["seed"] for row in block] != config["seeds"]:
            raise ValueError(f"Incomplete seed block: {run['name']}")
        scores = [row["test_accuracy"] for row in block]
        group_rows.append({"run": run["name"], "residual": run["residual"], "crop": run["crop"], "flip": run["flip"],
                           "n_seeds": len(scores), "test_mean": mean(scores), "test_sample_sd": stdev(scores),
                           "best_val_accuracy_mean": mean(row["best_val_accuracy"] for row in block),
                           "epochs_min": min(row["epochs_completed"] for row in block),
                           "epochs_max": max(row["epochs_completed"] for row in block),
                           "training_seconds_total": sum(row["training_seconds"] for row in block),
                           "training_seconds_mean": mean(row["training_seconds"] for row in block)})

    lookup = {(row["run"], row["seed"]): row for row in runs}
    paired_rows, contrast_rows = [], []
    for name, weights in contrast_plan(config).items():
        differences, validation = [], []
        for seed in config["seeds"]:
            test_delta = sum(weight * lookup[run, seed]["test_correct"] for run, weight in weights.items())
            val_delta = sum(weight * round(5000 * lookup[run, seed]["best_val_accuracy"]) for run, weight in weights.items())
            differences.append(test_delta)
            validation.append(val_delta)
            paired_rows.append({"contrast": name, "seed": seed, "test_delta_correct": test_delta,
                                "test_delta_pp": test_delta / 100, "validation_delta_pp": val_delta / 50})
        contrast_rows.append({"contrast": name, "n_seeds": len(differences),
                              "test_mean_delta_pp": mean(differences) / 100,
                              "test_sample_sd_delta_pp": stdev(differences) / 100,
                              "test_min_delta_pp": min(differences) / 100, "test_max_delta_pp": max(differences) / 100,
                              "positive_seeds": sum(delta > 0 for delta in differences),
                              "negative_seeds": sum(delta < 0 for delta in differences),
                              "tie_seeds": sum(delta == 0 for delta in differences),
                              "two_sided_exact_sign_p_descriptive": sign_test(differences),
                              "validation_mean_delta_pp": mean(validation) / 50})

    convergence = []
    for row in runs:
        run, seed = row["run"], row["seed"]
        history = read_csv(run_folder(output, {"name": run}, seed) / "history.csv")
        early = [float(item["val_accuracy"]) for item in history[:24]]
        best_row = history[row["best_epoch"] - 1]
        convergence.append({"run": run, "seed": seed, "epochs_completed": len(history),
                            "best_epoch": row["best_epoch"], "stop_reason": row["stop_reason"],
                            "best_val_accuracy_first_24": max(early), "best_val_accuracy_full": row["best_val_accuracy"],
                            "best_val_gain_after_first_24_pp": (row["best_val_accuracy"] - max(early)) * 100,
                            "best_at_cap": row["best_epoch"] == row["shared_epoch_cap"],
                            "val_loss_window_change": row["last_window_val_loss_change"],
                            "clean_train_val_gap_at_best_pp": (float(best_row["train_accuracy"]) - row["best_val_accuracy"]) * 100,
                            "training_seconds": row["training_seconds"]})
    write_csv(output / "runs.csv", runs)
    write_csv(output / "group_summary.csv", group_rows)
    write_csv(output / "paired_by_seed.csv", paired_rows)
    write_csv(output / "paired_summary.csv", contrast_rows)
    write_csv(output / "convergence.csv", convergence)
    result = {"group_summary": group_rows, "paired_summary": contrast_rows,
              "contrast_weights": contrast_plan(config), "convergence": convergence,
              "sampling_unit": "training seed, paired on one fixed official test set",
              "standard_deviation": "sample standard deviation (n-1)",
              "sign_test_scope": "descriptive seed-level exact sign test, ties excluded; minimum two-sided p=0.25 for n=3",
              "total_training_seconds": sum(row["training_seconds"] for row in runs)}
    write_json(output / "summary.json", result)
    return result
