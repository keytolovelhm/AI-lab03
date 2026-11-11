"""Stage-3 corrected-suite calibration analysis (CPU FP32, eval(), fixed preprocessing).

Standalone on purpose: imports only *stable* helpers (data/models/calibration) and never the
training loop, so a post-training edit to train.py/zcost.py cannot leak into these numbers.

Protocol
--------
* Model: runs/suite_corrected/experiments/<run>/<seed>/best.pt, eval() mode, CPU, float32
  (no autocast; the stored evaluations used bf16, this analysis uses FP32 end-to-end).
* Inputs: the frozen stage-1 split; per-channel mean/std from results/split.json (identical to
  the stage-2 protocol), applied once by refinement_v3.data.
* Temperature: fitted on VALIDATION logits only (5000 images), one scalar, golden-section +
  bracket expansion; the test set is used only to report the final metrics.
* Metrics: accuracy, 15-bin ECE (refinement_v3.calibration, N_BINS=15), NLL, Brier.
  This is the same code path and binning as the stage-2 zero-cost analysis, so numbers are
  comparable in definition.

Writes a new independent directory (default: refinement_v3/results/final_verified/calibration_standalone).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from refinement_v3.calibration import calibration_metrics, fit_temperature, reliability_rows
from refinement_v3.common import SEEDS, read_csv, read_json, write_csv, write_json
from refinement_v3.config import BASE, P0, P1
from refinement_v3.data import load_dataset
from refinement_v3.models import build_stage_model
from refinement_v3.state_loader import load_by_shape

OUT = ROOT / "refinement_v3/results/final_verified/calibration_standalone"
FOLDER = ROOT / "refinement_v3/runs/suite_corrected/experiments"
STAGE2_DIR = ROOT / "refinement/runs/suite/experiments"
CONFIGS = {run["name"]: run for run in P0 + P1}
# Same architecture the reference/dark baseline uses; stage-2 checkpoints are plain PixelUnshuffle.
STAGE2_RUN = {"name": "stage2_baseline_both", "residual": False, "attention": "none",
              "stem": "pixel_unshuffle", "crop": True, "flip": True, "mixing": "none",
              "label_smoothing": 0.0}


def load_model(run_name: str, seed: int):
    from analysis_code.canonical import load_checkpoint
    return load_checkpoint(FOLDER / run_name / str(seed) / "best.pt")

def load_stage2(seed: int):
    from analysis_code.canonical import load_checkpoint
    return load_checkpoint(STAGE2_DIR / "baseline_both" / str(seed) / "best.pt")

def logits_of(model,images):
    from analysis_code.canonical import infer
    return infer(model,images)

def analyse_run(run_name: str, seed: int, data, loader=None) -> dict:
    model, _ = loader(run_name, seed) if loader else load_model(run_name, seed)
    val_logits = logits_of(model, data.val_images)
    test_logits = logits_of(model, data.test_images)

    fit = fit_temperature(val_logits, data.val_labels)
    temperature = fit["temperature"]
    raw = calibration_metrics(test_logits, data.test_labels)
    cal = calibration_metrics(test_logits, data.test_labels, temperature)

    accuracy = (test_logits.argmax(1) == data.test_labels).float().mean().item()
    return {
        "run": run_name, "seed": seed,
        "temperature": temperature,
        "accuracy": accuracy,
        "argmax_preserved": bool(torch.equal(test_logits.argmax(1), (test_logits / fit["temperature"]).argmax(1))),
        "accuracy_after_temperature": cal["accuracy"],
        "ece_raw": raw["ece"], "ece_cal": cal["ece"],
        "nll_raw": raw["nll"], "nll_cal": cal["nll"],
        "brier_raw": raw["brier"], "brier_cal": cal["brier"],
        "mean_conf_raw": raw["mean_confidence"], "mean_conf_cal": cal["mean_confidence"],
        "raw": raw, "cal": cal,
    }


def main() -> None:
    torch.set_num_threads(BASE["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    from analysis_code.canonical import output_directory
    output_directory(OUT)
    data = load_dataset()

    per_run = []
    for run_name in CONFIGS:
        for seed in SEEDS:
            result = analyse_run(run_name, seed, data)
            print(f"  {run_name:<15}{seed}  T={result['temperature']:.3f}  "
                  f"ECE {result['ece_raw']:.4f}->{result['ece_cal']:.4f}  "
                  f"NLL {result['nll_raw']:.4f}->{result['nll_cal']:.4f}")
            per_run.append(result)

    # Same-code-path reference: the frozen stage-2 baseline_both checkpoints.
    stage2_rows = []
    for seed in SEEDS:
        model, _ = load_stage2(seed)
        val_logits = logits_of(model, data.val_images)
        test_logits = logits_of(model, data.test_images)
        fit = fit_temperature(val_logits, data.val_labels)
        raw = calibration_metrics(test_logits, data.test_labels)
        cal = calibration_metrics(test_logits, data.test_labels, fit["temperature"])
        row = {"run": "stage2_baseline_both", "seed": seed, "temperature": fit["temperature"],
               "accuracy": float((test_logits.argmax(1) == data.test_labels).float().mean()),
               "argmax_preserved": bool(torch.equal(test_logits.argmax(1), (test_logits / fit["temperature"]).argmax(1))),
               "accuracy_after_temperature": cal["accuracy"],
               "ece_raw": raw["ece"], "ece_cal": cal["ece"],
               "nll_raw": raw["nll"], "nll_cal": cal["nll"],
               "brier_raw": raw["brier"], "brier_cal": cal["brier"],
               "mean_conf_raw": raw["mean_confidence"], "mean_conf_cal": cal["mean_confidence"]}
        print(f"  stage2_baseline_both {seed}  T={row['temperature']:.3f}  "
              f"ECE {row['ece_raw']:.4f}->{row['ece_cal']:.4f}  NLL {row['nll_raw']:.4f}->{row['nll_cal']:.4f}")
        stage2_rows.append(row)

    fields = ["run", "seed", "accuracy", "temperature", "accuracy_after_temperature", "argmax_preserved",
              "ece_raw", "ece_cal", "nll_raw", "nll_cal", "brier_raw", "brier_cal",
              "mean_conf_raw", "mean_conf_cal"]
    write_csv(OUT / "calibration_per_seed.csv", [{k: r[k] for k in fields} for r in per_run])
    write_csv(OUT / "calibration_stage2_reference.csv", [{k: r[k] for k in fields} for r in stage2_rows])

    # ---- grouped summary (mean +- SD over the three seeds) --------------------------
    groups = []
    for run_name in CONFIGS:
        block = [r for r in per_run if r["run"] == run_name]
        row = {"run": run_name, "n_seeds": len(block),
               "label_smoothing": CONFIGS[run_name].get("label_smoothing", 0.0),
               "mixing": CONFIGS[run_name].get("mixing", "none")}
        for key in ("accuracy", "accuracy_after_temperature", "temperature", "ece_raw", "ece_cal",
                    "nll_raw", "nll_cal", "brier_raw", "brier_cal", "mean_conf_raw", "mean_conf_cal"):
            values = [r[key] for r in block]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_sd"] = stdev(values) if len(values) > 1 else 0.0
        row["ece_gain_pp"] = (mean([r["ece_raw"] for r in block]) - mean([r["ece_cal"] for r in block])) * 100
        row["nll_gain"] = mean([r["nll_raw"] for r in block]) - mean([r["nll_cal"] for r in block])
        row["brier_gain"] = mean([r["brier_raw"] for r in block]) - mean([r["brier_cal"] for r in block])
        row["all_argmax_preserved"] = all(r["argmax_preserved"] for r in block)
        row["ece_improved_but_nll_worse"] = (
            mean([r["ece_cal"] for r in block]) < mean([r["ece_raw"] for r in block])
            and mean([r["nll_cal"] for r in block]) > mean([r["nll_raw"] for r in block]))
        row["ece_improved_but_brier_worse"] = (
            mean([r["ece_cal"] for r in block]) < mean([r["ece_raw"] for r in block])
            and mean([r["brier_cal"] for r in block]) > mean([r["brier_raw"] for r in block]))
        groups.append(row)
    write_csv(OUT / "calibration_grouped.csv", groups)

    # ---- reliability diagrams for the representative models ------------------------
    from refinement_v3.plots import reliability

    reps = ["se_both", "ls_both", "mixup_both", "cutmix_both", "convstem_both"]
    for run_name in reps:
        block = [r for r in per_run if r["run"] == run_name]
        for label, metrics in (("raw", block[0]["raw"]), ("calibrated", block[0]["cal"])):
            points = {
                "confidence": [b["confidence"] for b in metrics["bins"] if b["count"]],
                "accuracy": [b["accuracy"] for b in metrics["bins"] if b["count"]],
            }
            reliability(points, f"{run_name} seed {SEEDS[0]} test reliability ({label})",
                        OUT / f"reliability_{run_name}_{SEEDS[0]}_{label}.png")
            write_csv(OUT / f"reliability_{run_name}_{SEEDS[0]}_{label}.csv",
                      reliability_rows(metrics, run_name, f"test_{label}"))

    stage2_block = stage2_rows
    stage2_summary = {
        "run": "stage2_baseline_both", "n_seeds": len(stage2_block),
        "ece_raw_mean": mean(r["ece_raw"] for r in stage2_block),
        "ece_raw_sd": stdev(r["ece_raw"] for r in stage2_block),
        "ece_cal_mean": mean(r["ece_cal"] for r in stage2_block),
        "nll_raw_mean": mean(r["nll_raw"] for r in stage2_block),
        "nll_cal_mean": mean(r["nll_cal"] for r in stage2_block),
        "brier_raw_mean": mean(r["brier_raw"] for r in stage2_block),
        "brier_cal_mean": mean(r["brier_cal"] for r in stage2_block),
        "accuracy_mean": mean(r["accuracy"] for r in stage2_block),
        "temperature_mean": mean(r["temperature"] for r in stage2_block),
    }
    write_json(OUT / "calibration_stage2_summary.json", stage2_summary)

    # ---- explicit answers to the five commissioned questions ------------------------
    by_run = {g["run"]: g for g in groups}
    ls = by_run["ls_both"]
    reference_stage2 = mean(r["ece_raw"] for r in stage2_rows)
    summary = {
        "protocol": {
            "device": "cpu", "precision": "float32 (no autocast)", "mode": "eval()",
            "ece_bins": 15, "temperature_fit_split": "validation only",
            "test_used_for": "final reporting only",
            "note": "stored run evaluations used bf16; these numbers are FP32 and may differ in "
                    "the last decimals",
        },
        "q1_label_smoothing_raw_ece": {
            "ls_both_ece_raw": ls["ece_raw_mean"],
            "ls_both_ece_raw_sd": ls["ece_raw_sd"],
            "reference_stage2_baseline_both_ece_raw": reference_stage2,
            "reference_accuracy": stage2_summary["accuracy_mean"],
            "ls_accuracy": ls["accuracy_mean"],
            "improves_raw_ece": (None if reference_stage2 is None
                                 else ls["ece_raw_mean"] < reference_stage2),
            "note": "ECE and accuracy move independently; this is a calibration claim only",
        },
        "q2_mixup_calibration": {
            "mixup_ece_raw": by_run["mixup_both"]["ece_raw_mean"],
            "mixup_nll_raw": by_run["mixup_both"]["nll_raw_mean"],
            "mixup_brier_raw": by_run["mixup_both"]["brier_raw_mean"],
        },
        "q3_temperature_benefit_by_method": {
            g["run"]: {"ece_gain_pp": g["ece_gain_pp"], "nll_gain": g["nll_gain"],
                       "brier_gain": g["brier_gain"], "temperature": g["temperature_mean"]}
            for g in groups},
        "q4_ece_improved_but_nll_or_brier_worse": [
            g["run"] for g in groups if g["ece_improved_but_nll_worse"] or g["ece_improved_but_brier_worse"]],
        "q5_argmax_preserved_all": all(g["all_argmax_preserved"] for g in groups),
        "accuracy_changed_by_temperature": any(
            abs(g["accuracy_after_temperature_mean"] - g["accuracy_mean"]) > 1e-12 for g in groups),
    }
    write_json(OUT / "calibration_summary.json", summary)
    from analysis_code.canonical import sha
    paths=[FOLDER/name/str(seed)/'best.pt' for name in CONFIGS for seed in SEEDS]
    paths += [STAGE2_DIR/'baseline_both'/str(seed)/'best.pt' for seed in SEEDS]
    write_json(OUT/'checkpoint_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    print("\nwrote", OUT)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out:OUT=args.out
    main()
