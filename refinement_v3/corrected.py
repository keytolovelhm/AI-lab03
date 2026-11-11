"""Corrected stage-three suite. Selection is complete before any test tensors are loaded."""
from pathlib import Path

import numpy as np
import torch

from refinement.core import suite_lock
from refinement.training import base_summary
from .common import ROOT, RUNS, SEEDS, STREAM_OFFSETS, frozen_json, read_json, sha256_file, write_json
from .config import BASE, P0, P1
from .train import train_one, configure, logits_for


def output_root():
    import os
    name = os.environ.get("LAB3_SUITE", "suite_corrected")
    if name not in {"suite_corrected", "suite_maintenance"}:
        raise ValueError("Unsupported suite name")
    return RUNS / name


def sources():
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in
            sorted((ROOT / "refinement_v3").glob("*.py"))}


def validate_split():
    with np.load(ROOT / "results/split_indices.npz", allow_pickle=False) as split:
        train, val = split["train"], split["val"]
    import hashlib
    combined = np.concatenate((train, val)).astype("<i8")
    if not np.array_equal(np.sort(combined), np.arange(50000)):
        raise ValueError("Training/validation indices must be a disjoint partition")
    digest = hashlib.sha256(combined.tobytes()).hexdigest()
    if len(train) != 45000 or len(val) != 5000 or digest != BASE["split_sha256"]:
        raise ValueError("Frozen stage-two split differs")
    return digest


def matrix():
    return [(run, seed) for seed in SEEDS for run in P0 + P1]


def folder(run, seed):
    return output_root() / "experiments" / run["name"] / str(seed)


def shared_cap(base):
    return BASE["extension_epochs"] if any(row["trigger"] for row in base) else BASE["epochs"]


def train(run_name=None, seed=None, group="all"):
    root = output_root()
    with suite_lock(root):
        frozen_json(root / "source_sha256.json", sources())
        frozen_json(root / "protocol.json", {"base": BASE, "runs": P0 + P1, "seeds": list(SEEDS),
                                             "streams": STREAM_OFFSETS, "split_sha256": validate_split()})
        decision = root / "budget_decision.json"
        selected = [(r, s) for r, s in matrix() if (run_name is None or r["name"] == run_name)
                    and (seed is None or s == seed) and (group == "all" or r in (P0 if group == "p0" else P1))]
        if not selected:
            raise ValueError("No experiment matches the selection")
        if not decision.exists():
            for run, value in selected:
                state = train_one(run, value, folder=folder(run, value), budget=BASE["epochs"])
                frozen_json(folder(run, value) / "base_selection.json", base_summary(state, BASE, run, value))
            initial = [folder(r, s) / "base_selection.json" for r, s in matrix()]
            if not all(path.exists() for path in initial):
                print("Initial selection saved. Shared extension waits for all 21 initial runs.", flush=True)
                return
            base = [read_json(path) for path in initial]
            write_json(decision, {"initial_cap": BASE["epochs"], "shared_cap": shared_cap(base),
                                  "base": base, "selection_inputs": "validation metrics only"})
        cap = read_json(decision)["shared_cap"]
        # A filtered command continues only its selected runs after the shared decision exists.
        for run, value in selected:
            train_one(run, value, folder=folder(run, value), budget=cap, finalize=True)
        if all((folder(r, s) / "training.json").exists() for r, s in matrix()):
            seal()


def seal(check_sources=True):
    root = output_root()
    if check_sources:
        frozen_json(root / "source_sha256.json", sources())
    cap = read_json(root / "budget_decision.json")["shared_cap"]
    entries = []
    from .common import read_csv
    for run, seed in matrix():
        path = folder(run, seed)
        summary = read_json(path / "training.json")
        history = read_csv(path / "history.csv")
        best = max(history, key=lambda r: (float(r["val_accuracy"]), -float(r["val_loss"])))
        ck = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        if (len(history) != summary["epochs_completed"] or int(best["epoch"]) != summary["best_epoch"]
                or summary["shared_epoch_cap"] != cap or ck["best_epoch"] != summary["best_epoch"]
                or ck["config"] != summary["config"] or ck["run"] != run or ck["seed"] != seed
                or float(best["val_accuracy"]) != summary["best_val_accuracy"]
                or float(best["val_loss"]) != summary["best_val_loss"]):
            raise ValueError(f"Inconsistent completed experiment: {path}")
        entries.append({"run": run["name"], "seed": seed, "best_epoch": summary["best_epoch"],
                        "sha256": sha256_file(path / "best.pt")})
    manifest = {"selection": "all 21 validation-selected checkpoints sealed before test access", "entries": entries}
    frozen_json(root / "test_manifest.json", manifest)
    return manifest


def evaluate():
    with suite_lock(output_root()):
        manifest = seal(check_sources=False)  # Verify sealed weights/history independently of maintenance source edits.
        from .data import load_dataset
        from .models import build_stage_model
        from .calibration import calibration_metrics
        from .common import write_csv
        configure(BASE)
        data = load_dataset()
        for entry in manifest["entries"]:
            run = next(r for r in P0 + P1 if r["name"] == entry["run"])
            path = folder(run, entry["seed"])
            if (path / "evaluation/test.json").exists():
                continue
            model = build_stage_model(BASE, run).to(memory_format=torch.channels_last)
            ck = torch.load(path / "best.pt", weights_only=True, map_location="cpu")
            model.load_state_dict(ck["model"], strict=True)
            logits = logits_for(model, data.test_images, BASE, torch.device("cpu"))
            pred = logits.argmax(1)
            pending = path / ".evaluation_pending"
            pending.mkdir(exist_ok=True)
            write_csv(pending / "predictions.csv", [{"test_index": i, "true_class": data.classes[int(y)],
                      "predicted_class": data.classes[int(p)], "confidence": float(c)}
                      for i, (y, p, c) in enumerate(zip(data.test_labels, pred, logits.softmax(1).amax(1)))])
            metrics = calibration_metrics(logits, data.test_labels)
            write_json(pending / "test.json", {"test_correct": int((pred == data.test_labels).sum()),
                       "test_accuracy": int((pred == data.test_labels).sum()) / len(pred),
                       "metrics": metrics, "checkpoint_sha256": entry["sha256"],
                       "protocol": {"precision": BASE["precision"], "batch_size": BASE["evaluation_batch_size"]}})
            pending.replace(path / "evaluation")
