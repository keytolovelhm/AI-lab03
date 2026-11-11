import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .data import prepare_training
from .runtime import ROOT, configure, environment, write_json
from .train import check_runs, train_model


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate controlled CIFAR-10 CNN experiments")
    parser.add_argument("command", choices=("check", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue training from each saved epoch")
    args = parser.parse_args()
    if args.resume and args.command != "train":
        parser.error("--resume is only supported by train")
    using_default = args.output is None
    args.output = (args.output or ROOT / "refinement" / "legacy_reproduction").resolve()
    for protected in (ROOT / "results", ROOT / "checkpoints", ROOT / "data", ROOT / "configs", ROOT / "src"):
        if args.output.is_relative_to(protected) or protected.is_relative_to(args.output):
            parser.error("The original project outputs are protected. Use an independent --output directory.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runs = config.pop("runs")
    names = [run["name"] for run in runs]
    if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[a-z0-9_]+", name) for name in names):
        parser.error("Run names must be unique and use lowercase letters, digits or underscores")
    if config.get("precision", "fp32") not in ("fp32", "bf16"):
        parser.error("precision must be fp32 or bf16")
    if min(config[key] for key in ("epochs", "patience", "batch_size", "cpu_threads")) < 1:
        parser.error("epochs, patience, batch_size and cpu_threads must be positive")
    device = configure(config)
    results = args.output / "results"
    results.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        if using_default and (results / "comparison.json").exists():
            print(f"Completed evaluation already exists in {results}. Use --output for a new reproduction.")
            return
        from .evaluate import evaluate_suite

        evaluate_suite(config, runs, device, args.data, args.output)
        return

    print(f"Loading official training images on {device}; building stratified split", flush=True)
    data = prepare_training(args.data, config["split_seed"], results / "split_indices.npz")
    print(f"Train={len(data.train)} validation={len(data.val)} | SHA-256 {data.metadata['split_sha256']}", flush=True)
    if args.command == "check":
        if using_default and (results / "preflight.json").exists():
            print(f"Existing preflight preserved in {results}.")
            return
        checks = check_runs(config, runs, data, device)
        result = {"config": config, "runs": runs, "environment": environment(device),
                  "data": data.metadata, "one_batch_checks": checks}
        write_json(results / "preflight.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return

    write_json(results / "split.json", data.metadata)
    print(f"Training {', '.join(names)}", flush=True)
    for run in runs:
        name = run["name"]
        finished = results / name / "training.json"
        if using_default and finished.exists():
            summary = json.loads(finished.read_text(encoding="utf-8"))
            if summary["run"] != run or summary["config"] != config:
                parser.error("The existing default reproduction uses another configuration. Specify --output.")
            print(f"Completed run preserved: {name}")
            continue
        train_model(config, run, data, device, results / name, args.output / "checkpoints" / name,
                    args.resume or using_default)


if __name__ == "__main__":
    main()
