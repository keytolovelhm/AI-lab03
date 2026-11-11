import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from src.runtime import ROOT, configure

from .core import REFINEMENT, file_digest, frozen_json, load_config, read_json, suite_lock, suite_path, verified_training_data
from .training import preflight, train_suite


def original_manifest():
    files = [ROOT / "requirements.txt"]
    for directory in ("data", "configs", "results", "checkpoints"):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    return {path.relative_to(ROOT).as_posix(): file_digest(path) for path in sorted(files)}


def check_original():
    expected = read_json(REFINEMENT / "original_artifacts_sha256.json")
    actual = original_manifest()
    if actual != expected:
        changed = [name for name in sorted(actual.keys() | expected.keys()) if actual.get(name) != expected.get(name)]
        raise RuntimeError(f"Historical artifacts differ: {changed}")
    return len(expected)


def freeze_inputs(config, output):
    training_sources = ("src/data.py", "src/models.py", "src/runtime.py", "src/train.py",
                        "refinement/core.py", "refinement/training.py")
    frozen_json(output / "config.json", config)
    frozen_json(output / "training_sources_sha256.json", {name: file_digest(ROOT / name) for name in training_sources})


def main():
    parser = argparse.ArgumentParser(description="Run the six-condition, three-seed CIFAR-10 refinement suite")
    parser.add_argument("command", choices=("snapshot", "check", "train", "evaluate", "verify"))
    parser.add_argument("--config", type=Path, default=REFINEMENT / "experiment.json")
    parser.add_argument("--output", default="suite", help="Directory name inside refinement/runs (default: suite)")
    args = parser.parse_args()
    if args.command == "snapshot":
        manifest = original_manifest()
        frozen_json(REFINEMENT / "original_artifacts_sha256.json", manifest)
        print(f"Protected original files: {len(manifest)}", flush=True)
        return

    config = load_config(args.config)
    output = suite_path(args.output)
    device = configure(config)
    with suite_lock(output):
        print(f"Original artifact SHA-256 checks passed ({check_original()} files)", flush=True)
        freeze_inputs(config, output)
        if args.command == "train" and (output / "training_index.csv").is_file():
            print(f"Completed training preserved in {output}. Choose a new --output for a fresh run.", flush=True)
            return
        if args.command in ("check", "train"):
            data = verified_training_data(config)
            frozen_json(output / "split.json", data.metadata)
            print(f"Training split {data.metadata['split_sha256']} | {len(data.train)}/{len(data.val)}", flush=True)
            if not (output / "preflight.json").exists():
                preflight(config, data, device, output)
            else:
                print("Existing preflight verified against the frozen input configuration", flush=True)
            if args.command == "train":
                train_suite(config, data, device, output)
        elif args.command == "evaluate":
            from .evaluation import evaluate_suite

            evaluate_suite(config, device, output)
        else:
            from .verification import verify_suite

            verify_suite(config, device, output)


if __name__ == "__main__":
    main()
