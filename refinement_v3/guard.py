"""Build the frozen audit snapshot of stage-1 / stage-2 artefacts and verify it unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from .common import (CONFIGS, PROTECTED_DIRS, PROTECTED_FILES, RESULTS, ROOT, STAGE2_SUITE,
                     V3, read_csv, sha256_file, write_json, frozen_json, read_json)

SNAPSHOT = RESULTS / "protected_snapshot.json"
STAGE2_MANIFEST = RESULTS / "stage2_manifest.json"
PREFLIGHT = CONFIGS / "preflight.json"


def protected_manifest() -> dict:
    """Every stage-1/2 delivery file, excluding the bundled virtualenv and caches."""
    files = [ROOT / name for name in PROTECTED_FILES]
    for directory in PROTECTED_DIRS:
        base = ROOT / directory
        if base.is_dir():
            files.extend(path for path in base.rglob("*")
                         if path.is_file() and not path.is_relative_to(ROOT / "refinement" / ".venv"))
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(files)}


def stage2_manifest() -> dict:
    """Frozen description of the 18 stage-2 runs that stage 3 reads but never writes."""
    runs = read_csv(STAGE2_SUITE / "runs.csv")
    entries = {}
    for row in runs:
        folder = STAGE2_SUITE / "experiments" / row["run"] / row["seed"]
        evaluation = folder / "evaluation" / "test.json"
        key = f"{row['run']}/{row['seed']}"
        entries[key] = {
            "best.pt": sha256_file(folder / "best.pt"),
            "history.csv": sha256_file(folder / "history.csv"),
            "training.json": sha256_file(folder / "training.json"),
            "test.json": sha256_file(evaluation),
            "predictions.csv": sha256_file(folder / "evaluation" / "predictions.csv"),
            "test_accuracy": float(row["test_accuracy"]),
        }
    return {
        "n_runs": len(entries),
        "suite_files": {name: sha256_file(STAGE2_SUITE / name) for name in
                        ("runs.csv", "group_summary.csv", "paired_summary.csv", "paired_by_seed.csv",
                         "convergence.csv", "training_index.csv", "budget_decision.json", "summary.json")},
        "runs": entries,
    }


def snapshot() -> dict:
    manifest = protected_manifest()
    stage2 = stage2_manifest()
    frozen_json(SNAPSHOT, {"n_files": len(manifest), "files": manifest})
    frozen_json(STAGE2_MANIFEST, stage2)
    print(f"protected snapshot: {len(manifest)} files")
    print(f"stage-2 manifest: {stage2['n_runs']} runs")
    return {"protected": len(manifest), "stage2": stage2["n_runs"]}


def verify_protected() -> int:
    expected = read_json(SNAPSHOT)
    def relevant(name):
        parts=Path(name).parts
        return not any(x in parts for x in ('__pycache__','.venv','.mpl-cache'))
    actual = {k:v for k,v in protected_manifest().items() if relevant(k)}
    expected['files'] = {k:v for k,v in expected['files'].items() if relevant(k)}
    if expected["files"] != actual:
        changed = [name for name in sorted(set(actual) | set(expected["files"]))
                   if expected["files"].get(name) != actual.get(name)]
        raise RuntimeError(f"protected stage-1/2 files changed: {changed[:10]} ({len(changed)} total)")
    return len(expected["files"])


def verify_stage2() -> int:
    expected = read_json(STAGE2_MANIFEST)
    actual = stage2_manifest()
    if expected != actual:
        raise RuntimeError("stage-2 runs differ from the frozen manifest")
    return expected["n_runs"]


def check_gate():
    """Refuse to run any training/evaluation when protected inputs were touched."""
    n1 = verify_protected()
    n2 = verify_stage2()
    return n1, n2
