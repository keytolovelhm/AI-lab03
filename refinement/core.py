import csv
import hashlib
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch

from src.data import prepare_training
from src.runtime import ROOT, write_json


REFINEMENT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing metadata differs: {path}. Choose another output name.")
    else:
        write_json(path, value)


def suite_path(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Use an output name containing only lowercase letters, digits, _ or -")
    root = REFINEMENT / "runs"
    path = (root / name).resolve()
    if path.parent != root or root.resolve() != root:
        raise ValueError("Suite output must stay inside refinement/runs")
    return path


@contextmanager
def suite_lock(path):
    # An OS lock is released automatically when a training process exits unexpectedly.
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".lock").open("a+b", buffering=0) as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another process is already using {path}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Another process is already using {path}") from exc
        yield


def load_config(path):
    config = read_json(path)
    names = [run["name"] for run in config["runs"]]
    keys = [(run["residual"], run["crop"], run["flip"]) for run in config["runs"]]
    expected = {(False, False, False), (False, True, True), (True, False, False),
                (True, True, True), (False, True, False), (False, False, True)}
    if len(names) != 6 or len(set(names)) != 6 or set(keys) != expected:
        raise ValueError("Expected six distinct architecture and augmentation conditions")
    if any(not re.fullmatch(r"[a-z0-9_]+", name) for name in names):
        raise ValueError("Invalid run name")
    if len(config["seeds"]) != 3 or len(set(config["seeds"])) != 3:
        raise ValueError("Expected three distinct training seeds")
    if not 1 <= config["epochs"] <= config["extension_epochs"] <= 80 or config["patience"] < 1:
        raise ValueError("Invalid epoch budget or patience")
    return config


def verified_training_data(config):
    indices = ROOT / "results" / "split_indices.npz"
    if not indices.is_file():
        raise FileNotFoundError(indices)
    original = read_json(ROOT / "results" / "split.json")
    data = prepare_training(ROOT / "data", config["split_seed"], indices)
    if data.metadata != original or original["split_sha256"] != config["split_sha256"]:
        raise ValueError("Phase-one split or normalization differs")
    return data


def specs(config):
    return [(run, seed) for seed in config["seeds"] for run in config["runs"]]


def run_folder(output, run, seed):
    return output / "experiments" / run["name"] / str(seed)


def seed_streams(seed):
    return {"model": seed, "order": seed + 10_000, "crop": seed + 20_000, "flip": seed + 30_000}


def batch_augmentation(images, crop_rng, flip_rng):
    if crop_rng is None and flip_rng is None:
        return images
    count, size = images.shape[:2]
    coordinate = torch.arange(size)
    rows = coordinate.expand(count, -1)
    columns = rows
    if flip_rng is not None:
        flip = torch.rand((count, 1), generator=flip_rng) < 0.5
        columns = torch.where(flip, size - 1 - coordinate, coordinate)
    if crop_rng is not None:
        offsets = torch.randint(-4, 5, (2, count, 1), generator=crop_rng)
        rows = rows + offsets[0]
        columns = columns + offsets[1]
        rows = (size - 1) - ((size - 1) - rows.abs()).abs()
        columns = (size - 1) - ((size - 1) - columns.abs()).abs()
    return images[torch.arange(count)[:, None, None], rows[:, :, None], columns[:, None, :]]


def batches(data, batch_size, run, generators):
    order = data.train[torch.randperm(len(data.train), generator=generators["order"])]
    crop_rng = generators["crop"] if run["crop"] else None
    flip_rng = generators["flip"] if run["flip"] else None
    for indices in order.split(batch_size):
        images = batch_augmentation(data.images[indices], crop_rng, flip_rng)
        yield images.permute(0, 3, 1, 2), data.labels[indices]
