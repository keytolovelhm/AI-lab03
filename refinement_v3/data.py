"""Data pipeline for stage 3.

All tensors come from the frozen split produced in stage 1 (results/split_indices.npz +
results/split.json). Nothing here re-splits the data, so the 45,000/5,000 partition and the
normalisation statistics are exactly those frozen before stage 1 training.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .common import DATA_DIR, STAGE1_RESULTS, read_json

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Dataset:
    train_images: torch.Tensor      # (45000, 32, 32, 3) normalised
    train_labels: torch.Tensor
    val_images: torch.Tensor        # (5000, 32, 32, 3) normalised
    val_labels: torch.Tensor
    test_images: torch.Tensor       # (10000, 32, 32, 3) normalised
    test_labels: torch.Tensor
    normalization: dict

    @property
    def classes(self):
        return list(self.normalization["classes"])


def _load_cifar_batch(path: Path):
    with path.open("rb") as stream:
        payload = pickle.load(stream, encoding="bytes")
    data = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(payload[b"labels"], dtype=np.int64)
    return data, labels


def _normalise(raw: np.ndarray, mean, std) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(raw)).float().div_(255.0)
    mean = torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 3)
    std = torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 3)
    return (tensor - mean) / std


@lru_cache(maxsize=2)
def _load(normalization_key: tuple, include_test: bool = True) -> Dataset:
    metadata = read_json(STAGE1_RESULTS / "split.json")
    split = np.load(STAGE1_RESULTS / "split_indices.npz")
    train_idx, val_idx = split["train"], split["val"]
    base = DATA_DIR / "cifar-10-batches-py"
    parts = [_load_cifar_batch(base / f"data_batch_{i}") for i in range(1, 6)]
    train_raw = np.concatenate([p[0] for p in parts])
    train_labels = np.concatenate([p[1] for p in parts])
    if include_test:
        test_raw, test_labels = _load_cifar_batch(base / "test_batch")
    else:
        test_raw = np.empty((0, 32, 32, 3), dtype=np.uint8)
        test_labels = np.empty(0, dtype=np.int64)
    mean = metadata["train_mean"] if normalization_key == ("cifar",) else IMAGENET_MEAN
    std = metadata["train_std"] if normalization_key == ("cifar",) else IMAGENET_STD
    return Dataset(
        train_images=_normalise(train_raw[train_idx], mean, std),
        train_labels=torch.from_numpy(train_labels[train_idx]),
        val_images=_normalise(train_raw[val_idx], mean, std),
        val_labels=torch.from_numpy(train_labels[val_idx]),
        test_images=_normalise(test_raw, mean, std),
        test_labels=torch.from_numpy(test_labels),
        normalization=metadata,
    )


def load_dataset(normalization: str = "cifar") -> Dataset:
    """normalization='cifar' uses the frozen train-set statistics; 'imagenet' for transfer."""
    return _load((normalization,))


def load_training_dataset(normalization: str = "cifar") -> Dataset:
    return _load((normalization,), False)


def resize_batch(images_chw: torch.Tensor, size: int) -> torch.Tensor:
    """Upsample a (B,3,32,32) batch to (B,3,size,size) with bicubic filtering."""
    return torch.nn.functional.interpolate(images_chw, size=(size, size), mode="bicubic",
                                           align_corners=False, antialias=True)


def aug_index(count: int, size: int, crop_rng, flip_rng):
    """Indices replicating reflect-pad-4 + random crop + random flip, sampled per image.

    Mirrors ``refinement.core.batch_augmentation`` so same-seed streams stay comparable.
    """
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
    return rows, columns


def gather_augmented(images: torch.Tensor, indices: torch.Tensor, rows, columns) -> torch.Tensor:
    return images[indices][torch.arange(len(indices))[:, None, None], rows[:, :, None], columns[:, None, :]]
