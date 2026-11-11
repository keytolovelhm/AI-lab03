import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Normalize, ToTensor


class TorontoCIFAR10(CIFAR10):
    url = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


@dataclass
class TrainingData:
    images: torch.Tensor
    labels: torch.Tensor
    train: torch.Tensor
    val: torch.Tensor
    metadata: dict


def prepare_training(root: Path, seed: int, split_file: Path) -> TrainingData:
    dataset = TorontoCIFAR10(root, train=True, download=True)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for label in range(len(dataset.classes)):
        members = np.flatnonzero(labels == label)
        if len(members) != 5000:
            raise ValueError(f"Expected 5000 training images for class {label}")
        rng.shuffle(members)
        val_parts.append(members[:500])
        train_parts.append(members[500:])
    train = np.sort(np.concatenate(train_parts))
    val = np.sort(np.concatenate(val_parts))
    combined = np.concatenate([train, val])
    if not np.array_equal(np.sort(combined), np.arange(50_000)):
        raise ValueError("Train and validation must be a disjoint partition")

    split_digest = hashlib.sha256(combined.astype("<i8").tobytes()).hexdigest()
    if split_file.exists():
        with np.load(split_file, allow_pickle=False) as saved:
            if not (np.array_equal(saved["train"], train) and np.array_equal(saved["val"], val)):
                raise ValueError(f"Existing split differs from the configured split: {split_file}")
    else:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, train=train, val=val)
        temporary = split_file.with_suffix(".tmp")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(split_file)

    # Fit channel statistics on the 45,000 training images in bounded chunks.
    sums, squares = np.zeros(3), np.zeros(3)
    for indices in np.array_split(train, 45):
        pixels = dataset.data[indices].astype(np.float64) / 255
        sums += pixels.sum(axis=(0, 1, 2))
        squares += np.square(pixels).sum(axis=(0, 1, 2))
    count = len(train) * 32 * 32
    mean = sums / count
    std = np.sqrt(squares / count - mean**2)
    metadata = {
        "dataset": "CIFAR-10",
        "classes": dataset.classes,
        "split_seed": seed,
        "split_sha256": split_digest,
        "train_size": len(train),
        "validation_size": len(val),
        "train_per_class": np.bincount(labels[train]).tolist(),
        "validation_per_class": np.bincount(labels[val]).tolist(),
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
    }
    images = normalize_images(dataset.data, metadata)
    return TrainingData(images, torch.from_numpy(labels), torch.from_numpy(train), torch.from_numpy(val), metadata)


def normalize_images(raw: np.ndarray, metadata: dict) -> torch.Tensor:
    to_tensor = ToTensor()
    # ToTensor runs once per image. Store NHWC for fast CPU gathers and convolutions.
    images = torch.empty(raw.shape, dtype=torch.float32)
    for index, image in enumerate(raw):
        images[index].copy_(to_tensor(image).permute(1, 2, 0))
    normalize = Normalize(metadata["train_mean"], metadata["train_std"], inplace=True)
    normalize(images.permute(0, 3, 1, 2))
    return images


def training_batches(data, batch_size, order_rng, augment_rng=None):
    order = data.train[torch.randperm(len(data.train), generator=order_rng)]
    for indices in order.split(batch_size):
        images = data.images[indices]
        if augment_rng is not None:
            images = crop_and_flip(images, augment_rng)
        yield images.permute(0, 3, 1, 2), data.labels[indices]


def crop_and_flip(images: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    # Independently sample a reflected 4px crop and horizontal flip per image.
    batch, size = images.shape[:2]
    coordinates = torch.arange(size)
    offset = torch.randint(-4, 5, (2, batch, 1), generator=rng)
    flip = torch.rand((batch, 1), generator=rng) < 0.5
    rows = coordinates + offset[0]
    columns = torch.where(flip, size - 1 - coordinates, coordinates) + offset[1]
    rows = (size - 1) - ((size - 1) - rows.abs()).abs()
    columns = (size - 1) - ((size - 1) - columns.abs()).abs()
    return images[torch.arange(batch)[:, None, None], rows[:, :, None], columns[:, None, :]]


def evaluation_batches(images, labels, indices, batch_size):
    for batch in indices.split(batch_size):
        yield images[batch].permute(0, 3, 1, 2), labels[batch]
