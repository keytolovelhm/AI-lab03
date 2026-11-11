"""Stage-3 training loop.

The corrected coordinator owns the shared 60->80 decision. Historical outputs are read-only.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .common import (CONFIGS, RUNS, SEEDS, STREAM_OFFSETS, peak_memory_mb, read_json, save_checkpoint,
                     sha256_file, write_csv, write_json)
from .config import BASE
from .data import aug_index, gather_augmented, load_training_dataset, resize_batch
from .mixing import apply_mixing, soft_target_loss


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


_INTEROP_CONFIGURED = False


def configure(config: dict = BASE):
    global _INTEROP_CONFIGURED
    torch.set_num_threads(config["cpu_threads"])
    if not _INTEROP_CONFIGURED:
        torch.set_num_interop_threads(1)
        _INTEROP_CONFIGURED = True
    torch.use_deterministic_algorithms(True)
    return torch.device(config["device"])


def precision(config: dict):
    if config.get("precision", "fp32") == "bf16":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


# --------------------------------------------------------------------------- model factory
def build(run: dict, config: dict = BASE):
    kind = run.get("family", "stage")
    if kind == "stage":
        from .models import build_stage_model

        return build_stage_model(config, run), {"params": None}
    if kind == "mobilenet":
        from .models import MobileNetV3SmallCifar, load_pretrained_backbone, set_trainable

        model = MobileNetV3SmallCifar(num_classes=10)
        info = {"pretrained": False, "mode": run.get("trainable", "frozen")}
        weights = run.get("weights_path")
        if weights:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            load_pretrained_backbone(model, state)
            info.update({"pretrained": True, "weights_path": str(weights),
                         "weights_sha256": sha256_file(Path(weights))})
        info.update(set_trainable(model, run.get("trainable", "frozen")))
        info["params"] = sum(p.numel() for p in model.parameters())
        return model, info
    raise ValueError(f"unknown family {kind!r}")


# --------------------------------------------------------------------------- evaluation
@torch.inference_mode()
def clean_metrics(model, images: torch.Tensor, labels: torch.Tensor, config: dict, device, indices=None,
                  input_size: int | None = None, bs: int | None = None):
    indices = torch.arange(len(labels)) if indices is None else indices
    bs = bs or config["evaluation_batch_size"]
    model.eval()
    loss_sum, correct = 0.0, 0
    for positions in indices.split(bs):
        batch = images[positions].permute(0, 3, 1, 2).to(device)
        if input_size:
            batch = resize_batch(batch, input_size)
        batch = batch.to(memory_format=torch.channels_last)
        target = labels[positions].to(device)
        with precision(config):
            logits = model(batch).float()
        loss_sum += nn.functional.cross_entropy(logits, target, reduction="sum").item()
        correct += int((logits.argmax(1) == target).sum())
    return loss_sum / len(indices), correct / len(indices)


@torch.inference_mode()
def logits_for(model, images: torch.Tensor, config: dict, device, indices=None, input_size=None, bs=None):
    indices = torch.arange(len(images)) if indices is None else indices
    bs = bs or config["evaluation_batch_size"]
    model.eval()
    out = []
    for positions in indices.split(bs):
        batch = images[positions].permute(0, 3, 1, 2).to(device)
        if input_size:
            batch = resize_batch(batch, input_size)
        batch = batch.to(memory_format=torch.channels_last)
        with precision(config):
            out.append(model(batch).float().cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------- one run
def run_dir(run: dict, seed: int) -> Path:
    return RUNS / "suite_corrected" / "experiments" / run["name"] / str(seed)


def _signature(run: dict, seed: int, config: dict) -> dict:
    return {"config": config, "run": run, "seed": seed,
            "streams": {k: seed + v for k, v in STREAM_OFFSETS.items()},
            "split_sha256": config["split_sha256"]}


def epoch_batches(dataset, run, batch_size, generators):
    order = torch.randperm(len(dataset.train_labels), generator=generators["order"])
    for positions in order.split(batch_size):
        rows, columns = aug_index(len(positions), 32,
                                 generators["crop"] if run.get("crop", True) else None,
                                 generators["flip"] if run.get("flip", True) else None)
        images = gather_augmented(dataset.train_images, positions, rows, columns).permute(0, 3, 1, 2)
        yield images, dataset.train_labels[positions], positions


def train_one(run: dict, seed: int, overrides: dict | None = None, log=print,
              *, folder=None, budget=None, finalize=False) -> dict:
    config = {**BASE, **(overrides or {})}
    config["trainer_protocol"] = "corrected-v1"
    folder = folder or RUNS / "suite_corrected" / "experiments" / run["name"] / str(seed)
    resolved = folder.resolve()
    allowed = [RUNS / "suite_corrected", RUNS / "suite_maintenance"]
    if not any(resolved.is_relative_to(root.resolve()) for root in allowed):
        raise ValueError("Training must stay inside corrected or maintenance suite")
    folder.mkdir(parents=True, exist_ok=True)
    identity = _signature(run, seed, config)
    from .common import frozen_json

    frozen_json(folder / "config.json", identity)
    if (folder / "training.json").exists():
        summary = read_json(folder / "training.json")
        log(f"[skip] {run['name']}/{seed} already trained ({summary['epochs_completed']} epochs)")
        return summary

    device = configure(config)
    normalization = "imagenet" if run.get("family") == "mobilenet" else "cifar"
    dataset = load_training_dataset(normalization)
    cap = budget or config["epochs"]
    input_size = run.get("input_size")
    batch_size = run.get("batch_size", config["batch_size"])

    resume = folder / "resume.pt"
    history, best, stale = [], None, 0
    seed_everything(seed)
    model, info = build(run, config)
    model = model.to(device, memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.get("learning_rate", BASE["learning_rate"]),
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
                                                           eta_min=config["min_learning_rate"])
    generators = {name: torch.Generator().manual_seed(seed + offset) for name, offset in STREAM_OFFSETS.items()}
    # 'mix' stream must not diverge for unmixed runs: identical start state is fine.
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["signature"] != identity:
            raise ValueError(f"resume state belongs to another configuration: {folder}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        for name, gen_state in state["generators"].items():
            generators[name].set_state(gen_state)
        history, best, stale = state["history"], state["best"], state["stale"]
        log(f"[resume] {run['name']}/{seed} at epoch {len(history)}")

    mixing_kind = run.get("mixing", "none")
    mix_alpha = run.get("mix_alpha", 0.0)
    smoothing = run.get("label_smoothing", 0.0)
    from .models import freeze_bn_for_mode

    for epoch in range(len(history) + 1, cap + 1):
        if stale >= config["patience"]:
            break
        started = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        if run.get("family") == "mobilenet":
            freeze_bn_for_mode(model, run.get("trainable", "frozen"))
        online = 0.0
        for images, labels, positions in epoch_batches(dataset, run, batch_size, generators):
            if input_size:
                images = resize_batch(images, input_size)
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)
            images, partner, lam, _ = apply_mixing(images, mixing_kind, mix_alpha, generators["mix"])
            optimizer.zero_grad(set_to_none=True)
            with precision(config):
                logits = model(images)
                loss = soft_target_loss(logits, labels, partner, lam, smoothing)
            loss.backward()
            optimizer.step()
            online += loss.item() * len(positions)
        online /= len(dataset.train_labels)
        train_loss, train_acc = clean_metrics(model, dataset.train_images, dataset.train_labels, config, device,
                                              indices=torch.arange(len(dataset.train_labels)),
                                              input_size=input_size)
        val_loss, val_acc = clean_metrics(model, dataset.val_images, dataset.val_labels, config, device,
                                          input_size=input_size)
        if not all(math.isfinite(v) for v in (online, train_loss, train_acc, val_loss, val_acc)):
            raise ValueError(f"non-finite metrics at {folder} epoch {epoch}")
        record = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_accuracy": train_acc,
                  "online_loss": online, "val_loss": val_loss, "val_accuracy": val_acc,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        if best is None or (val_acc, -val_loss) > (best["val_accuracy"], -best["val_loss"]):
            best = {**record, "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if epoch <= config["epochs"]:
            scheduler.step()
        save_checkpoint(resume, {"signature": identity, "model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "history": history, "best": best, "stale": stale,
                                 "torch_rng": torch.get_rng_state(),
                                 "generators": {k: g.get_state() for k, g in generators.items()}})
        write_csv(folder / "history.csv", history)
        log(f"{run['name']}/{seed} | {epoch:02d}/{cap} | train {train_acc:.2%} {train_loss:.4f} | "
            f"val {val_acc:.2%} {val_loss:.4f} | best {best['epoch']} {best['val_accuracy']:.2%} | "
            f"{record['seconds']:.1f}s")

    write_csv(folder / "history.csv", history)
    state = {"history": history, "best": best, "stale": stale, "parameters": params,
             "trainable_parameters": trainable_count, "model_info": info}
    if not finalize:
        return state
    summary = {**identity, "parameters": params, "trainable_parameters": trainable_count,
               "epochs_completed": len(history), "shared_epoch_cap": cap,
               "stop_reason": "early_stopping" if len(history) < cap else "epoch_budget",
               "training_seconds": sum(r["seconds"] for r in history), "best_epoch": best["epoch"],
               "best_val_accuracy": best["val_accuracy"], "best_val_loss": best["val_loss"],
               "peak_memory_mb": peak_memory_mb(), "model_info": info,
               "last_window_val_loss_change": loss_change(history, config["loss_trend_window"])}
    save_checkpoint(folder / "best.pt", {**identity, "model": best["model"], "best_epoch": best["epoch"],
                                         "normalization": dataset.normalization})
    write_json(folder / "training.json", summary)
    resume.unlink(missing_ok=True)
    return summary


def loss_change(history, window):
    if len(history) < 2 * window:
        return None
    previous = float(np.mean([r["val_loss"] for r in history[-2 * window:-window]]))
    latest = float(np.mean([r["val_loss"] for r in history[-window:]]))
    return latest - previous


_TRAIN_INDEX = None


def train_index():
    global _TRAIN_INDEX
    if _TRAIN_INDEX is None:
        from .common import STAGE1_RESULTS

        _TRAIN_INDEX = torch.from_numpy(np.load(STAGE1_RESULTS / "split_indices.npz")["train"]).long()
    return _TRAIN_INDEX


