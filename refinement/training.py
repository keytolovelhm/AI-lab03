import math
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.data import evaluation_batches
from src.models import build_model
from src.runtime import environment, precision_context, save_checkpoint, seed_everything, write_csv, write_json
from src.train import run_epoch, step

from .core import batches, read_csv, read_json, run_folder, seed_streams, specs


def signature(config, run, seed):
    return {"config": config, "run": run, "seed": seed, "random_seeds": seed_streams(seed),
            "split_sha256": config["split_sha256"]}


@torch.inference_mode()
def clean_metrics(model, data, indices, config, device):
    model.eval()
    loss = 0.0
    correct = 0
    with precision_context(config, device):
        for inputs, labels in evaluation_batches(data.images, data.labels, indices, config["evaluation_batch_size"]):
            labels = labels.to(device)
            logits = model(inputs.to(device, memory_format=torch.channels_last)).float()
            loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return loss / len(indices), correct / len(indices)


def generators_for(seed):
    return {name: torch.Generator().manual_seed(value) for name, value in seed_streams(seed).items()
            if name != "model"}


def preflight(config, data, device, output):
    checks = []
    temporary = output / "preflight_weights.pt"
    for run in config["runs"]:
        seed = config["seeds"][0]
        seed_everything(seed)
        model = build_model(config, run, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        x, y = next(batches(data, config["batch_size"], run, generators_for(seed)))
        expected = (len(y), 3, 32, 32)
        if x.shape != expected or not x.isfinite().all():
            raise ValueError(f"Invalid input in {run['name']}")
        before = model.classifier[-1].weight.detach().clone()
        logits, _, loss = step(model, x, y, nn.CrossEntropyLoss(), config, device, optimizer)
        difference = (model.classifier[-1].weight.detach() - before).abs().max().item()
        if logits.shape != (len(y), 10) or not logits.isfinite().all() or not loss.isfinite() or difference == 0:
            raise RuntimeError(f"Invalid preflight forward/backward in {run['name']}")
        val_indices = data.val[:config["batch_size"]]
        expected_metrics = clean_metrics(model, data, val_indices, config, device)
        save_checkpoint(temporary, model.state_dict())
        restored = build_model(config, run, device)
        restored.load_state_dict(torch.load(temporary, map_location="cpu", weights_only=True))
        if clean_metrics(restored, data, val_indices, config, device) != expected_metrics:
            raise RuntimeError("Validation forward changed after checkpoint reload")
        checks.append({"run": run["name"], "batch": list(x.shape), "loss": loss.item(),
                       "max_weight_change": difference, "validation_loss": expected_metrics[0],
                       "validation_accuracy": expected_metrics[1], "checkpoint_reload": True})
        print(f"Preflight {run['name']} | train loss {loss.item():.4f} | checkpoint round-trip passed", flush=True)
    temporary.unlink()
    write_json(output / "preflight.json", {"environment": environment(device), "data": data.metadata, "runs": checks})


def checkpoint_state(model, optimizer, scheduler, generators, identity, device, history, best, stale):
    return {"signature": identity, "device": str(device), "model": model.state_dict(), "best": best,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "stale": stale, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "generators": {name: rng.get_state() for name, rng in generators.items()}}


def train_until(config, run, seed, data, device, folder, budget):
    folder.mkdir(parents=True, exist_ok=True)
    resume_file = folder / "resume.pt"
    identity = signature(config, run, seed)
    seed_everything(seed)
    model = build_model(config, run, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
                                                           eta_min=config["min_learning_rate"])
    generators = generators_for(seed)
    criterion = nn.CrossEntropyLoss()
    history, best, stale = [], None, 0
    if resume_file.exists():
        state = torch.load(resume_file, weights_only=True, map_location="cpu")
        if state["signature"] != identity or state["device"] != str(device):
            raise ValueError(f"Resume state differs from run configuration: {folder}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        for name, rng in generators.items():
            rng.set_state(state["generators"][name])
        history, best, stale = state["history"], state["best"], state["stale"]
        del state

    for epoch in range(len(history) + 1, budget + 1):
        if stale >= config["patience"]:
            break
        started = perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        fitting = batches(data, config["batch_size"], run, generators)
        online_loss, _ = run_epoch(model, fitting, criterion, config, device, optimizer)
        train_loss, train_acc = clean_metrics(model, data, data.train, config, device)
        val_loss, val_acc = clean_metrics(model, data, data.val, config, device)
        if not all(math.isfinite(metric) for metric in (online_loss, train_loss, val_loss, train_acc, val_acc)):
            raise ValueError(f"Nonfinite training or validation metrics: {folder} epoch {epoch}")
        metrics = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_accuracy": train_acc,
                   "val_loss": val_loss, "val_accuracy": val_acc, "online_loss": online_loss,
                   "seconds": perf_counter() - started}
        history.append(metrics)
        if best is None or (val_acc, -val_loss) > (best["val_accuracy"], -best["val_loss"]):
            best = {**metrics, "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        # Finish the shared cosine schedule at epoch 60. Extension holds the LR at its floor.
        if epoch <= config["epochs"]:
            scheduler.step()
        save_checkpoint(resume_file, checkpoint_state(model, optimizer, scheduler, generators, identity,
                                                      device, history, best, stale))
        write_csv(folder / "history.csv", history)
        print(f"{run['name']}/{seed} | {epoch:02d}/{budget} | clean train {train_acc:.2%} {train_loss:.4f} | "
              f"val {val_acc:.2%} {val_loss:.4f} | best {best['epoch']} {best['val_accuracy']:.2%} | "
              f"{metrics['seconds']:.1f}s", flush=True)

    write_csv(folder / "history.csv", history)
    return {"history": history, "best": best, "stale": stale,
            "parameters": sum(parameter.numel() for parameter in model.parameters())}


def loss_change(history, window):
    if len(history) < 2 * window:
        return None
    previous = np.mean([row["val_loss"] for row in history[-2 * window:-window]])
    latest = np.mean([row["val_loss"] for row in history[-window:]])
    return float(latest - previous)


def base_summary(state, config, run, seed):
    history, best = state["history"], state["best"]
    delta = loss_change(history, config["loss_trend_window"])
    reached_cap = len(history) == config["epochs"]
    loss_falling = delta is not None and delta <= -config["loss_trend_threshold"]
    return {"run": run["name"], "seed": seed, "epochs": len(history), "best_epoch": best["epoch"],
            "best_val_accuracy": best["val_accuracy"], "last_window_val_loss_change": delta,
            "stale": state["stale"], "trigger": reached_cap and (best["epoch"] == len(history) or loss_falling)}


def finish(config, run, seed, state, data, device, folder, budget):
    history, best = state["history"], state["best"]
    identity = signature(config, run, seed)
    summary = {**identity, "environment": environment(device), "parameters": state["parameters"],
               "epochs_completed": len(history), "shared_epoch_cap": budget,
               "stop_reason": "early_stopping" if len(history) < budget else "epoch_budget",
               "training_seconds": sum(row["seconds"] for row in history), "best_epoch": best["epoch"],
               "best_val_accuracy": best["val_accuracy"], "best_val_loss": best["val_loss"],
               "last_window_val_loss_change": loss_change(history, config["loss_trend_window"])}
    save_checkpoint(folder / "best.pt", {**identity, "model": best["model"], "best_epoch": best["epoch"],
                                        "normalization": data.metadata})
    write_json(folder / "training.json", summary)
    (folder / "resume.pt").unlink(missing_ok=True)
    return summary


def train_suite(config, data, device, output):
    all_specs = specs(config)
    decision_path = output / "budget_decision.json"
    if not decision_path.exists():
        base = []
        for index, (run, seed) in enumerate(all_specs, 1):
            folder = run_folder(output, run, seed)
            print(f"[{index}/{len(all_specs)}] Initial budget | {run['name']}/{seed}", flush=True)
            state = train_until(config, run, seed, data, device, folder, config["epochs"])
            base.append(base_summary(state, config, run, seed))
        extend = any(row["trigger"] for row in base)
        cap = config["extension_epochs"] if extend else config["epochs"]
        write_json(decision_path, {"initial_cap": config["epochs"], "shared_cap": cap, "base": base,
                                   "selection_inputs": "validation metrics only"})
    cap = read_json(decision_path)["shared_cap"]
    print(f"Shared cap after validation-only review: {cap}", flush=True)
    summaries = []
    for index, (run, seed) in enumerate(all_specs, 1):
        folder = run_folder(output, run, seed)
        result = folder / "training.json"
        if result.exists():
            summary = read_json(result)
            if any(summary[key] != value for key, value in signature(config, run, seed).items()):
                raise ValueError(f"Existing training result differs from suite: {folder}")
            if not (folder / "best.pt").is_file():
                raise FileNotFoundError(folder / "best.pt")
            (folder / "resume.pt").unlink(missing_ok=True)
        else:
            if not (folder / "resume.pt").is_file():
                raise FileNotFoundError(f"Missing committed training state: {folder}")
            print(f"[{index}/{len(all_specs)}] Final budget | {run['name']}/{seed}", flush=True)
            state = train_until(config, run, seed, data, device, folder, cap)
            summary = finish(config, run, seed, state, data, device, folder, cap)
        summaries.append({"run": run["name"], "seed": seed, "epochs": summary["epochs_completed"],
                          "best_epoch": summary["best_epoch"], "best_val_accuracy": summary["best_val_accuracy"],
                          "stop_reason": summary["stop_reason"], "seconds": summary["training_seconds"]})
    write_csv(output / "training_index.csv", summaries)
