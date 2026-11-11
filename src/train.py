from time import perf_counter

import torch
from torch import nn

from .data import evaluation_batches, training_batches
from .models import build_model
from .runtime import environment, precision_context, save_checkpoint, seed_everything, write_csv, write_json


def step(model, images, labels, criterion, config, device, optimizer=None):
    images = images.to(device, memory_format=torch.channels_last)
    labels = labels.to(device)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    with precision_context(config, device):
        logits = model(images)
        loss = criterion(logits, labels)
    if optimizer is not None:
        loss.backward()
        optimizer.step()
    return logits.detach(), labels, loss.detach()


def run_epoch(model, batches, criterion, config, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    stats = torch.zeros(2, dtype=torch.float64, device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in batches:
            logits, labels, loss = step(model, images, labels, criterion, config, device, optimizer)
            stats[0] += loss * len(labels)
            stats[1] += (logits.argmax(dim=1) == labels).sum()
            count += len(labels)
    loss, accuracy = (stats / count).tolist()
    return loss, accuracy


def check_runs(config, runs, data, device):
    checks = []
    for run in runs:
        seed_everything(config["train_seed"])
        model = build_model(config, run, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        order = torch.Generator().manual_seed(config["train_seed"])
        augmentation = torch.Generator().manual_seed(config["train_seed"] + 1) if run["augment"] else None
        images, labels = next(training_batches(data, config["batch_size"], order, augmentation))
        if images.shape != (config["batch_size"], 3, 32, 32) or not images.isfinite().all():
            raise ValueError("Invalid image batch")
        before = model.classifier[-1].weight.detach().clone()
        start = perf_counter()
        logits, _, loss = step(model, images, labels, nn.CrossEntropyLoss(), config, device, optimizer)
        seconds = perf_counter() - start
        if logits.shape != (len(labels), 10) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("Invalid logits or loss")
        delta = (model.classifier[-1].weight.detach() - before).abs().max().item()
        if delta == 0:
            raise RuntimeError("Optimizer did not change the classifier weights")
        checks.append({"run": run["name"], "input_shape": list(images.shape), "logits_shape": list(logits.shape),
                       "loss": loss.item(), "max_weight_change": delta, "seconds": seconds})
    return checks


def train_model(config, run, data, device, result_dir, checkpoint_dir, resume):
    signature = {"config": config, "run": run, "split_sha256": data.metadata["split_sha256"]}
    last_file = checkpoint_dir / "last.pt"
    if last_file.exists() and not resume:
        raise FileExistsError(f"Use --resume to continue {run['name']}, or choose a new output directory")

    seed = config["train_seed"]
    seed_everything(seed)
    model = build_model(config, run, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["min_learning_rate"]
    )
    order_rng = torch.Generator().manual_seed(seed)
    augment_rng = torch.Generator().manual_seed(seed + 1)
    history = []
    best = None
    stale = 0
    if last_file.exists():
        state = torch.load(last_file, map_location="cpu", weights_only=True)
        if state["signature"] != signature or state["device"] != str(device):
            raise ValueError(f"Cannot resume with a different configuration, split or device: {last_file}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        order_rng.set_state(state["order_rng"])
        augment_rng.set_state(state["augment_rng"])
        history, best, stale = state["history"], state["best"], state["stale"]

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch_size = config["batch_size"]
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        if stale >= config["patience"]:
            break
        started = perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_batches = training_batches(data, batch_size, order_rng, augment_rng if run["augment"] else None)
        train_loss, train_acc = run_epoch(model, train_batches, criterion, config, device, optimizer)
        val_batches = evaluation_batches(data.images, data.labels, data.val, batch_size)
        val_loss, val_acc = run_epoch(model, val_batches, criterion, config, device)
        metrics = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_accuracy": train_acc,
                   "val_loss": val_loss, "val_accuracy": val_acc, "seconds": perf_counter() - started}
        history.append(metrics)
        if best is None or (val_acc, -val_loss) > (best["val_accuracy"], -best["val_loss"]):
            best = {**metrics, "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        save_checkpoint(last_file, {
            "signature": signature, "device": str(device), "model": model.state_dict(), "best": best,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history, "stale": stale,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "order_rng": order_rng.get_state(), "augment_rng": augment_rng.get_state(),
        })
        write_csv(result_dir / "history.csv", history)
        print(f"{run['name']} | epoch {epoch:02d}/{config['epochs']} | "
              f"train {train_loss:.4f}, {train_acc:.2%} | val {val_loss:.4f}, {val_acc:.2%} | "
              f"{metrics['seconds']:.1f}s", flush=True)

    summary = {
        **signature,
        "environment": environment(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs_completed": len(history),
        "stop_reason": "early_stopping" if len(history) < config["epochs"] else "epoch_budget",
        "training_seconds": sum(row["seconds"] for row in history),
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val_accuracy"],
        "best_val_loss": best["val_loss"],
    }
    save_checkpoint(checkpoint_dir / "best.pt", {**signature, "model": best["model"], "normalization": data.metadata,
                                                 "best_epoch": best["epoch"]})
    write_csv(result_dir / "history.csv", history)
    write_json(result_dir / "training.json", summary)
    return summary
