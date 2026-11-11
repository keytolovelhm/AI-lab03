"""Calibration metrics and temperature scaling.

ECE uses 15 equal-width confidence bins (fixed). Temperature is fitted on the validation set
only, by minimising NLL with a bounded one-dimensional search; if the optimum lands on a
search boundary the interval is widened and the search restarted. Only one scalar is fitted,
which does not change the arg-max, so Top-1 predictions are invariant by construction.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

N_BINS = 15


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().softmax(dim=1)


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> dict:
    scaled = logits.float() / temperature
    probabilities = softmax(scaled)
    nll = F.cross_entropy(scaled, labels).item()
    one_hot = F.one_hot(labels, num_classes=scaled.shape[1]).float()
    brier = ((probabilities - one_hot) ** 2).sum(dim=1).mean().item()
    accuracy = (scaled.argmax(dim=1) == labels).float().mean().item()
    confidence = probabilities.max(dim=1).values
    predicted = scaled.argmax(dim=1)
    correctness = (predicted == labels).float()

    edges = torch.linspace(0.0, 1.0, N_BINS + 1)
    ece = 0.0
    bins = []
    for index in range(N_BINS):
        low, high = edges[index], edges[index + 1]
        mask = (confidence > low) & (confidence <= high) if index < N_BINS - 1 else (confidence > low)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": index + 1, "low": float(low), "high": float(high), "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        mean_conf = confidence[mask].mean().item()
        mean_acc = correctness[mask].mean().item()
        ece += count / len(labels) * abs(mean_acc - mean_conf)
        bins.append({"bin": index + 1, "low": float(low), "high": float(high), "count": count,
                     "confidence": mean_conf, "accuracy": mean_acc})
    return {"accuracy": accuracy, "nll": nll, "brier": brier, "ece": ece,
            "mean_confidence": confidence.mean().item(), "temperature": temperature, "bins": bins}


def _nll_at(temperature: float, logits: torch.Tensor, labels: torch.Tensor) -> float:
    return F.cross_entropy(logits.float() / temperature, labels).item()


def _golden_section(objective, low: float, high: float, tolerance: float = 1e-6, iterations: int = 200):
    ratio = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if b - a < tolerance:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = objective(d)
    return (a + b) / 2


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, low: float = 0.5, high: float = 5.0,
                    max_expansions: int = 8) -> dict:
    """Fit T>0 minimising validation NLL; expand the bracket while the optimum sits on an edge."""
    objective = lambda t: _nll_at(t, logits, labels)  # noqa: E731
    history = []
    for _ in range(max_expansions):
        optimum = _golden_section(objective, low, high)
        history.append({"low": low, "high": high, "optimum": optimum})
        span = high - low
        if optimum - low < 0.02 * span:
            low = max(0.05, low / 2.0)
            continue
        if high - optimum < 0.02 * span:
            high = high * 2.0
            continue
        break
    grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    best = min(grid + [optimum], key=objective)
    before = logits.argmax(dim=1)
    after = (logits.float() / best).argmax(dim=1)
    return {"temperature": best, "nll": objective(best), "nll_t1": objective(1.0),
            "brackets": history, "argmax_unchanged": bool(torch.equal(before, after))}


def reliability_rows(metrics: dict, model: str, split: str) -> list:
    rows = []
    for item in metrics["bins"]:
        rows.append({"model": model, "split": split, "bin": item["bin"], "low": item["low"],
                     "high": item["high"], "count": item["count"],
                     "confidence": item["confidence"], "accuracy": item["accuracy"]})
    return rows


def brier_multiclass(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probabilities = softmax(logits)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
    return ((probabilities - one_hot) ** 2).sum(dim=1).mean().item()


def summarise_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    return calibration_metrics(torch.from_numpy(np.asarray(logits)), torch.from_numpy(np.asarray(labels)))
