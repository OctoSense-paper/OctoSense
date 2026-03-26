"""Classification benchmark metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _as_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            return None
    return None


def _as_label_tensor(value: Any) -> torch.Tensor | None:
    tensor = _as_tensor(value)
    if tensor is None:
        return None
    if tensor.ndim > 1:
        tensor = tensor.argmax(dim=-1)
    return tensor.reshape(-1).to(dtype=torch.long)


def _macro_f1_score(pred: Any, target: Any) -> float:
    pred_tensor = _as_label_tensor(pred)
    target_tensor = _as_label_tensor(target)
    if pred_tensor is None or target_tensor is None or pred_tensor.numel() == 0:
        return 0.0
    target_tensor = target_tensor.reshape_as(pred_tensor)
    labels = torch.unique(torch.cat((pred_tensor, target_tensor)))
    if labels.numel() == 0:
        return 0.0

    f1_values: list[torch.Tensor] = []
    for label in labels:
        pred_mask = pred_tensor == label
        target_mask = target_tensor == label
        true_positive = (pred_mask & target_mask).sum().float()
        predicted_positive = pred_mask.sum().float()
        actual_positive = target_mask.sum().float()
        if predicted_positive.item() == 0.0 and actual_positive.item() == 0.0:
            continue
        precision = true_positive / predicted_positive.clamp_min(1.0)
        recall = true_positive / actual_positive.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        f1_values.append(f1)
    if not f1_values:
        return 0.0
    return float(torch.stack(f1_values).mean().item())


def classification_metrics(pred: Any, target: Any) -> dict[str, float]:
    pred_tensor = _as_label_tensor(pred)
    target_tensor = _as_label_tensor(target)
    if pred_tensor is None or target_tensor is None:
        return {"accuracy": 0.0, "macro_f1": 0.0, "correct": 0.0, "count": 0.0}
    target_tensor = target_tensor.reshape_as(pred_tensor)
    correct = float((pred_tensor == target_tensor).sum().item())
    count = float(pred_tensor.numel())
    accuracy = float(correct / count) if count else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": _macro_f1_score(pred_tensor, target_tensor),
        "correct": correct,
        "count": count,
    }


__all__ = ["classification_metrics"]
