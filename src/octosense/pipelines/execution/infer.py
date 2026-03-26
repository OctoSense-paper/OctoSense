"""Inference entrypoints for executable pipelines."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def infer_pipeline(
    model,
    inputs: torch.Tensor,
    *,
    device: str | torch.device | None = None,
):
    was_training = bool(getattr(model, "training", False))
    if device is not None:
        model = model.to(device)
        inputs = inputs.to(device)
    model.eval()
    try:
        with torch.no_grad():
            return model(inputs)
    finally:
        if was_training:
            model.train()


def infer_batches(
    model,
    batches: Iterable[torch.Tensor],
    *,
    device: str | torch.device | None = None,
) -> list[torch.Tensor]:
    return [
        infer_pipeline(model, batch, device=device)
        for batch in batches
    ]


__all__ = [
    "infer_batches",
    "infer_pipeline",
]
