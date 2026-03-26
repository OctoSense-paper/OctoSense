"""Canonical package-level entrypoints for pipeline loading and public inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from octosense.io.tensor import RadioTensor
from octosense.models import ModelHandle
from octosense.pipelines.execution.infer import infer_batches, infer_pipeline

from .builder import (
    _apply_inference_transform,
    _instantiate_trainable_model,
    _materialize_inference_tensor,
    load,
)


def _chunk_samples(samples: Sequence[torch.Tensor], *, batch_size: int | None) -> list[torch.Tensor]:
    if not samples:
        raise ValueError("pipelines.infer(...) requires at least one input sample.")
    resolved_batch_size = len(samples) if batch_size is None else int(batch_size)
    if resolved_batch_size <= 0:
        raise ValueError(f"pipelines.infer(..., batch_size=...) must be positive, got {resolved_batch_size}")
    batches: list[torch.Tensor] = []
    for start in range(0, len(samples), resolved_batch_size):
        chunk = samples[start : start + resolved_batch_size]
        batches.append(torch.stack(chunk, dim=0))
    return batches


def infer(
    inputs: RadioTensor | torch.Tensor | Sequence[RadioTensor | torch.Tensor],
    *,
    model_id: str,
    transform: object | None = None,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    weights_id: str | None = None,
    entry_overrides: dict[str, Any] | None = None,
):
    """Run canonical public inference without reaching into internal dataloading helpers."""

    model_handle = ModelHandle(
        model_id=model_id,
        weights_id=weights_id,
        entry_overrides=dict(entry_overrides or {}),
    )

    if isinstance(inputs, (RadioTensor, torch.Tensor)):
        sample_tensor = _materialize_inference_tensor(
            _apply_inference_transform(inputs, transform)
        )
        model = _instantiate_trainable_model(model_handle, sample_tensor)
        prepared = sample_tensor.unsqueeze(0)
        return infer_pipeline(model, prepared, device=device)

    transformed = [
        _materialize_inference_tensor(_apply_inference_transform(sample, transform))
        for sample in inputs
    ]
    batches = _chunk_samples(transformed, batch_size=batch_size)
    model = _instantiate_trainable_model(model_handle, transformed[0])
    outputs = infer_batches(model, batches, device=device)
    if batch_size is None or len(transformed) <= batch_size:
        return outputs[0]
    return outputs


__all__ = ["infer", "load"]
