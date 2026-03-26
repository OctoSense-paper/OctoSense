"""Sampler construction helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler


def build_sampler(
    dataset,
    shuffle: bool,
    *,
    weights: Sequence[float] | torch.Tensor | None = None,
):
    if weights is not None:
        tensor = weights if torch.is_tensor(weights) else torch.as_tensor(list(weights), dtype=torch.double)
        if tensor.numel() != len(dataset):
            raise ValueError(
                "Sampler weights must match dataset length, "
                f"got {tensor.numel()} weights for {len(dataset)} samples"
            )
        return WeightedRandomSampler(weights=tensor, num_samples=len(dataset), replacement=True)
    if shuffle:
        return RandomSampler(dataset)
    return SequentialSampler(dataset)


__all__ = ["build_sampler"]
