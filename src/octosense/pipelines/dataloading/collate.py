"""Canonical collate functions owned by the pipelines execution kernel."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, Literal

import torch

TargetContractKind = Literal["scalar_index", "structured_tensor_mapping"]


def _as_model_entry_tensor(sample: Any) -> torch.Tensor:
    if torch.is_tensor(sample):
        return sample
    to_tensor = getattr(sample, "to_tensor", None)
    if callable(to_tensor):
        return to_tensor(contiguous=True)
    raise TypeError(
        "Pipeline collate expects torch.Tensor or tensor-like samples exposing "
        f"to_tensor(contiguous=True), got {type(sample)!r}."
    )


def collate_scalar_index_batch(
    batch: list[tuple[Any, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate model-entry tensors with scalar-index targets."""

    samples, labels = zip(*batch, strict=True)
    return torch.stack([_as_model_entry_tensor(sample) for sample in samples], dim=0), torch.tensor(
        labels,
        dtype=torch.long,
    )


def collate_structured_target_batch(
    batch: list[tuple[Any, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Collate model-entry tensors with structured tensor-mapping targets."""

    samples, targets = zip(*batch, strict=True)
    expected_key_set = {str(key) for key in targets[0].keys()}
    expected_keys = tuple(sorted(expected_key_set))
    merged: dict[str, list[torch.Tensor]] = {key: [] for key in expected_keys}
    for target in targets:
        current_key_set = {str(key) for key in target.keys()}
        if current_key_set != expected_key_set:
            raise KeyError(
                "Structured-target batches require consistent target fields across samples, "
                f"expected keys {expected_keys!r}, got {tuple(sorted(current_key_set))!r}."
            )
        for key in expected_keys:
            merged[key].append(target[key])
    return torch.stack([_as_model_entry_tensor(sample) for sample in samples], dim=0), {
        key: torch.stack(values, dim=0) for key, values in merged.items()
    }


__all__ = [
    "collate_scalar_index_batch",
    "collate_structured_target_batch",
    "TargetContractKind",
]
