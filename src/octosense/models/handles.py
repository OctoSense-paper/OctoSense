"""Generalized model handles for canonical model loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from octosense.models.boundary import (
    ModelBoundarySpec,
    ModelInputContract,
    ModelOutputContract,
    can_materialize_model,
    describe_model_boundary,
    infer_entry_overrides_from_sample,
)
from octosense.models.registry import (
    _build_model,
    _get_model_registration,
    _normalize_entry_overrides,
)
from octosense.models.weights.loaders import normalize_weights_id


@dataclass(frozen=True)
class ModelHandle:
    """Deferred model specification resolved against boundary overrides."""

    model_id: str
    weights_id: str | None = None
    entry_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_model_id = _get_model_registration(self.model_id).model_id
        merged_overrides = _normalize_entry_overrides(
            normalized_model_id,
            entry_overrides=self.entry_overrides,
        )
        resolved_weights_id = normalize_weights_id(
            normalized_model_id,
            weights_id=self.weights_id,
        )
        object.__setattr__(self, "model_id", normalized_model_id)
        object.__setattr__(self, "weights_id", resolved_weights_id)
        object.__setattr__(self, "entry_overrides", merged_overrides)

    @property
    def boundary(self) -> ModelBoundarySpec:
        return describe_model_boundary(
            self.model_id,
            entry_overrides=self.entry_overrides,
        )

    @property
    def unresolved_entry_overrides(self) -> tuple[str, ...]:
        return self.boundary.unresolved_entry_overrides

    def get_input_contract(self) -> ModelInputContract:
        return self.boundary.input_contract

    def get_output_contract(self) -> ModelOutputContract:
        return self.boundary.output_contract

    def can_instantiate(self) -> bool:
        return can_materialize_model(
            self.model_id,
            entry_overrides=self.entry_overrides,
        )

    def with_entry_overrides(self, **kwargs: Any) -> ModelHandle:
        return ModelHandle(
            model_id=self.model_id,
            weights_id=self.weights_id,
            entry_overrides=_normalize_entry_overrides(
                self.model_id,
                entry_overrides=self.entry_overrides,
                **kwargs,
            ),
        )


def instantiate_model_handle(
    handle: ModelHandle,
    sample_tensor: torch.Tensor | None = None,
    num_classes: int | None = None,
) -> nn.Module:
    """Instantiate a deferred handle once enough contract overrides are known."""

    resolved_entry_overrides = infer_entry_overrides_from_sample(
        handle.model_id,
        sample_tensor,
        entry_overrides=handle.entry_overrides,
        num_classes=num_classes,
    )
    return _build_model(
        handle.model_id,
        weights_id=handle.weights_id,
        entry_overrides=resolved_entry_overrides,
    )


__all__ = ["ModelHandle", "instantiate_model_handle"]
