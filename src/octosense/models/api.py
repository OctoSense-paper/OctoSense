"""Canonical public API for model loading."""

from __future__ import annotations

from typing import Any

from octosense.models.handles import ModelHandle, instantiate_model_handle
from octosense.models.registry import _get_model_registration, _normalize_entry_overrides
from octosense.models.weights.loaders import normalize_weights_id


def _build_handle(
    model_id: str,
    *,
    weights_id: str | None = None,
    entry_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelHandle:
    normalized_model_id = _get_model_registration(model_id).model_id
    normalized_entry_overrides = _normalize_entry_overrides(
        normalized_model_id,
        entry_overrides=entry_overrides,
        **kwargs,
    )
    normalized_weights_id = normalize_weights_id(
        normalized_model_id,
        weights_id=weights_id,
    )
    return ModelHandle(
        model_id=normalized_model_id,
        weights_id=normalized_weights_id,
        entry_overrides=normalized_entry_overrides,
    )


def load(
    model_id: str,
    *,
    weights_id: str | None = None,
    entry_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Resolve the canonical model surface.

    Returns an instantiated module once boundary-owned materialization requirements
    are satisfied; otherwise returns a deferred ``ModelHandle``.
    """

    handle = _build_handle(
        model_id,
        weights_id=weights_id,
        entry_overrides=entry_overrides,
        **kwargs,
    )
    if handle.can_instantiate():
        return instantiate_model_handle(handle)
    return handle


__all__ = ["load"]
