"""Human-readable catalog metadata for canonical model identifiers."""

from __future__ import annotations

from typing import Any

from octosense.models.boundary import describe_model_boundary
from octosense.models.registry import _get_model_registration, _list_registered_model_ids
from octosense.models.weights.manifests import WEIGHT_MANIFESTS


def _list_models() -> list[str]:
    return _list_registered_model_ids()


def _get_model_card(model_id: str) -> dict[str, Any]:
    registration = _get_model_registration(model_id)
    boundary = describe_model_boundary(registration.model_id)
    return {
        "model_id": registration.model_id,
        "family": registration.family,
        "task": registration.task,
        "description": registration.description,
        "default_weights_id": registration.default_weights_id,
        "available_weights": [
            weights_id
            for weights_id, manifest in sorted(WEIGHT_MANIFESTS.items())
            if manifest.model_id == registration.model_id
        ],
        "required_entry_overrides": list(boundary.required_entry_overrides),
        "optional_entry_overrides": list(boundary.optional_entry_overrides),
        "input_contract": boundary.input_contract.to_dict(),
        "output_contract": boundary.output_contract.to_dict(),
    }


_MODEL_CARDS: dict[str, dict[str, Any]] = {
    model_id: _get_model_card(model_id)
    for model_id in _list_models()
}
