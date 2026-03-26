"""Internal weight resolution helpers for canonical model ids."""

from __future__ import annotations

from octosense.models.weights.manifests import DEFAULT_WEIGHTS_BY_MODEL, WEIGHT_MANIFESTS, WeightManifest


def get_default_weights_id(model_id: str) -> str | None:
    return DEFAULT_WEIGHTS_BY_MODEL.get(model_id)


def normalize_weights_id(
    model_id: str,
    *,
    weights_id: str | None = None,
) -> str | None:
    if weights_id is None:
        return None
    normalized = str(weights_id).strip()

    manifest = WEIGHT_MANIFESTS.get(normalized)
    if manifest is None:
        supported = ", ".join(sorted(WEIGHT_MANIFESTS))
        raise ValueError(f"Unknown weights_id '{normalized}'. Supported: {supported}")
    if manifest.model_id != model_id:
        raise ValueError(
            f"weights_id '{normalized}' belongs to model '{manifest.model_id}', not '{model_id}'"
        )
    return normalized


def get_weight_manifest(
    model_id: str,
    *,
    weights_id: str | None = None,
) -> WeightManifest | None:
    normalized_weights_id = normalize_weights_id(
        model_id,
        weights_id=weights_id,
    )
    if normalized_weights_id is None:
        return None
    return WEIGHT_MANIFESTS[normalized_weights_id]


def resolve_factory_weights(
    model_id: str,
    *,
    weights_id: str | None = None,
) -> object | None:
    manifest = get_weight_manifest(model_id, weights_id=weights_id)
    if manifest is None:
        return None
    return manifest.factory_weights
