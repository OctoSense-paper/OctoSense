"""Internal canonical recipe parsing for ``octosense.pipelines.load(...)``."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from octosense.tasks import load as load_task


@dataclass(frozen=True)
class _PipelineRecipe:
    """Canonical recipe selector resolved into explicit pipeline owners."""

    recipe_id: str
    dataset_id: str
    model_id: str
    task_id: str
    weights_id: str | None = None
    modalities: tuple[str, ...] | None = None
    variant: str | None = None
    path: str | None = None
    input_selection: dict[str, object] | None = None


def _normalize_token(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string when provided")
    return normalized


def _resolve_task_id(task_id: str) -> str:
    normalized = _normalize_token("task_id", task_id)
    assert normalized is not None
    return load_task(normalized).task_id


def _normalize_modalities(value: object | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise TypeError("modalities must be a string or sequence of strings when provided")
    normalized = tuple(str(item).strip() for item in items if str(item).strip())
    return normalized or None


def _resolve_override(overrides: dict[str, Any], key: str, fallback: Any) -> Any:
    value = overrides.get(key, fallback)
    return fallback if value is None else value


def _reject_identity_overrides(overrides: dict[str, Any]) -> None:
    forbidden = tuple(
        key
        for key in ("dataset_id", "model_id", "task_id", "weights_id")
        if overrides.get(key) is not None
    )
    if not forbidden:
        return
    names = ", ".join(forbidden)
    raise ValueError(
        "Canonical recipe_id is an immutable identity '<dataset_id>/<model_id>@<task_id>' "
        f"and cannot be combined with same-level selector overrides: {names}."
    )


def _parse_pipeline_recipe(recipe_id: str) -> _PipelineRecipe:
    """Parse a canonical composite recipe id like ``widar3/rfnet@classification/gesture``."""

    normalized = _normalize_token("recipe_id", recipe_id)
    assert normalized is not None

    if "@" not in normalized:
        raise ValueError(
            "Canonical pipeline recipe ids must use '<dataset_id>/<model_id>@<task_id>' syntax"
        )
    dataset_model_selector, task_id = normalized.rsplit("@", 1)
    if "/" not in dataset_model_selector:
        raise ValueError(
            "Canonical pipeline recipe ids must include both dataset_id and model_id "
            "before '@', for example 'widar3/rfnet@classification/gesture'"
        )

    dataset_id, model_selector = dataset_model_selector.split("/", 1)
    dataset_id = _normalize_token("dataset_id", dataset_id)
    model_selector = _normalize_token("model selector", model_selector)
    assert dataset_id is not None
    assert model_selector is not None

    if ":" in model_selector:
        raise ValueError(
            "Canonical pipeline recipe ids only accept '<dataset_id>/<model_id>@<task_id>' syntax. "
            "Pass weights_id=... as a separate selector instead of encoding it in recipe_id."
        )

    return _PipelineRecipe(
        recipe_id=normalized,
        dataset_id=_normalize_token("dataset_id", dataset_id),
        model_id=_normalize_token("model_id", model_selector),
        task_id=_resolve_task_id(task_id),
    )


def _resolve_pipeline_recipe(recipe_id: str, **overrides: Any) -> _PipelineRecipe:
    """Resolve a recipe id plus explicit selector overrides into one canonical recipe."""

    _reject_identity_overrides(overrides)
    recipe = _parse_pipeline_recipe(recipe_id)
    dataset_id = _normalize_token("dataset_id", recipe.dataset_id)
    model_id = _normalize_token("model_id", recipe.model_id)
    task_id = recipe.task_id
    variant = _normalize_token("variant", _resolve_override(overrides, "variant", recipe.variant))
    modalities = _normalize_modalities(_resolve_override(overrides, "modalities", recipe.modalities))
    path = _resolve_override(overrides, "path", recipe.path)
    resolved_path = None if path is None else str(path).strip()
    input_selection = _resolve_override(overrides, "input_selection", recipe.input_selection)
    if input_selection is not None and not isinstance(input_selection, Mapping):
        raise TypeError("input_selection must be a mapping when provided")
    resolved_selection = dict(input_selection) if isinstance(input_selection, Mapping) else None

    return _PipelineRecipe(
        recipe_id=recipe.recipe_id,
        dataset_id=dataset_id,
        model_id=model_id,
        task_id=task_id,
        weights_id=recipe.weights_id,
        modalities=modalities,
        variant=variant,
        path=resolved_path or None,
        input_selection=resolved_selection,
    )

__all__: list[str] = []
