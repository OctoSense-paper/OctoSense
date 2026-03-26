"""Reference normalization and lowering for benchmark specs."""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from octosense.benchmarks.protocols import (
    canonical_execution_adapter_for_task_kind,
    resolve_runtime_protocol_payload,
)
from octosense.specs.schemas.benchmark import BenchmarkSpec
from octosense.specs.schemas.transform import TransformStepSpec
from octosense.tasks import load as load_task


@dataclass(frozen=True, slots=True)
class ReferenceResolverContext:
    """Injected cross-owner lookups needed during canonical reference resolution."""

    load_transform_preset: Callable[[str], Mapping[str, Any] | None] | None = None
    resolve_model_input_contract: Callable[[BenchmarkSpec], Mapping[str, Any] | None] | None = None


@dataclass(frozen=True, slots=True)
class _PresetVariantSpec:
    """One named preset variant resolved from its sidecar document."""

    steps: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class _PresetSpec:
    """Canonical in-memory shape for one transform preset owner."""

    preset_id: str
    defaults: dict[str, Any] = field(default_factory=dict)
    contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    variants: dict[str, _PresetVariantSpec] = field(default_factory=dict)
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "_PresetSpec":
        payload = dict(payload or {})
        preset_id = str(payload.get("preset_id", "") or "").strip()
        if not preset_id:
            raise ValueError("PresetSpec.preset_id must be a non-empty string")
        variants_payload = payload.get("variants", {})
        if not isinstance(variants_payload, Mapping):
            raise TypeError("PresetSpec.variants must be a mapping")
        variants = {
            str(name): _PresetVariantSpec(
                steps=tuple(
                    _mapping_list(entry.get("steps"), field_name=f"variants.{name}.steps")
                )
            )
            for name, entry in variants_payload.items()
            for entry in [_mapping(entry, field_name=f"variants.{name}")]
        }
        if not variants:
            raise ValueError(f"PresetSpec {preset_id!r} must define at least one variant")
        return cls(
            preset_id=preset_id,
            defaults=_mapping(payload.get("defaults"), field_name="defaults"),
            contracts=_mapping_of_mappings(payload.get("contracts"), field_name="contracts"),
            variants=variants,
            backends=_mapping_of_mappings(payload.get("backends"), field_name="backends"),
            metadata=_mapping(payload.get("metadata"), field_name="metadata"),
        )


def resolve_references(
    spec: BenchmarkSpec,
    *,
    context: ReferenceResolverContext | None = None,
) -> BenchmarkSpec:
    """Normalize identifiers and lower declarative references into canonical payloads."""

    resolved = BenchmarkSpec.from_dict(spec.to_dict())
    resolved.api_version = resolved.api_version.strip()
    resolved.kind = resolved.kind.strip()
    resolved.dataset.dataset_id = resolved.dataset.dataset_id.strip()
    resolved.dataset.modalities = [item.strip() for item in resolved.dataset.modalities if item.strip()]
    resolved.dataset.variant = _strip_optional(resolved.dataset.variant)
    resolved.dataset.split_scheme = _strip_optional(resolved.dataset.split_scheme)
    resolved.dataset.dataset_root = _strip_optional(resolved.dataset.dataset_root)
    resolved.task.task_id = resolved.task.task_id.strip()
    resolved.task.task_binding = resolved.task.task_binding.strip()
    resolved.transform.preset_id = _strip_optional(resolved.transform.preset_id)
    for step in resolved.transform.steps:
        step.operator_id = step.operator_id.strip()
    resolved.model.model_id = resolved.model.model_id.strip()
    resolved.model.weights_id = _strip_optional(resolved.model.weights_id)
    resolved.runtime.device = resolved.runtime.device.strip()
    if resolved.ingestion is not None:
        resolved.ingestion.reader_id = resolved.ingestion.reader_id.strip().lower()
        resolved.ingestion.source_root = _strip_optional(resolved.ingestion.source_root)
        resolved.ingestion.file_glob = _strip_optional(resolved.ingestion.file_glob)
        resolved.ingestion.label_source = _strip_optional(resolved.ingestion.label_source)
    _resolve_transform_references(resolved, context=context)
    resolved.protocol = _resolve_benchmark_protocol(resolved)
    return resolved


def require_canonical_preset_id(preset_id: str) -> str:
    """Validate that callers pass a canonical preset id instead of a sidecar path."""

    canonical_preset_id = str(preset_id).strip()
    if not canonical_preset_id:
        raise ValueError("preset_id must be a non-empty canonical id")
    if canonical_preset_id.endswith((".yaml", ".yml")):
        raise ValueError(
            f"preset_id must be a canonical id, not a sidecar path: {preset_id!r}"
        )
    parts = PurePosixPath(canonical_preset_id).parts
    if (
        not parts
        or PurePosixPath(canonical_preset_id).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in canonical_preset_id
    ):
        raise ValueError(f"preset_id must be a canonical slash-delimited id: {preset_id!r}")
    return canonical_preset_id


def load_preset_spec(preset_id: str) -> _PresetSpec:
    """Load one canonical transform preset sidecar by preset_id."""

    canonical_preset_id = require_canonical_preset_id(preset_id)
    return _cached_load_preset_spec(canonical_preset_id)


def load_transform_preset_payload(preset_id: str) -> dict[str, Any]:
    """Load one canonical transform preset sidecar payload as a plain mapping."""

    spec = load_preset_spec(preset_id)
    return {
        "preset_id": spec.preset_id,
        "defaults": dict(spec.defaults),
        "contracts": {name: dict(value) for name, value in spec.contracts.items()},
        "variants": {
            name: {"steps": [dict(step) for step in variant.steps]}
            for name, variant in spec.variants.items()
        },
        "backends": {name: dict(value) for name, value in spec.backends.items()},
        "metadata": dict(spec.metadata),
    }


def _resolve_transform_references(
    spec: BenchmarkSpec,
    *,
    context: ReferenceResolverContext | None,
) -> None:
    transform = spec.transform
    if transform.preset_id:
        transform.steps = _lower_preset_steps(spec, context=context)
        transform.preset_id = None
        transform.params = {}
        return
    transform.steps = [
        TransformStepSpec(
            operator_id=step.operator_id,
            params=dict(step.params),
        )
        for step in transform.steps
    ]


def _lower_preset_steps(
    spec: BenchmarkSpec,
    *,
    context: ReferenceResolverContext | None,
) -> list[TransformStepSpec]:
    preset_id = _require_preset_id(spec.transform.preset_id)
    preset = _load_preset_spec(preset_id, context=context)
    variant_id = _resolve_preset_variant_id(spec, preset, context=context)
    params = dict(spec.transform.params)
    contract = _resolve_model_input_contract(spec, context=context)
    params = prepare_preset_variant_params(
        preset,
        variant_id,
        params=params,
        output_contract=contract,
    )
    return [
        TransformStepSpec.from_dict(step)
        for step in resolved_variant_steps(preset, variant_id, params=params)
    ]


def _require_preset_id(preset_id: str | None) -> str:
    if not preset_id:
        raise ValueError("Transform preset lowering requires a non-empty preset_id")
    return require_canonical_preset_id(preset_id)


def _load_preset_spec(
    preset_id: str,
    *,
    context: ReferenceResolverContext | None,
) -> _PresetSpec:
    if context is None or context.load_transform_preset is None:
        return load_preset_spec(preset_id)
    payload = context.load_transform_preset(preset_id)
    return _PresetSpec.from_dict(_coerce_preset_payload(payload, preset_id=preset_id))


def _resolve_preset_variant_id(
    spec: BenchmarkSpec,
    preset: _PresetSpec,
    *,
    context: ReferenceResolverContext | None,
) -> str:
    if len(preset.variants) == 1:
        return next(iter(preset.variants))

    contract = _resolve_model_input_contract(spec, context=context)
    candidates = [
        variant_id
        for variant_id in preset.variants
        if _contract_profile_matches(preset.contracts.get(variant_id), contract)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "Cannot resolve transform preset "
            f"{preset.preset_id!r} for model {spec.model.model_id!r}: "
            f"no preset variant matches model input contract axes={tuple(contract.get('axes', ()))!r}, "
            f"dtype_kind={contract.get('dtype_kind')!r}, layout={contract.get('layout')!r}."
        )
    raise ValueError(
        f"Cannot resolve transform preset {preset.preset_id!r}: ambiguous contract match {candidates!r} "
        f"for model {spec.model.model_id!r}."
    )


def _resolve_model_input_contract(
    spec: BenchmarkSpec,
    *,
    context: ReferenceResolverContext | None,
) -> dict[str, Any]:
    if context is None or context.resolve_model_input_contract is None:
        raise ValueError(
            f"Model {spec.model.model_id!r} requires an injected input-contract resolver during resolution"
        )
    payload = context.resolve_model_input_contract(spec)
    return _mapping(payload, field_name="model_input_contract")


def _contract_profile_matches(profile: dict[str, Any] | None, contract: dict[str, Any]) -> bool:
    if not profile:
        return False
    expected_axes = tuple(str(item) for item in profile.get("axes", ()))
    actual_axes = tuple(str(item) for item in contract.get("axes", ()))
    if expected_axes and expected_axes != actual_axes:
        return False
    expected_dtype = profile.get("dtype_kind")
    actual_dtype = contract.get("dtype_kind")
    if expected_dtype is not None and expected_dtype != actual_dtype:
        return False
    expected_layout = str(profile.get("layout", "") or "").strip()
    actual_layout = str(contract.get("layout", "") or "").strip()
    if expected_layout and expected_layout != actual_layout:
        return False
    return True


def resolved_variant_steps(
    spec: _PresetSpec,
    variant_id: str,
    *,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve placeholders and conditional flags for one preset variant."""

    try:
        variant = spec.variants[variant_id]
    except KeyError as exc:
        raise KeyError(
            f"Preset variant {variant_id!r} is not defined for preset {spec.preset_id!r}"
        ) from exc
    context: dict[str, Any] = dict(spec.defaults)
    if params:
        context.update(dict(params))

    resolved_steps: list[dict[str, Any]] = []
    for step in variant.steps:
        condition = step.get("enabled_if")
        if condition is not None and not bool(_resolve_template(condition, context)):
            continue
        resolved = _resolve_template(step, context)
        if not isinstance(resolved, dict):
            raise TypeError(f"Resolved preset step for variant {variant_id!r} must be a mapping")
        resolved.pop("enabled_if", None)
        operator_id = str(resolved.get("operator_id", "") or "").strip()
        if not operator_id:
            raise ValueError(
                f"Preset {spec.preset_id!r} variant {variant_id!r} contains a step without operator_id"
            )
        resolved["operator_id"] = operator_id
        resolved_steps.append(resolved)
    return resolved_steps


def prepare_preset_variant_params(
    spec: _PresetSpec,
    variant_id: str,
    *,
    params: Mapping[str, Any] | None = None,
    output_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare compiler params using one shared preset-derivation rule."""

    prepared = dict(params or {})
    contract_payload = _mapping(output_contract, field_name="output_contract")
    if contract_payload:
        if _variant_uses_placeholder(spec, variant_id, "output_contract"):
            prepared.setdefault("output_contract", contract_payload)
        _inject_derived_preset_params(
            spec,
            variant_id,
            params=prepared,
            output_contract=contract_payload,
        )
    return prepared


def _resolve_template(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        if value.startswith("$"):
            return _lookup(context, value[1:])
        return value
    if isinstance(value, list):
        return [_resolve_template(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_template(item, context) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _resolve_template(item, context) for key, item in value.items()}
    return value


def _lookup(context: Mapping[str, Any], path: str) -> Any:
    current: Any = context
    for segment in path.split("."):
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
            continue
        raise KeyError(f"Unknown preset placeholder ${path}")
    return current


def _variant_uses_placeholder(spec: _PresetSpec, variant_id: str, placeholder: str) -> bool:
    for step in spec.variants[variant_id].steps:
        if _contains_placeholder(step.get("params"), placeholder):
            return True
    return False


def _contains_placeholder(value: Any, placeholder: str) -> bool:
    if isinstance(value, str):
        return value == f"${placeholder}" or value.startswith(f"${placeholder}.")
    if isinstance(value, list):
        return any(_contains_placeholder(item, placeholder) for item in value)
    if isinstance(value, dict):
        return any(_contains_placeholder(item, placeholder) for item in value.values())
    return False


def _inject_derived_preset_params(
    spec: _PresetSpec,
    variant_id: str,
    *,
    params: dict[str, Any],
    output_contract: Mapping[str, Any],
) -> None:
    if "target_time_bins" in params or not _variant_uses_placeholder(
        spec, variant_id, "target_time_bins"
    ):
        return
    fixed_sizes = output_contract.get("fixed_sizes", {})
    if not isinstance(fixed_sizes, Mapping):
        return
    target_size_axis = (
        _mapping(spec.backends.get("cuda"), field_name="backends.cuda")
        .get(variant_id, {})
        .get("target_size_axis")
    )
    if target_size_axis is None:
        return
    target_size = fixed_sizes.get(str(target_size_axis))
    if target_size is not None:
        params["target_time_bins"] = int(target_size)


def _strip_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _resolve_benchmark_protocol(spec: BenchmarkSpec) -> dict[str, Any]:
    task_spec = load_task(spec.task.task_id)
    return resolve_runtime_protocol_payload(
        _mapping(spec.protocol, field_name="protocol"),
        task_kind=task_spec.kind,
        primary_metric=str(task_spec.output_schema.primary_metric),
        target_fields=tuple(str(field) for field in task_spec.target_schema.fields),
        execution_adapter=canonical_execution_adapter_for_task_kind(task_spec.kind),
        field_path="BenchmarkSpec.protocol",
    )


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _mapping_of_mappings(value: object, *, field_name: str) -> dict[str, dict[str, Any]]:
    parent = _mapping(value, field_name=field_name)
    return {
        name: _mapping(child, field_name=f"{field_name}.{name}")
        for name, child in parent.items()
    }


def _mapping_list(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [
        _mapping(entry, field_name=f"{field_name}[{index}]")
        for index, entry in enumerate(value)
    ]


@lru_cache(maxsize=None)
def _cached_load_preset_spec(preset_id: str) -> _PresetSpec:
    spec_path = _resolve_transform_preset_path(preset_id)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    return _PresetSpec.from_dict(_coerce_preset_payload(payload, preset_id=preset_id, spec_path=spec_path))


def _resolve_transform_preset_path(preset_id: str) -> Path:
    sidecar_root = Path(__file__).resolve().parents[2] / "transforms" / "presets"
    return (sidecar_root / PurePosixPath(preset_id)).with_suffix(".yaml")


def _coerce_preset_payload(
    payload: object,
    *,
    preset_id: str,
    spec_path: Path | None = None,
) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        location = f" sidecar {spec_path}" if spec_path is not None else ""
        raise TypeError(
            f"Transform preset{location} for {preset_id!r} must decode to a mapping"
        )
    return {str(key): value for key, value in payload.items()}


__all__ = [
    "ReferenceResolverContext",
    "load_transform_preset_payload",
    "load_preset_spec",
    "prepare_preset_variant_params",
    "require_canonical_preset_id",
    "resolve_references",
    "resolved_variant_steps",
]
