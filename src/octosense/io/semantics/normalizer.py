"""Normalize canonical IO semantic registry and binding structures."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import re
from typing import Any, Callable, Mapping

_UPPER_SNAKE_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
_LOWER_SNAKE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_SEMANTIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9_]*)+$")
_BINDING_STATUS_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_SECTION_KIND = {
    "axes": "axis",
    "fields": "field",
    "tensors": "tensor",
    "provenance": "provenance",
    "representation": "representation",
}


def ensure_upper_snake_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _UPPER_SNAKE_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be UPPER_SNAKE_CASE, got {value!r}")
    return value


def ensure_lower_snake_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _LOWER_SNAKE_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be lower_snake_case, got {value!r}")
    return value


def ensure_semantic_id(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _SEMANTIC_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must be a dotted semantic id such as 'octo.common.axis.time', got {value!r}"
        )
    return value


@dataclass(frozen=True)
class BindingEntry:
    source: str
    target: str | None
    semantic: str | None
    via: str | None
    status: str | None
    source_namespace: str | None
    scale: float | None
    unit: str | None
    description: str | None

    @property
    def name(self) -> str:
        return self.source

    @property
    def converter(self) -> str | None:
        return self.via


@dataclass(frozen=True)
class ReaderBinding:
    signal_source: str
    signal_dims: tuple[str, ...]
    axes: dict[str, BindingEntry]
    fields: dict[str, BindingEntry]


@dataclass(frozen=True)
class ResolvedBinding:
    signal_source: str
    signal_dims: tuple[str, ...]
    axes: dict[str, "ResolvedBindingTarget"]
    fields: dict[str, "ResolvedBindingTarget"]


@dataclass(frozen=True)
class ResolvedBindingTarget:
    semantic_id: str
    preferred_name: str
    code: str
    kind: str | None

    @property
    def exported_name(self) -> str:
        return self.preferred_name


@dataclass(frozen=True)
class SemanticEntry:
    semantic_id: str
    code: str
    preferred_name: str
    aliases: tuple[str, ...] = ()
    kind: str | None = None
    status: str | None = None
    category: str | None = None
    unit: str | None = None
    dtype: str | None = None
    source_namespace: str | None = None
    description: str | None = None
    axis_semantics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def canonical_name(self) -> str:
        return self.preferred_name

    @property
    def identifier(self) -> str:
        return self.code

    @property
    def scope(self) -> str | None:
        return self.kind


@dataclass(frozen=True)
class ReaderMetadataSpec:
    reader_id: str
    modality: str
    device_family: str
    reader_class: str | None
    hardware_vendor: str | None
    hardware_family: str | None
    source_type: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ReaderDeviceSpec:
    modality: str
    device_family: str
    device_name: str
    reader_version: str | None
    validation: dict[Any, Any]
    signal_defaults: dict[Any, Any]
    extras: dict[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "device_name": self.device_name,
            "validation": dict(self.validation),
            "signal_defaults": dict(self.signal_defaults),
        }
        if self.reader_version is not None:
            payload["reader_version"] = self.reader_version
        payload.update(self.extras)
        return payload


def normalize_binding(binding: Mapping[str, Any]) -> ReaderBinding:
    signal = binding.get("signal", {})
    if not isinstance(signal, Mapping):
        raise ValueError("binding.signal must be a mapping")
    source = signal.get("source")
    if not isinstance(source, str) or not source.strip():
        source = "data"
    dims = signal.get("dims", [])
    if not isinstance(dims, list) or not dims:
        axis_payload = binding.get("axes", binding.get("axis", {}))
        if isinstance(axis_payload, Mapping) and axis_payload:
            dims = list(axis_payload.keys())
        else:
            dims = ["time"]
    normalized_dims = tuple(
        ensure_lower_snake_identifier(str(dim), label=f"binding.signal.dims[{idx}]")
        for idx, dim in enumerate(dims)
    )
    return ReaderBinding(
        signal_source=source.strip(),
        signal_dims=normalized_dims,
        axes=_normalize_section(binding.get("axes", binding.get("axis", {})), section="axes"),
        fields=_normalize_section(binding.get("fields", binding.get("field", {})), section="fields"),
    )


def normalize_aliases(
    payload: Mapping[str, Any],
    *,
    modality: str | None = None,
    semantic_registry: Mapping[str, SemanticEntry] | None = None,
) -> dict[str, str]:
    raw_aliases = payload.get("aliases", payload)
    if raw_aliases in (None, {}):
        normalized: dict[str, str] = {}
    else:
        if not isinstance(raw_aliases, Mapping):
            raise ValueError("aliases.yaml must define a mapping or an {aliases: ...} payload")
        normalized = _normalize_alias_mapping(raw_aliases, prefix="aliases")
    if modality is not None:
        modality_payload = payload.get("modalities", {})
        if modality_payload not in (None, {}):
            if not isinstance(modality_payload, Mapping):
                raise ValueError("aliases.yaml modalities must be a mapping when provided")
            raw_modality_aliases = modality_payload.get(modality, {})
            if raw_modality_aliases not in (None, {}):
                if not isinstance(raw_modality_aliases, Mapping):
                    raise ValueError(f"aliases.yaml modalities.{modality} must be a mapping")
                normalized.update(
                    _normalize_alias_mapping(
                        raw_modality_aliases,
                        prefix=f"modalities.{modality}",
                    )
                )
    if semantic_registry is not None:
        for entry in semantic_registry.values():
            for alias in entry.aliases:
                existing = normalized.get(alias)
                if existing is not None and existing != entry.preferred_name:
                    raise ValueError(
                        f"semantic alias {alias!r} points to both {existing!r} and "
                        f"{entry.preferred_name!r}"
                    )
                normalized[alias] = entry.preferred_name
    return normalized


def _normalize_alias_mapping(
    raw_aliases: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, str]:
    if raw_aliases in (None, {}):
        return {}
    if not isinstance(raw_aliases, Mapping):
        raise ValueError(f"{prefix} must be a mapping")
    normalized: dict[str, str] = {}
    for raw_alias, raw_target in raw_aliases.items():
        alias = ensure_lower_snake_identifier(str(raw_alias), label=f"{prefix}.{raw_alias}")
        target = ensure_lower_snake_identifier(str(raw_target), label=f"{prefix}.{raw_alias}.target")
        if alias == target:
            raise ValueError(f"{prefix}.{alias} cannot point to itself")
        normalized[alias] = target
    return normalized


def canonicalize_semantic_name(
    value: str,
    *,
    aliases: Mapping[str, str],
    semantic_registry: Mapping[str, SemanticEntry] | None = None,
) -> str:
    raw_value = str(value).strip()
    if semantic_registry is not None:
        entry = _lookup_semantic_entry(raw_value, semantic_registry)
        if entry is not None:
            return entry.preferred_name
    canonical = ensure_lower_snake_identifier(raw_value, label="semantic name")
    seen = {canonical}
    while canonical in aliases:
        canonical = aliases[canonical]
        if canonical in seen:
            chain = " -> ".join(sorted(seen | {canonical}))
            raise ValueError(f"Alias cycle detected while canonicalizing semantic name: {chain}")
        seen.add(canonical)
    return canonical


def resolve_semantic_entry(
    value: str,
    *,
    semantic_registry: Mapping[str, SemanticEntry],
    kind: str | None = None,
) -> SemanticEntry | None:
    entry = _lookup_semantic_entry(value, semantic_registry)
    if entry is None:
        return None
    if kind is not None and entry.kind != kind:
        return None
    return entry


def normalize_semantic_registry(
    core_payload: Mapping[str, Any],
    modality_payload: Mapping[str, Any],
) -> dict[str, SemanticEntry]:
    registry: dict[str, SemanticEntry] = {}
    registry.update(_normalize_core_payload(core_payload))
    registry.update(_normalize_modality_payload(modality_payload))
    _validate_registry_uniqueness(registry)
    return registry


def normalize_reader_metadata(
    payload: Mapping[str, Any],
    *,
    modality: str,
    device_family: str,
) -> ReaderMetadataSpec:
    reader_id = str(payload.get("reader_id", f"{modality}/{device_family}")).strip().lower()
    if reader_id != f"{modality}/{device_family}":
        raise ValueError(
            f"metadata.reader_id must match '{modality}/{device_family}', got {reader_id!r}"
        )
    metadata_modality = str(payload.get("modality", modality)).strip().lower()
    metadata_device_family = str(payload.get("device_family", device_family)).strip().lower()
    if metadata_modality != modality:
        raise ValueError(
            f"metadata.modality must match directory modality {modality!r}, got {metadata_modality!r}"
        )
    if metadata_device_family != device_family:
        raise ValueError(
            "metadata.device_family must match directory device family "
            f"{device_family!r}, got {metadata_device_family!r}"
        )
    reader_class = payload.get("reader_class")
    if reader_class is not None and not isinstance(reader_class, str):
        raise ValueError("metadata.reader_class must be a string when provided")
    notes_payload = payload.get("notes", [])
    if notes_payload in (None, ""):
        notes: tuple[str, ...] = ()
    elif isinstance(notes_payload, list) and all(isinstance(item, str) for item in notes_payload):
        notes = tuple(notes_payload)
    else:
        raise ValueError("metadata.notes must be a list of strings")
    return ReaderMetadataSpec(
        reader_id=reader_id,
        modality=metadata_modality,
        device_family=metadata_device_family,
        reader_class=reader_class,
        hardware_vendor=_optional_string(payload.get("hardware_vendor"), label="metadata.hardware_vendor"),
        hardware_family=_optional_string(payload.get("hardware_family"), label="metadata.hardware_family"),
        source_type=_optional_string(payload.get("source_type"), label="metadata.source_type"),
        notes=notes,
    )


def normalize_reader_device_spec(
    payload: Mapping[str, Any],
    *,
    modality: str,
    device_family: str,
    device_identity: str | None = None,
) -> ReaderDeviceSpec:
    device_name = _optional_string(payload.get("device_name"), label="device.device_name")
    if device_name is None:
        device_name = (device_identity or device_family).strip()
    reader_version = _optional_string(payload.get("reader_version"), label="device.reader_version")
    validation = _normalize_device_mapping(
        payload.get("validation", {}),
        label="device.validation",
    )
    signal_defaults = _normalize_device_mapping(
        payload.get("signal_defaults", {}),
        label="device.signal_defaults",
    )
    extras: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"device_name", "reader_version", "validation", "signal_defaults"}:
            continue
        normalized_key = _required_string(key, label="device.<key>")
        extras[normalized_key] = _normalize_device_value(
            value,
            label=f"device.{normalized_key}",
        )
    return ReaderDeviceSpec(
        modality=modality,
        device_family=device_family,
        device_name=device_name,
        reader_version=reader_version,
        validation=validation,
        signal_defaults=signal_defaults,
        extras=extras,
    )


def validate_binding_against_semantics(
    binding: ReaderBinding,
    *,
    semantic_registry: Mapping[str, SemanticEntry],
    aliases: Mapping[str, str],
) -> None:
    known_preferred_names = {entry.preferred_name for entry in semantic_registry.values()}
    for section_name, section in (("axes", binding.axes), ("fields", binding.fields)):
        for entry_name, entry in section.items():
            if entry.target is not None:
                semantic = _lookup_semantic_entry(entry.target, semantic_registry)
                if semantic is None or semantic.semantic_id != entry.target:
                    raise ValueError(
                        f"binding.{section_name}.{entry_name}.target={entry.target!r} "
                        "must be a declared semantic id from semantics/core.yaml or "
                        "semantics/<modality>.yaml"
                    )
                target = canonicalize_semantic_name(
                    semantic.preferred_name,
                    aliases=aliases,
                    semantic_registry=semantic_registry,
                )
                if target not in known_preferred_names:
                    raise ValueError(
                        f"binding.{section_name}.{entry_name}.target={target!r} "
                        "is not a known canonical semantic name"
                    )
                allowed_kinds = _binding_section_allowed_kinds(section_name)
                if semantic.kind not in allowed_kinds:
                    allowed_kind_list = " | ".join(sorted(allowed_kinds))
                    raise ValueError(
                        f"binding.{section_name}.{entry_name}.target={entry.target!r} "
                        f"must resolve to one of [{allowed_kind_list}], got {semantic.kind!r}"
                    )


def validate_aliases_against_semantics(
    aliases: Mapping[str, str],
    *,
    semantic_registry: Mapping[str, SemanticEntry],
) -> None:
    known_preferred_names = {entry.preferred_name for entry in semantic_registry.values()}
    for alias, target in aliases.items():
        if alias in known_preferred_names:
            raise ValueError(
                f"aliases.{alias} must not shadow a canonical semantic name; "
                "use the canonical name directly instead"
            )
        if target not in known_preferred_names:
            raise ValueError(
                f"aliases.{alias}.target={target!r} is not declared in the canonical semantic registry"
            )
        if target in aliases:
            raise ValueError(
                f"aliases.{alias}.target={target!r} must resolve directly to a canonical name, "
                "not another alias"
            )


def resolve_binding_entry_target(
    entry: BindingEntry,
    *,
    semantic_registry: Mapping[str, SemanticEntry],
    aliases: Mapping[str, str],
) -> ResolvedBindingTarget | None:
    if entry.target is not None:
        semantic = semantic_registry.get(entry.target)
        if semantic is None:
            raise KeyError(entry.target)
        return _make_resolved_binding_target(semantic)
    return None


def resolve_binding(
    binding: ReaderBinding,
    *,
    semantic_registry: Mapping[str, SemanticEntry],
    aliases: Mapping[str, str],
) -> ResolvedBinding:
    validate_binding_against_semantics(
        binding,
        semantic_registry=semantic_registry,
        aliases=aliases,
    )
    return ResolvedBinding(
        signal_source=binding.signal_source,
        signal_dims=binding.signal_dims,
        axes=_resolve_binding_section(
            binding.axes,
            semantic_registry=semantic_registry,
            aliases=aliases,
        ),
        fields=_resolve_binding_section(
            binding.fields,
            semantic_registry=semantic_registry,
            aliases=aliases,
        ),
    )


def apply_binding(
    raw_payload: Mapping[str, Any],
    binding_map: Mapping[str, str | ResolvedBindingTarget],
    *,
    binding_entries: Mapping[str, BindingEntry] | None = None,
    binding_converters: Mapping[str, Callable[..., Any]] | None = None,
    converter_context: Mapping[str, Any] | None = None,
    known_canonical_names: set[str] | None = None,
    keep_unmapped: bool = False,
    raw_value_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    normalized_converter_context = _normalize_converter_context(raw_payload)
    normalized_converter_context.update(
        _normalize_converter_context(raw_value_payload) if raw_value_payload is not None else {}
    )
    normalized_converter_context.update(converter_context or {})
    for raw_name, value in raw_payload.items():
        normalized_raw_name = ensure_lower_snake_identifier(
            str(raw_name),
            label=f"binding input field {raw_name!r}",
        )
        binding_entry = (
            binding_entries.get(normalized_raw_name)
            if binding_entries is not None
            else None
        )
        raw_value = _binding_raw_value(
            raw_name=str(raw_name),
            normalized_raw_name=normalized_raw_name,
            raw_value_payload=raw_value_payload,
            fallback_value=value,
        )
        bound_value = raw_value
        if binding_entry is not None and binding_entry.via is not None:
            bound_value = _apply_binding_converter(
                binding_entry=binding_entry,
                raw_name=normalized_raw_name,
                value=raw_value,
                binding_converters=binding_converters,
                converter_context=normalized_converter_context,
            )
        if binding_entry is not None and binding_entry.scale is not None:
            bound_value = _apply_binding_scale(
                binding_entry=binding_entry,
                raw_name=normalized_raw_name,
                value=bound_value,
            )
        canonical_name = binding_map.get(normalized_raw_name)
        if canonical_name is None and known_canonical_names and normalized_raw_name in known_canonical_names:
            canonical_name = normalized_raw_name
        if canonical_name is None:
            if keep_unmapped:
                renamed[normalized_raw_name] = value
            continue
        exported_name = _binding_target_export_name(canonical_name)
        _store_bound_value(
            renamed,
            canonical_name=exported_name,
            raw_name=normalized_raw_name,
            value=bound_value,
        )
    return renamed


def _apply_binding_converter(
    *,
    binding_entry: BindingEntry,
    raw_name: str,
    value: Any,
    binding_converters: Mapping[str, Callable[..., Any]] | None,
    converter_context: Mapping[str, Any],
) -> Any:
    via = binding_entry.via
    if via is None or binding_converters is None:
        return value
    converter = binding_converters.get(via)
    if converter is None:
        raise ValueError(
            f"binding field {raw_name!r} references via={via!r}, but no matching converter was loaded"
        )
    return _invoke_binding_converter(
        converter,
        value=value,
        context=converter_context,
        label=f"binding field {raw_name!r}",
    )


def _apply_binding_scale(
    *,
    binding_entry: BindingEntry,
    raw_name: str,
    value: Any,
) -> Any:
    scale = binding_entry.scale
    if scale is None:
        return value
    if isinstance(value, bool):
        raise ValueError(
            f"binding field {raw_name!r} declares scale={scale!r}, but boolean values do not "
            "support declarative linear scaling"
        )
    try:
        return value * scale
    except TypeError as exc:
        raise ValueError(
            f"binding field {raw_name!r} declares scale={scale!r}, but value type "
            f"{type(value).__name__!r} does not support declarative linear scaling"
        ) from exc


def _invoke_binding_converter(
    converter: Callable[..., Any],
    *,
    value: Any,
    context: Mapping[str, Any],
    label: str,
) -> Any:
    signature = inspect.signature(converter)
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}
    consumed_value = False
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if not consumed_value:
                positional_args.append(value)
                consumed_value = True
                continue
            if parameter.name in context:
                positional_args.append(context[parameter.name])
                continue
            if parameter.default is inspect.Signature.empty:
                raise ValueError(
                    f"{label} converter {converter.__name__!r} requires argument "
                    f"{parameter.name!r}, but it is not available in converter_context"
                )
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if parameter.name in context:
                keyword_args[parameter.name] = context[parameter.name]
                continue
            if parameter.default is inspect.Signature.empty:
                raise ValueError(
                    f"{label} converter {converter.__name__!r} requires keyword argument "
                    f"{parameter.name!r}, but it is not available in converter_context"
                )
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            keyword_args.update(context)
            continue
    if not consumed_value:
        return converter(**keyword_args)
    return converter(*positional_args, **keyword_args)


def _normalize_converter_context(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    normalized: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key).strip()
        if not key:
            continue
        normalized[key] = value
        try:
            normalized_key = ensure_lower_snake_identifier(key, label=f"converter_context.{key}")
        except ValueError:
            continue
        normalized[normalized_key] = value
    return normalized
    return renamed


def _normalize_section(payload: Any, *, section: str) -> dict[str, BindingEntry]:
    if payload in (None, {}):
        return {}
    normalized: dict[str, BindingEntry] = {}
    if isinstance(payload, Mapping):
        iterator = payload.items()
    elif isinstance(payload, list):
        iterator = ((item.get("source"), item) for item in payload)
    else:
        raise ValueError(f"binding.{section} must be a mapping or a list of entries")
    for raw_name, raw_spec in iterator:
        if raw_name in (None, ""):
            raise ValueError(f"binding.{section} entries must declare a source field name")
        source = ensure_lower_snake_identifier(str(raw_name), label=f"binding.{section}.{raw_name}")
        if isinstance(raw_spec, str):
            target = raw_spec
            semantic = None
            via = None
            status = None
            source_namespace = None
            scale = None
            unit = None
            description = None
        else:
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"binding.{section}.{source} must be a string or mapping")
            if "to" in raw_spec:
                raise ValueError(
                    f"binding.{section}.{source}.to is a legacy key; use target with a semantic id"
                )
            if "semantic" in raw_spec:
                raise ValueError(
                    f"binding.{section}.{source}.semantic is a legacy key; use target with a semantic id"
                )
            if "converter" in raw_spec:
                raise ValueError(
                    f"binding.{section}.{source}.converter is a legacy key; use via"
                )
            if "keep_raw" in raw_spec:
                raise ValueError(
                    f"binding.{section}.{source}.keep_raw is no longer supported; "
                    "declare an explicit semantic or runtime field instead"
                )
            target = raw_spec.get("target")
            if target is not None:
                if not isinstance(target, str):
                    raise ValueError(f"binding.{section}.{source}.target must be a string")
                target = ensure_semantic_id(
                    target,
                    label=f"binding.{section}.{source}.target",
                )
            semantic = None
            via = raw_spec.get("via")
            if via is not None:
                via = ensure_lower_snake_identifier(
                    str(via),
                    label=f"binding.{section}.{source}.via",
                )
            status = raw_spec.get("status")
            if status is not None:
                status = _normalize_binding_status(status, label=f"binding.{section}.{source}.status")
            source_namespace = raw_spec.get("source_namespace")
            if source_namespace is not None and not isinstance(source_namespace, str):
                raise ValueError(f"binding.{section}.{source}.source_namespace must be a string")
            scale = raw_spec.get("scale")
            if scale is not None:
                scale = float(scale)
            unit = raw_spec.get("unit")
            if unit is not None and not isinstance(unit, str):
                raise ValueError(f"binding.{section}.{source}.unit must be a string")
            description = raw_spec.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError(f"binding.{section}.{source}.description must be a string")
        normalized[source] = BindingEntry(
            source=source,
            target=target,
            semantic=semantic,
            via=via,
            status=status,
            source_namespace=source_namespace,
            scale=scale,
            unit=unit,
            description=description,
        )
    return normalized


def _resolve_binding_section(
    section: Mapping[str, BindingEntry],
    *,
    semantic_registry: Mapping[str, SemanticEntry],
    aliases: Mapping[str, str],
) -> dict[str, ResolvedBindingTarget]:
    resolved: dict[str, ResolvedBindingTarget] = {}
    for entry_name, entry in section.items():
        target = resolve_binding_entry_target(
            entry,
            semantic_registry=semantic_registry,
            aliases=aliases,
        )
        if target is None:
            continue
        resolved[entry_name] = target
    return resolved


def _store_bound_value(
    payload: dict[str, Any],
    *,
    canonical_name: str,
    raw_name: str,
    value: Any,
) -> None:
    if canonical_name in payload:
        existing = payload[canonical_name]
        if existing != value:
            raise ValueError(
                f"Conflicting values map to canonical field {canonical_name!r}: "
                f"{existing!r} vs {value!r} from {raw_name!r}"
            )
        return
    payload[canonical_name] = value


def _binding_raw_value(
    *,
    raw_name: str,
    normalized_raw_name: str,
    raw_value_payload: Mapping[str, Any] | None,
    fallback_value: Any,
) -> Any:
    if raw_value_payload is None:
        return fallback_value
    if raw_name in raw_value_payload:
        return raw_value_payload[raw_name]
    if normalized_raw_name in raw_value_payload:
        return raw_value_payload[normalized_raw_name]
    return fallback_value


def _normalize_registry_payload(
    payload: Mapping[str, Any],
    *,
    namespace_label: str,
) -> dict[str, SemanticEntry]:
    registry: dict[str, SemanticEntry] = {}
    for section_name, section in payload.items():
        if section_name not in _SECTION_KIND:
            raise ValueError(
                f"{namespace_label} semantics section {section_name!r} is unsupported; "
                f"expected one of {sorted(_SECTION_KIND)}"
            )
        if section in (None, {}):
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"{namespace_label}.{section_name} must be a mapping")
        expected_kind = _SECTION_KIND[section_name]
        for raw_semantic_id, raw_spec in section.items():
            semantic_id = ensure_semantic_id(
                str(raw_semantic_id),
                label=f"{namespace_label}.{section_name}.{raw_semantic_id}",
            )
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"{namespace_label}.{section_name}.{semantic_id} must be a mapping")
            code = ensure_upper_snake_identifier(
                str(raw_spec.get("code", "")),
                label=f"{namespace_label}.{section_name}.{semantic_id}.code",
            )
            preferred_name = ensure_lower_snake_identifier(
                str(raw_spec.get("preferred_name", raw_spec.get("name", ""))),
                label=f"{namespace_label}.{section_name}.{semantic_id}.preferred_name",
            )
            kind = _optional_kind(raw_spec.get("kind"), label=f"{namespace_label}.{section_name}.{semantic_id}.kind")
            if kind is not None and kind != expected_kind:
                raise ValueError(
                    f"{namespace_label}.{section_name}.{semantic_id}.kind must be {expected_kind!r}, got {kind!r}"
                )
            aliases = _normalize_inline_aliases(
                raw_spec.get("aliases", []),
                label=f"{namespace_label}.{section_name}.{semantic_id}.aliases",
            )
            axis_semantics = _normalize_axis_semantics(
                raw_spec.get("axis_semantics", []),
                label=f"{namespace_label}.{section_name}.{semantic_id}.axis_semantics",
            )
            description = raw_spec.get("description")
            if description is not None and not isinstance(description, str):
                raise ValueError(f"{namespace_label}.{section_name}.{semantic_id}.description must be a string")
            registry[semantic_id] = SemanticEntry(
                semantic_id=semantic_id,
                code=code,
                preferred_name=preferred_name,
                aliases=aliases,
                kind=expected_kind,
                status=_optional_identifier(
                    raw_spec.get("status"),
                    label=f"{namespace_label}.{section_name}.{semantic_id}.status",
                ),
                category=_optional_identifier(
                    raw_spec.get("category"),
                    label=f"{namespace_label}.{section_name}.{semantic_id}.category",
                ),
                unit=_optional_string(
                    raw_spec.get("unit"),
                    label=f"{namespace_label}.{section_name}.{semantic_id}.unit",
                ),
                dtype=_optional_string(
                    raw_spec.get("dtype"),
                    label=f"{namespace_label}.{section_name}.{semantic_id}.dtype",
                ),
                source_namespace=_optional_string(
                    raw_spec.get("source_namespace"),
                    label=f"{namespace_label}.{section_name}.{semantic_id}.source_namespace",
                ),
                description=description,
                axis_semantics=axis_semantics,
            )
    return registry


def _normalize_core_payload(payload: Mapping[str, Any]) -> dict[str, SemanticEntry]:
    if _payload_uses_legacy_core(payload):
        raise ValueError(
            "legacy core semantic payloads are no longer supported; "
            "use canonical grouped sections with semantic ids"
        )
    return _normalize_registry_payload(payload, namespace_label="core")


def _normalize_modality_payload(payload: Mapping[str, Any]) -> dict[str, SemanticEntry]:
    if _payload_uses_legacy_modality(payload):
        raise ValueError(
            "legacy modality semantic payloads are no longer supported; "
            "use canonical grouped sections with semantic ids"
        )
    return _normalize_registry_payload(payload, namespace_label="modality")


def _payload_uses_legacy_core(payload: Mapping[str, Any]) -> bool:
    return "attrs" in payload


def _payload_uses_legacy_modality(payload: Mapping[str, Any]) -> bool:
    return not any(section_name in _SECTION_KIND for section_name in payload)


def _normalize_inline_aliases(value: Any, *, label: str) -> tuple[str, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(
        ensure_lower_snake_identifier(str(alias), label=f"{label}[{idx}]")
        for idx, alias in enumerate(value)
    )


def _normalize_axis_semantics(value: Any, *, label: str) -> tuple[str, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(
        ensure_semantic_id(str(semantic_id), label=f"{label}[{idx}]")
        for idx, semantic_id in enumerate(value)
    )


def _validate_registry_uniqueness(registry: Mapping[str, SemanticEntry]) -> None:
    codes: dict[str, str] = {}
    preferred_names: dict[str, str] = {}
    for semantic_id, entry in registry.items():
        code_owner = codes.get(entry.code)
        if code_owner is not None and code_owner != semantic_id:
            raise ValueError(
                f"semantic code {entry.code!r} is declared by both {code_owner!r} and {semantic_id!r}"
            )
        preferred_owner = preferred_names.get(entry.preferred_name)
        if preferred_owner is not None and preferred_owner != semantic_id:
            raise ValueError(
                f"preferred_name {entry.preferred_name!r} is declared by both "
                f"{preferred_owner!r} and {semantic_id!r}"
            )
        codes[entry.code] = semantic_id
        preferred_names[entry.preferred_name] = semantic_id


def _make_resolved_binding_target(entry: SemanticEntry) -> ResolvedBindingTarget:
    return ResolvedBindingTarget(
        semantic_id=entry.semantic_id,
        preferred_name=entry.preferred_name,
        code=entry.code,
        kind=entry.kind,
    )


def _binding_target_export_name(target: str | ResolvedBindingTarget) -> str:
    if isinstance(target, ResolvedBindingTarget):
        return target.exported_name
    return target


def _lookup_semantic_entry(
    value: str,
    semantic_registry: Mapping[str, SemanticEntry],
) -> SemanticEntry | None:
    raw_value = str(value).strip()
    if raw_value in semantic_registry:
        return semantic_registry[raw_value]
    by_code: dict[str, SemanticEntry] = {}
    by_preferred_name: dict[str, SemanticEntry] = {}
    for entry in semantic_registry.values():
        by_code[entry.code] = entry
        by_preferred_name[entry.preferred_name] = entry
        for alias in entry.aliases:
            by_preferred_name.setdefault(alias, entry)
    if raw_value in by_code:
        return by_code[raw_value]
    if raw_value in by_preferred_name:
        return by_preferred_name[raw_value]
    return None


def _normalize_binding_status(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _BINDING_STATUS_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be lower_snake_case, got {value!r}")
    return value


def _optional_kind(value: Any, *, label: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _optional_identifier(value: Any, *, label: str) -> str | None:
    if value in (None, ""):
        return None
    return ensure_lower_snake_identifier(str(value), label=label)


def _optional_string(value: Any, *, label: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _binding_section_allowed_kinds(section: str) -> set[str]:
    if section == "axes":
        return {"axis"}
    if section == "fields":
        return {"field", "tensor", "provenance", "representation"}
    raise ValueError(f"Unsupported binding section: {section}")


def _normalize_device_mapping(value: Any, *, label: str) -> dict[Any, Any]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    normalized: dict[Any, Any] = {}
    for raw_key, raw_value in value.items():
        key = _normalize_device_key(raw_key, label=f"{label}.<key>")
        normalized[key] = _normalize_device_value(raw_value, label=f"{label}.{key}")
    return normalized


def _normalize_device_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_normalize_device_value(item, label=f"{label}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, Mapping):
        return _normalize_device_mapping(value, label=label)
    raise ValueError(
        f"{label} must contain only YAML scalars, lists, or mappings; got {type(value)!r}"
    )


def _normalize_device_key(value: Any, *, label: str) -> Any:
    if isinstance(value, str):
        return _required_string(value, label=label)
    if isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"{label} must be a YAML scalar key, got {type(value)!r}")


__all__ = [
    "BindingEntry",
    "ReaderDeviceSpec",
    "ResolvedBinding",
    "ResolvedBindingTarget",
    "ReaderMetadataSpec",
    "ReaderBinding",
    "SemanticEntry",
    "apply_binding",
    "canonicalize_semantic_name",
    "ensure_lower_snake_identifier",
    "ensure_semantic_id",
    "ensure_upper_snake_identifier",
    "normalize_aliases",
    "normalize_binding",
    "normalize_reader_device_spec",
    "normalize_reader_metadata",
    "normalize_semantic_registry",
    "resolve_semantic_entry",
    "resolve_binding",
    "resolve_binding_entry_target",
    "validate_aliases_against_semantics",
    "validate_binding_against_semantics",
]
