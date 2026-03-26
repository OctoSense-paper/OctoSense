"""Internal IO-owned semantic registry bootstrap.

This module wires builtin signal-family specs for ``octosense.io`` internals.
It is not the public control-plane surface for reader or semantic selection.
"""

from __future__ import annotations

import types as _types
from collections.abc import Iterator as _Iterator
from contextlib import contextmanager as _contextmanager

from octosense.io.semantics.loader import load_semantic_bundle as _load_semantic_bundle
from octosense.io.semantics.normalizer import (
    normalize_aliases as _normalize_aliases,
    normalize_semantic_registry as _normalize_semantic_registry,
)
from octosense.io.semantics.schema import (
    AxisMetadata as _AxisMetadata,
    AxisSchema as _AxisSchema,
    MetadataSchema as _MetadataSchema,
    build_axis_schema as _build_axis_schema,
)
from octosense.io.semantics.signal_family import (
    DTypePolicy as _DTypePolicy,
    FamilyDiff as _FamilyDiff,
    SignalFamilySpec as _SignalFamilySpec,
)

__all__ = ["get_schema_version"]

_FAMILY_SPECS: dict[str, _SignalFamilySpec] | _types.MappingProxyType = {}
_AXIS_DEPRECATIONS: dict[str, tuple[str, str]] = {}
_AXIS_ALIASES: dict[str, str] = {}
_SCHEMA_VERSION = "1.0.0"
_REGISTRY_FROZEN = False
_REGISTRY_MUTATION_DEPTH = 0

_WIFI_SEMANTIC_BUNDLE = _load_semantic_bundle("wifi")
_WIFI_SEMANTIC_REGISTRY = _normalize_semantic_registry(
    _WIFI_SEMANTIC_BUNDLE["core"],
    _WIFI_SEMANTIC_BUNDLE["modality"],
)
_WIFI_ALIASES = _normalize_aliases(_WIFI_SEMANTIC_BUNDLE["aliases"], modality="wifi")


def _lookup_wifi_semantic_entry(semantic_id: str):
    entry = _WIFI_SEMANTIC_REGISTRY.get(semantic_id)
    if entry is None:
        raise ValueError(f"Unknown WiFi semantic id: {semantic_id!r}")
    return entry


def _lookup_signal_axis_semantics(signal_semantic_id: str) -> tuple[str, ...]:
    signal_entry = _lookup_wifi_semantic_entry(signal_semantic_id)
    axis_semantics = tuple(signal_entry.axis_semantics)
    if not axis_semantics:
        raise ValueError(f"Signal semantic {signal_semantic_id!r} does not declare axis_semantics")
    return axis_semantics


def _axis_identity(schema: _AxisSchema, axis_name: str) -> str:
    metadata = schema.axis_metadata.get(axis_name)
    semantic_id = metadata.semantic_id if metadata is not None else None
    return semantic_id or axis_name


def _axis_identity_sequence(schema: _AxisSchema) -> tuple[str, ...]:
    return tuple(_axis_identity(schema, axis_name) for axis_name in schema.axes)


def _axis_names_by_identity(schema: _AxisSchema) -> dict[str, str]:
    return {_axis_identity(schema, axis_name): axis_name for axis_name in schema.axes}


def _collect_axes_missing_from(
    source: _AxisSchema,
    target: _AxisSchema,
) -> list[str]:
    target_identities = set(_axis_identity_sequence(target))
    missing: list[str] = []
    seen: set[str] = set()
    for axis_name in source.axes:
        identity = _axis_identity(source, axis_name)
        if identity in target_identities or identity in seen:
            continue
        missing.append(axis_name)
        seen.add(identity)
    return missing


def _collect_axis_export_renames(
    a: _AxisSchema,
    b: _AxisSchema,
) -> list[tuple[str, str]]:
    names_a = _axis_names_by_identity(a)
    names_b = _axis_names_by_identity(b)
    renames: list[tuple[str, str]] = []
    seen: set[str] = set()
    for identity in _axis_identity_sequence(a):
        if identity in seen or identity not in names_b:
            continue
        seen.add(identity)
        old_name = names_a[identity]
        new_name = names_b[identity]
        if old_name != new_name:
            renames.append((old_name, new_name))
    return renames


def is_registry_frozen() -> bool:
    return _REGISTRY_FROZEN


def freeze_registry() -> None:
    global _REGISTRY_FROZEN, _FAMILY_SPECS
    _REGISTRY_FROZEN = True
    _FAMILY_SPECS = _types.MappingProxyType(dict(_FAMILY_SPECS))


@_contextmanager
def mutation_session() -> _Iterator[None]:
    global _REGISTRY_FROZEN, _REGISTRY_MUTATION_DEPTH, _FAMILY_SPECS
    _REGISTRY_MUTATION_DEPTH += 1
    _REGISTRY_FROZEN = False
    if isinstance(_FAMILY_SPECS, _types.MappingProxyType):
        _FAMILY_SPECS = dict(_FAMILY_SPECS)
    try:
        yield
    finally:
        _REGISTRY_MUTATION_DEPTH -= 1
        if _REGISTRY_MUTATION_DEPTH <= 0:
            _REGISTRY_MUTATION_DEPTH = 0
            _REGISTRY_FROZEN = True
            _FAMILY_SPECS = _types.MappingProxyType(dict(_FAMILY_SPECS))


def register_family(spec: _SignalFamilySpec) -> None:
    if _REGISTRY_FROZEN:
        raise RuntimeError(f"Registry is frozen; cannot register family '{spec.name}'.")
    if spec.name in _FAMILY_SPECS:
        raise ValueError(f"SignalFamilySpec '{spec.name}' already registered")
    _FAMILY_SPECS[spec.name] = spec


def get_family(name: str) -> _SignalFamilySpec:
    if name not in _FAMILY_SPECS:
        raise ValueError(
            f"SignalFamilySpec '{name}' not found. Available families: {list_families()}"
        )
    return _FAMILY_SPECS[name]


def list_families() -> list[str]:
    return list(_FAMILY_SPECS.keys())


def compare_family(spec_a: _SignalFamilySpec, spec_b: _SignalFamilySpec) -> _FamilyDiff:
    schema_diff = compare_schema(spec_a.axis_schema, spec_b.axis_schema)
    added_axes = list(schema_diff["added_axes"])
    removed_axes = list(schema_diff["removed_axes"])

    changed_metadata: dict[str, tuple[object, object]] = {}
    for layer in ["required_physical", "required_coords", "required_provenance"]:
        old_fields = getattr(spec_a.metadata_schema, layer, [])
        new_fields = getattr(spec_b.metadata_schema, layer, [])
        if old_fields != new_fields:
            changed_metadata[layer] = (old_fields, new_fields)
    renamed_axes = list(schema_diff.get("renamed_axes", []))
    if renamed_axes:
        changed_metadata["axis_export_names"] = (
            list(spec_a.axis_schema.axes),
            list(spec_b.axis_schema.axes),
        )

    version_change = None
    if spec_a.version != spec_b.version:
        version_change = (spec_a.version, spec_b.version)

    contract_version_change = None
    if spec_a.contract_version != spec_b.contract_version:
        contract_version_change = (spec_a.contract_version, spec_b.contract_version)

    identities_a = _axis_identity_sequence(spec_a.axis_schema)
    identities_b = _axis_identity_sequence(spec_b.axis_schema)
    reordered_axes: list[str] = []
    if set(identities_a) == set(identities_b) and identities_a != identities_b:
        names_b = _axis_names_by_identity(spec_b.axis_schema)
        seen: set[str] = set()
        for idx, identity in enumerate(identities_a):
            if identity in seen:
                continue
            seen.add(identity)
            if identities_b.index(identity) != idx:
                reordered_axes.append(names_b.get(identity, spec_a.axis_schema.axes[idx]))

    return _FamilyDiff(
        added_axes=added_axes,
        removed_axes=removed_axes,
        reordered_axes=reordered_axes,
        changed_metadata=changed_metadata,
        version_change=version_change,
        contract_version_change=contract_version_change,
        breaking_change=bool(removed_axes or contract_version_change or reordered_axes),
    )


def compare_schema(a: _AxisSchema, b: _AxisSchema) -> dict[str, object]:
    identities_a = _axis_identity_sequence(a)
    identities_b = _axis_identity_sequence(b)
    return {
        "added_axes": _collect_axes_missing_from(b, a),
        "removed_axes": _collect_axes_missing_from(a, b),
        "reordered": identities_a != identities_b and set(identities_a) == set(identities_b),
        "renamed_axes": _collect_axis_export_renames(a, b),
    }


def deprecate_axis(name: str, replacement: str, version: str) -> None:
    if _REGISTRY_FROZEN:
        raise RuntimeError(f"Registry is frozen; cannot deprecate axis '{name}'.")
    _AXIS_DEPRECATIONS[name] = (replacement, version)


def get_deprecation(name: str) -> tuple[str, str] | None:
    return _AXIS_DEPRECATIONS.get(name)


def alias_axis(canonical: str, aliases: list[str], *, allow_override: bool = False) -> None:
    if _REGISTRY_FROZEN:
        raise RuntimeError(f"Registry is frozen; cannot register aliases for '{canonical}'.")
    for alias in aliases:
        existing = _AXIS_ALIASES.get(alias)
        if existing is not None and existing != canonical and not allow_override:
            raise ValueError(
                f"Alias conflict: '{alias}' already points to '{existing}', cannot reassign to "
                f"'{canonical}'. Use allow_override=True to force."
            )
        _AXIS_ALIASES[alias] = canonical


def get_canonical_axis(name: str) -> str:
    return _AXIS_ALIASES.get(name, name)


def get_schema_version() -> str:
    return _SCHEMA_VERSION


_WIFI_CSI_SPEC = _SignalFamilySpec(
    name="WiFi_CSI",
    version="1.0.0",
    axis_schema=_build_axis_schema(
        _lookup_signal_axis_semantics("octo.wifi.tensor.csi"),
        semantic_registry=_WIFI_SEMANTIC_REGISTRY,
        aliases=_WIFI_ALIASES,
    ),
    metadata_schema=_MetadataSchema(
        required_physical=["center_freq", "bandwidth"],
        optional_physical=["sample_rate", "subcarrier_spacing"],
        required_coords=["timestamp", "subc"],
        required_provenance=["reader_id", "capture_device"],
    ),
    dtype_policy=_DTypePolicy("complex64", "complex-first", "Canonical WiFi CSI"),
    canonicalization={},
    contract_version="1.0",
    description="Canonical WiFi CSI carrier shared across supported WiFi readers",
)

_WIFI_CSI_RAW_FAMILIES = ("WiFi_CSI",)

_WIFI_SPECTROGRAM_SPEC = _SignalFamilySpec(
    name="WiFi_Spectrogram",
    version="1.0.0",
    axis_schema=_AxisSchema(
        axes=("subc", "tx", "rx", "freq", "frame"),
        axis_metadata={
            "subc": _AxisMetadata(
                "subc",
                None,
                "Subcarriers",
                semantic_id="octo.wifi.axis.subc",
                axis_role="spectral",
            ),
            "tx": _AxisMetadata(
                "tx",
                None,
                "Transmit antennas",
                semantic_id="octo.common.axis.tx",
                axis_role="spatial",
            ),
            "rx": _AxisMetadata(
                "rx",
                None,
                "Receive antennas",
                semantic_id="octo.common.axis.rx",
                axis_role="spatial",
            ),
            "freq": _AxisMetadata("freq", "Hz", "Frequency bins", axis_role="layout"),
            "frame": _AxisMetadata(
                "frame",
                "s",
                "Time frames",
                semantic_id="octo.common.axis.frame",
                axis_role="layout",
            ),
        },
    ),
    metadata_schema=_MetadataSchema(
        required_physical=["center_freq", "bandwidth"],
        optional_physical=["sample_rate"],
        required_coords=["freq", "frame"],
        required_provenance=["reader_id", "capture_device"],
    ),
    dtype_policy=_DTypePolicy(
        "complex64",
        "complex-first",
        "Time-frequency WiFi spectrogram before model-ready projection",
    ),
    canonicalization={"freq": {"unit": "Hz"}, "frame": {"unit": "s"}},
    contract_version="1.0",
    description="Canonical WiFi CSI spectrogram representation",
)

_WIFI_IMAGE_SPEC = _SignalFamilySpec(
    name="WiFi_Image",
    version="1.0.0",
    axis_schema=_AxisSchema(
        axes=("channel", "freq", "frame"),
        axis_metadata={
            "channel": _AxisMetadata(
                "channel",
                "index",
                "Merged RF channel axis",
                axis_role="feature",
            ),
            "freq": _AxisMetadata("freq", "Hz", "Frequency bins", axis_role="layout"),
            "frame": _AxisMetadata(
                "frame",
                "s",
                "Time frames",
                semantic_id="octo.common.axis.frame",
                axis_role="layout",
            ),
        },
    ),
    metadata_schema=_MetadataSchema(
        required_physical=["center_freq", "bandwidth"],
        optional_physical=["sample_rate"],
        required_coords=["freq", "frame"],
        required_provenance=["reader_id", "capture_device"],
    ),
    dtype_policy=_DTypePolicy(
        "float32",
        "real-first",
        "Image-style WiFi representation consumed by CNN backbones",
    ),
    canonicalization={"freq": {"unit": "Hz"}, "frame": {"unit": "s"}},
    contract_version="1.0",
    description="Model-ready WiFi image representation",
)

_WIFI_SEQUENCE_SPEC = _SignalFamilySpec(
    name="WiFi_Sequence",
    version="1.0.0",
    axis_schema=_AxisSchema(
        axes=("time", "feature"),
        axis_metadata={
            "time": _AxisMetadata(
                "time",
                "s",
                "Temporal steps",
                semantic_id="octo.common.axis.time",
                axis_role="temporal",
            ),
            "feature": _AxisMetadata(
                "feature",
                "index",
                "Flattened RF feature axis",
                axis_role="feature",
            ),
        },
    ),
    metadata_schema=_MetadataSchema(
        required_physical=["center_freq", "bandwidth"],
        optional_physical=["sample_rate"],
        required_coords=["timestamp"],
        required_provenance=["reader_id", "capture_device"],
    ),
    dtype_policy=_DTypePolicy(
        "float32",
        "real-first",
        "Sequence-style WiFi representation for recurrent backbones",
    ),
    canonicalization={"time": {"unit": "s"}},
    contract_version="1.0",
    description="Model-ready WiFi sequence representation",
)

_WIFI_SELECTED_SUBCARRIER_SPEC = _SignalFamilySpec(
    name="WiFiCSI_SelectedSubcarrier",
    version="1.0.0",
    axis_schema=_AxisSchema(
        axes=("time", "sensor"),
        axis_metadata={
            "time": _AxisMetadata(
                "time",
                "s",
                "Temporal samples",
                semantic_id="octo.common.axis.time",
                axis_role="temporal",
            ),
            "sensor": _AxisMetadata(
                "sensor",
                "index",
                "Flattened spatial channel",
                axis_role="feature",
            ),
        },
    ),
    metadata_schema=_MetadataSchema(
        required_physical=["center_freq", "bandwidth"],
        optional_physical=["sample_rate"],
        required_coords=["timestamp", "sensor"],
        required_provenance=["reader_id", "capture_device"],
    ),
    dtype_policy=_DTypePolicy(
        "complex64",
        "complex-first",
        "Per-sensor motion-selected CSI time series",
    ),
    canonicalization={"time": {"unit": "s"}},
    contract_version="1.0",
    description="Motion-selected WiFi CSI subcarriers for Doppler-oriented pipelines",
)

_BUILTIN_FAMILY_SPECS = (
    _WIFI_CSI_SPEC,
    _WIFI_SPECTROGRAM_SPEC,
    _WIFI_IMAGE_SPEC,
    _WIFI_SEQUENCE_SPEC,
    _WIFI_SELECTED_SUBCARRIER_SPEC,
)

for _family_spec in _BUILTIN_FAMILY_SPECS:
    register_family(_family_spec)

freeze_registry()
