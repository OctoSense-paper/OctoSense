"""Load canonical IO semantic, device, binding, and metadata sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from octosense.io.readers._base import ReaderError
from octosense.io.readers._catalog import reader_catalog_by_id
from octosense.io.semantics.normalizer import (
    ReaderBinding,
    ReaderDeviceSpec,
    ReaderMetadataSpec,
    ResolvedBinding,
    SemanticEntry,
    normalize_aliases,
    normalize_binding,
    normalize_reader_device_spec,
    normalize_reader_metadata,
    normalize_semantic_registry,
    resolve_binding,
    validate_aliases_against_semantics,
    validate_binding_against_semantics,
)


@dataclass(frozen=True)
class ReaderDefinitionBundle:
    modality: str
    device: str
    reader_root: Path
    semantics: dict[str, Any]
    semantic_registry: dict[str, SemanticEntry]
    aliases: dict[str, str]
    device_spec: ReaderDeviceSpec
    device_config: dict[str, Any]
    binding: ReaderBinding
    binding_plan: ResolvedBinding
    binding_converters: dict[str, Callable[..., Any]]
    converter_context: dict[str, Any]
    metadata: dict[str, Any]
    metadata_spec: ReaderMetadataSpec
    bibtex: str

    @property
    def config(self) -> dict[str, Any]:
        return self.device_config

    @property
    def reader_id(self) -> str:
        return self.metadata_spec.reader_id

    @property
    def canonical_names(self) -> set[str]:
        return {entry.preferred_name for entry in self.semantic_registry.values()}

    @property
    def canonical_export_names(self) -> set[str]:
        return {entry.preferred_name for entry in self.semantic_registry.values()}

    @property
    def canonical_semantic_ids(self) -> set[str]:
        return set(self.semantic_registry)

    @property
    def canonical_axes(self) -> tuple[str, ...]:
        return tuple(
            entry.preferred_name
            for entry in self.semantic_registry.values()
            if entry.kind == "axis"
        )


def io_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(payload)!r}")
    return payload


def load_optional_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_yaml_mapping(path)


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_semantic_bundle(modality: str) -> dict[str, Any]:
    root = io_root() / "semantics"
    bundle = {
        "core": load_yaml_mapping(root / "core.yaml"),
        "modality": load_yaml_mapping(root / f"{modality}.yaml"),
        "aliases": load_optional_yaml_mapping(root / "aliases.yaml"),
    }
    semantic_registry = normalize_semantic_registry(bundle["core"], bundle["modality"])
    aliases = normalize_aliases(
        bundle["aliases"],
        modality=modality,
        semantic_registry=semantic_registry,
    )
    validate_aliases_against_semantics(aliases, semantic_registry=semantic_registry)
    bundle["semantic_registry"] = semantic_registry
    bundle["resolved_aliases"] = aliases
    return bundle


def load_reader_definition_bundle(modality: str, device: str) -> ReaderDefinitionBundle:
    root = io_root()
    reader_root = root / "readers" / modality / device
    if not reader_root.exists():
        raise ReaderError(f"Unknown reader root: {reader_root}")
    semantic_bundle = load_semantic_bundle(modality)
    semantic_registry = semantic_bundle["semantic_registry"]
    aliases = semantic_bundle["resolved_aliases"]
    device_spec = normalize_reader_device_spec(
        load_yaml_mapping(reader_root / "device.yaml"),
        modality=modality,
        device_family=device,
        device_identity=reader_root.name,
    )
    binding = normalize_binding(load_yaml_mapping(reader_root / "binding.yaml"))
    validate_binding_against_semantics(
        binding,
        semantic_registry=semantic_registry,
        aliases=aliases,
    )
    binding_plan = resolve_binding(
        binding,
        semantic_registry=semantic_registry,
        aliases=aliases,
    )
    binding_converters = load_binding_converters(reader_root)
    _validate_binding_converters(binding, binding_converters, reader_root=reader_root)
    metadata = load_optional_yaml_mapping(reader_root / "metadata.yaml")
    metadata_spec = normalize_reader_metadata(metadata, modality=modality, device_family=device)
    catalog_entry = reader_catalog_by_id().get(metadata_spec.reader_id)
    if catalog_entry is not None and catalog_entry.metadata.get("reader_class") is not None:
        expected_class = str(catalog_entry.metadata["reader_class"])
        if metadata_spec.reader_class is not None and metadata_spec.reader_class != expected_class:
            raise ReaderError(
                "Reader metadata class does not match reader catalog: "
                f"{metadata_spec.reader_class!r} != {expected_class!r}"
            )
    return ReaderDefinitionBundle(
        modality=modality,
        device=device,
        reader_root=reader_root,
        semantics=semantic_bundle,
        semantic_registry=semantic_registry,
        aliases=aliases,
        device_spec=device_spec,
        device_config=device_spec.as_mapping(),
        binding=binding,
        binding_plan=binding_plan,
        binding_converters=binding_converters,
        converter_context=device_spec.as_mapping(),
        metadata=metadata,
        metadata_spec=metadata_spec,
        bibtex=read_text_file(reader_root / "bibtex.bib"),
    )


def load_binding_converters(reader_root: Path) -> dict[str, Callable[..., Any]]:
    # Reader-root converter.py modules are intentionally no longer imported.
    # Declarative `scale` handles linear normalization, and any remaining
    # device-specific value cleanup must happen inside the reader itself.
    return {}


def _validate_binding_converters(
    binding: ReaderBinding,
    converters: dict[str, Callable[..., Any]],
    *,
    reader_root: Path,
) -> None:
    missing_via_names = sorted(
        {
            entry.via
            for section in (binding.axes.values(), binding.fields.values())
            for entry in section
            if entry.via is not None and entry.via not in converters
        }
    )
    if missing_via_names:
        raise ReaderError(
            "Reader bundle "
            f"{reader_root} references unsupported converter(s): {missing_via_names}. "
            "Reader-root converter.py loading has been removed; use declarative scale "
            "or normalize the value inside reader.py before applying the binding."
        )


__all__ = [
    "ReaderDefinitionBundle",
    "ReaderError",
    "io_root",
    "load_optional_yaml_mapping",
    "load_binding_converters",
    "load_reader_definition_bundle",
    "load_semantic_bundle",
    "load_yaml_mapping",
    "read_text_file",
]
