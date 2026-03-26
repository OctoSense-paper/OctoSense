"""YAML serialization helpers for benchmark documents."""

from __future__ import annotations

from pathlib import Path

import yaml

from octosense.specs.serde.canonical import (
    SerializableDocument,
    document_from_mapping,
    normalize_document,
)


def load_yaml(payload_or_path: str | Path) -> SerializableDocument:
    payload = _read_text(payload_or_path)
    data = yaml.safe_load(payload)
    if not isinstance(data, dict):
        raise ValueError("Spec YAML payload must decode to a mapping")
    return document_from_mapping(data)


def dump_yaml(document: SerializableDocument) -> str:
    return yaml.safe_dump(normalize_document(document), sort_keys=False)


def _read_text(payload_or_path: str | Path) -> str:
    if isinstance(payload_or_path, Path):
        return payload_or_path.read_text(encoding="utf-8")
    if _looks_like_yaml_payload(payload_or_path):
        return payload_or_path
    candidate = Path(payload_or_path)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return payload_or_path


def _looks_like_yaml_payload(value: str) -> bool:
    return "\n" in value or ":" in value or value.lstrip().startswith("---")


__all__ = ["dump_yaml", "load_yaml"]
