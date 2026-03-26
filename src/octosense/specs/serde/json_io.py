"""JSON serialization helpers for benchmark documents."""

from __future__ import annotations

import json
from pathlib import Path

from octosense.specs.serde.canonical import (
    SerializableDocument,
    document_from_mapping,
    normalize_document,
)


def dump_json(document: SerializableDocument) -> str:
    return json.dumps(normalize_document(document), indent=2, sort_keys=True) + "\n"


def load_json(payload_or_path: str | Path) -> SerializableDocument:
    payload = _read_text(payload_or_path)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Spec JSON payload must decode to an object")
    return document_from_mapping(data)


def _read_text(payload_or_path: str | Path) -> str:
    if isinstance(payload_or_path, Path):
        return payload_or_path.read_text(encoding="utf-8")
    if _looks_like_json_payload(payload_or_path):
        return payload_or_path
    candidate = Path(payload_or_path)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return payload_or_path


def _looks_like_json_payload(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


__all__ = ["dump_json", "load_json"]
