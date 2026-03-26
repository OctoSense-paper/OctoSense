"""Canonical document normalization and digest helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from octosense.specs.schemas.benchmark import BENCHMARK_SPEC_KIND, BenchmarkSpec
from octosense.specs.schemas.run_manifest import RUN_MANIFEST_KIND, RunManifest

SerializableDocument = BenchmarkSpec | RunManifest


def document_from_mapping(payload: Mapping[str, Any]) -> SerializableDocument:
    kind = str(payload.get("kind", "") or "")
    data = dict(payload)
    if kind == RUN_MANIFEST_KIND or {"run_id", "spec_digest", "dataset_digest"} <= data.keys():
        return RunManifest.from_dict(data)
    if kind == BENCHMARK_SPEC_KIND or {"dataset", "task", "transform", "model", "runtime"} <= data.keys():
        return BenchmarkSpec.from_dict(data)
    raise ValueError(
        "Unable to resolve serialization document kind. "
        "Expected BenchmarkSpec or RunManifest payload."
    )


def normalize_document(document: SerializableDocument) -> dict[str, Any]:
    return _normalize_value(document.to_dict())


def canonical_dump(document: SerializableDocument) -> str:
    return json.dumps(normalize_document(document), sort_keys=True, separators=(",", ":"))


def document_digest(document: SerializableDocument) -> str:
    return hashlib.sha256(canonical_dump(document).encode("utf-8")).hexdigest()


def spec_digest(spec: BenchmarkSpec) -> str:
    return document_digest(spec)


def manifest_digest(manifest: RunManifest) -> str:
    return document_digest(manifest)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            normalized[key] = _normalize_value(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


__all__ = [
    "SerializableDocument",
    "canonical_dump",
    "document_digest",
    "document_from_mapping",
    "manifest_digest",
    "normalize_document",
    "spec_digest",
]
