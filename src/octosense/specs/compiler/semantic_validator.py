"""Semantic validation for benchmark specs and run manifests."""

from __future__ import annotations

import re

from octosense.specs.schemas.benchmark import BENCHMARK_SPEC_KIND, BenchmarkSpec
from octosense.specs.schemas.run_manifest import RUN_MANIFEST_KIND, RunManifest

_READER_ID_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")
_RUN_STATUS = {"running", "completed", "failed"}


def _require_non_blank(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def validate_semantics(spec: BenchmarkSpec) -> BenchmarkSpec:
    """Validate benchmark semantics without reaching into runtime owners."""

    if spec.kind != BENCHMARK_SPEC_KIND:
        raise ValueError(f"BenchmarkSpec.kind must be {BENCHMARK_SPEC_KIND!r}, got {spec.kind!r}")
    _require_non_blank(spec.api_version, field_name="BenchmarkSpec.api_version")

    dataset = spec.dataset
    _require_non_blank(dataset.dataset_id, field_name="BenchmarkSpec.dataset.dataset_id")
    if not dataset.modalities:
        raise ValueError("BenchmarkSpec.dataset.modalities must be a non-empty list")
    if any(not str(item).strip() for item in dataset.modalities):
        raise ValueError("BenchmarkSpec.dataset.modalities cannot contain blank entries")
    if len(set(dataset.modalities)) != len(dataset.modalities):
        raise ValueError("BenchmarkSpec.dataset.modalities must not contain duplicates")

    task = spec.task
    _require_non_blank(task.task_id, field_name="BenchmarkSpec.task.task_id")
    _require_non_blank(task.task_binding, field_name="BenchmarkSpec.task.task_binding")

    transform = spec.transform
    has_preset = bool(str(transform.preset_id or "").strip())
    has_steps = bool(transform.steps)
    if has_preset and has_steps:
        raise ValueError("TransformSpec cannot declare both preset_id and steps")
    if not has_preset and transform.params:
        raise ValueError("TransformSpec.params is only allowed when preset_id is used")
    if has_steps:
        for index, step in enumerate(transform.steps):
            if not str(step.operator_id).strip():
                raise ValueError(f"TransformSpec.steps[{index}].operator_id is required")

    model = spec.model
    _require_non_blank(model.model_id, field_name="BenchmarkSpec.model.model_id")

    runtime = spec.runtime
    if runtime.batch_size <= 0:
        raise ValueError("RuntimeSpec.batch_size must be positive")
    if runtime.epochs <= 0:
        raise ValueError("RuntimeSpec.epochs must be positive")
    if runtime.num_workers < 0:
        raise ValueError("RuntimeSpec.num_workers must be >= 0")
    if runtime.seed < 0:
        raise ValueError("RuntimeSpec.seed must be >= 0")
    _require_non_blank(runtime.device, field_name="RuntimeSpec.device")

    ingestion = spec.ingestion
    if ingestion is not None:
        reader_id = str(ingestion.reader_id).strip()
        if not reader_id:
            raise ValueError("IngestionSpec.reader_id is required when ingestion is provided")
        if not _READER_ID_PATTERN.fullmatch(reader_id):
            raise ValueError(
                "IngestionSpec.reader_id must use the logical '<modality>/<device_family>' form"
            )
        if not str(ingestion.source_root).strip():
            raise ValueError("IngestionSpec.source_root is required when ingestion is provided")
    return spec


def validate_run_manifest(manifest: RunManifest) -> RunManifest:
    """Validate execution-fact manifests."""

    if manifest.kind != RUN_MANIFEST_KIND:
        raise ValueError(f"RunManifest.kind must be {RUN_MANIFEST_KIND!r}, got {manifest.kind!r}")
    _require_non_blank(manifest.run_id, field_name="RunManifest.run_id")
    _require_non_blank(manifest.spec_digest, field_name="RunManifest.spec_digest")
    _require_non_blank(manifest.dataset_digest, field_name="RunManifest.dataset_digest")
    _require_non_blank(manifest.git_sha, field_name="RunManifest.git_sha")
    if manifest.seed is None:
        raise ValueError("RunManifest.seed is required")
    _require_non_blank(manifest.device or "", field_name="RunManifest.device")
    if manifest.status not in _RUN_STATUS:
        raise ValueError(f"RunManifest.status must be one of {sorted(_RUN_STATUS)}")
    return manifest


__all__ = ["validate_run_manifest", "validate_semantics"]
