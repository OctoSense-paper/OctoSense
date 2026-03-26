"""Canonical RunManifest materialization and persistence helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from octosense.specs.compiler.semantic_validator import validate_run_manifest
from octosense.specs.schemas.run_manifest import RUN_MANIFEST_KIND, RunManifest


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, payload: Any) -> Path:
    artifact_path = Path(path)
    _ensure_parent_dir(artifact_path)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _optional_str(value: str | Path | object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def build_run_manifest(
    *,
    run_id: str,
    spec_digest: str,
    dataset_digest: str,
    git_sha: str,
    seed: int,
    device: str,
    status: str = "completed",
    started_at: str | None = None,
    finished_at: str | None = None,
    artifact_root: str | Path | None = None,
    metrics_path: str | Path | None = None,
    environment_path: str | Path | None = None,
    timing_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
) -> RunManifest:
    """Materialize and validate the canonical RunManifest artifact."""

    return validate_run_manifest(
        RunManifest(
            kind=RUN_MANIFEST_KIND,
            run_id=str(run_id),
            spec_digest=str(spec_digest),
            dataset_digest=str(dataset_digest),
            git_sha=str(git_sha),
            seed=_optional_int(seed),
            device=str(device),
            status=str(status),
            started_at=_optional_str(started_at),
            finished_at=_optional_str(finished_at),
            artifact_root=_optional_str(artifact_root),
            metrics_path=_optional_str(metrics_path),
            environment_path=_optional_str(environment_path),
            timing_path=_optional_str(timing_path),
            protocol_path=_optional_str(protocol_path),
        )
    )


def run_manifest_from_dict(payload: Mapping[str, Any] | None) -> RunManifest:
    """Load and validate a RunManifest from a JSON payload."""

    manifest = RunManifest.from_dict(dict(payload or {}))
    return validate_run_manifest(manifest)


def run_manifest_to_dict(run_manifest: RunManifest) -> dict[str, Any]:
    """Convert a RunManifest to a JSON-serializable mapping."""

    return run_manifest.to_dict()


def write_run_manifest(path: str | Path, run_manifest: RunManifest) -> Path:
    """Write a validated RunManifest JSON artifact."""

    return _write_json(path, run_manifest_to_dict(validate_run_manifest(run_manifest)))


def read_run_manifest(path: str | Path, *, validate: bool = True) -> RunManifest:
    """Read a RunManifest JSON artifact."""

    payload = dict(_read_json(path))
    manifest = RunManifest.from_dict(payload)
    if validate:
        return validate_run_manifest(manifest)
    return manifest


__all__ = [
    "build_run_manifest",
    "read_run_manifest",
    "run_manifest_from_dict",
    "run_manifest_to_dict",
    "write_run_manifest",
]
