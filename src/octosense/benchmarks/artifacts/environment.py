"""Environment artifact helpers for benchmark runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from octosense._internal.env import collect_runtime_env


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


def build_environment_artifact(environment: Mapping[str, Any]) -> dict[str, str]:
    """Normalize environment values into string form for portable artifacts."""
    return {str(key): str(value) for key, value in environment.items()}


def collect_environment(*, extra: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Capture a compact environment payload for benchmark artifacts."""
    return collect_runtime_env(extra=dict(extra) if extra else None)


def write_environment_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Persist an environment artifact to JSON."""
    return _write_json(path, build_environment_artifact(payload))


def read_environment_artifact(path: str | Path) -> dict[str, str]:
    """Load an environment artifact from JSON."""
    payload = dict(_read_json(path))
    return {str(key): str(value) for key, value in payload.items()}


__all__ = [
    "build_environment_artifact",
    "collect_environment",
    "read_environment_artifact",
    "write_environment_artifact",
]
