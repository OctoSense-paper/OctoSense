"""Human-readable report helpers for benchmark artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from octosense.benchmarks.artifacts.run_manifest import run_manifest_to_dict
from octosense.specs.schemas.run_manifest import RunManifest


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: str | Path, payload: str) -> Path:
    artifact_path = Path(path)
    _ensure_parent_dir(artifact_path)
    artifact_path.write_text(payload, encoding="utf-8")
    return artifact_path


def render_markdown_report(
    run_manifest: RunManifest,
    metrics: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> str:
    """Render a compact, auditable benchmark report."""
    lines = ["# Benchmark Report", ""]

    lines.append("## Run Manifest")
    for key, value in run_manifest_to_dict(run_manifest).items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    if protocol:
        lines.append("## Protocol")
        for key, value in protocol.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## Metrics")
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    if provenance:
        lines.append("## Provenance")
        for key, value in provenance.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    if environment:
        lines.append("## Environment")
        for key, value in environment.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(path: str | Path, content: str) -> Path:
    """Persist a markdown benchmark report."""
    return _write_text(path, content)


__all__ = ["render_markdown_report", "write_markdown_report"]
