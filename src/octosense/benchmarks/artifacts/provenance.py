"""Provenance record helpers for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def build_provenance_artifact(
    *,
    spec_digest: str,
    dataset_digest: str,
    seed: int | None = None,
    git_sha: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact provenance payload without pulling in execution state."""
    payload: dict[str, Any] = {
        "spec_digest": str(spec_digest),
        "dataset_digest": str(dataset_digest),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if git_sha is not None:
        payload["git_sha"] = str(git_sha)
    if extras:
        payload["extras"] = dict(extras)
    return payload


def write_provenance_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Persist a provenance artifact to JSON."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a stable SHA-256 fingerprint for a contiguous CPU view of a tensor."""

    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def resolve_git_sha(root: str | Path) -> str:
    """Best-effort git commit resolution for provenance-bearing artifacts."""

    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def read_provenance_artifact(path: str | Path) -> dict[str, Any]:
    """Load a provenance artifact from JSON."""

    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "build_provenance_artifact",
    "read_provenance_artifact",
    "resolve_git_sha",
    "tensor_sha256",
    "write_provenance_artifact",
]
