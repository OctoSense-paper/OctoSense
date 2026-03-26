"""Checksum helpers for dataset storage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChecksumRecord:
    path: str
    algorithm: str
    digest: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "algorithm": self.algorithm,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


def sha256_digest(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_checksum_record(path: str | Path) -> ChecksumRecord:
    resolved = Path(path)
    return ChecksumRecord(
        path=str(resolved),
        algorithm="sha256",
        digest=sha256_digest(resolved),
        size_bytes=resolved.stat().st_size,
    )


def verify_sha256(path: str | Path, expected: str) -> bool:
    digest = sha256_digest(path)
    return digest == expected


__all__ = ["ChecksumRecord", "build_checksum_record", "sha256_digest", "verify_sha256"]
