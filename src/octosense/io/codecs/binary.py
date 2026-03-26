"""Generic binary readers shared by RF hardware readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_binary(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def read_int16_array(path: str | Path, *, mmap_threshold_bytes: int | None = None) -> np.ndarray:
    resolved = Path(path)
    size = resolved.stat().st_size
    if mmap_threshold_bytes is not None and size > mmap_threshold_bytes:
        return np.memmap(resolved, dtype=np.int16, mode="r")
    return np.fromfile(resolved, dtype=np.int16)


__all__ = ["read_binary", "read_int16_array"]
