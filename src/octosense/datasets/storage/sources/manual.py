"""Manual storage source helper."""

from __future__ import annotations

from pathlib import Path


def mark_manual_source(
    dataset_id: str,
    note: str | None = None,
    *,
    source_root: str | Path | None = None,
) -> dict[str, str | None]:
    return {
        "transport": "manual",
        "dataset_id": dataset_id,
        "note": note,
        "source_root": str(source_root) if source_root is not None else None,
    }


__all__ = ["mark_manual_source"]
