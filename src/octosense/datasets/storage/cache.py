"""Cache path helpers for dataset storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_CACHE_ROOT = Path.home() / ".cache" / "octosense" / "datasets"


@dataclass(frozen=True)
class DatasetCacheLayout:
    """Resolved storage layout for one dataset cache root."""

    dataset_id: str
    dataset_root: Path
    downloads_root: Path
    cache_root: Path
    receipt_path: Path

    def ensure(self) -> "DatasetCacheLayout":
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.downloads_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self


def resolve_cache_layout(
    dataset_id: str,
    *,
    root: str | Path | None = None,
    dataset_root_override: str | Path | None = None,
    downloads_dirname: str = "downloads",
    cache_dirname: str = "cache",
    receipt_name: str = "octosense.dataset.json",
) -> DatasetCacheLayout:
    if dataset_root_override is not None:
        dataset_root = Path(dataset_root_override)
    else:
        base = Path(root) if root is not None else DEFAULT_DATASET_CACHE_ROOT
        dataset_root = base / dataset_id
    return DatasetCacheLayout(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        downloads_root=dataset_root / downloads_dirname,
        cache_root=dataset_root / cache_dirname,
        receipt_path=dataset_root / receipt_name,
    )


def cache_root(dataset_id: str, root: str | Path | None = None) -> Path:
    return resolve_cache_layout(dataset_id, root=root).dataset_root


__all__ = ["DEFAULT_DATASET_CACHE_ROOT", "DatasetCacheLayout", "cache_root", "resolve_cache_layout"]
