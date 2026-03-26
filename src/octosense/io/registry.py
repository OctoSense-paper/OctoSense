"""Internal reader discovery helpers keyed by canonical ``<modality>/<device_family>`` ids."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

from octosense.io.readers._catalog import canonicalize_reader_id, reader_catalog_by_id


@lru_cache(maxsize=1)
def _reader_import_paths() -> dict[str, str]:
    return {
        reader_id: entry.import_path
        for reader_id, entry in sorted(reader_catalog_by_id().items())
    }

def _normalize_reader_id(reader_id: str) -> str:
    normalized = canonicalize_reader_id(reader_id)
    if normalized not in _reader_import_paths():
        known = ", ".join(sorted(_reader_import_paths()))
        raise KeyError(
            f"Unknown reader id {reader_id!r}; expected '<modality>/<device_family>' such as "
            f"'wifi/iwl5300'. Known ids: {known}"
        )
    return normalized


def _resolve_reader_import_path(reader_id: str) -> str:
    return _reader_import_paths()[_normalize_reader_id(reader_id)]


def _load_reader_class(reader_id: str) -> type:
    module_name, attr_name = _resolve_reader_import_path(reader_id).split(":")
    return getattr(import_module(module_name), attr_name)


def _create_reader(reader_id: str, **kwargs):
    return _load_reader_class(reader_id)(**kwargs)


__all__: list[str] = []
