"""Canonical public reader surface rooted at slash ids and modality namespaces."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

from octosense.io.readers._catalog import canonicalize_reader_id, reader_catalog_by_id


@lru_cache(maxsize=1)
def _catalog():
    return reader_catalog_by_id()


def _resolve_catalog_entry(reader_id: str):
    normalized = canonicalize_reader_id(reader_id)
    entry = _catalog().get(normalized)
    if entry is None:
        known = ", ".join(sorted(_catalog()))
        raise KeyError(
            f"Unknown reader id {reader_id!r}; expected '<modality>/<device_family>' such as "
            f"'wifi/iwl5300'. Known ids: {known}"
        )
    return entry


@lru_cache(maxsize=1)
def _reader_classes_by_modality() -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for entry in _catalog().values():
        grouped.setdefault(entry.descriptor.modality, {})[entry.descriptor.reader_class] = entry.import_path
    return {modality: dict(sorted(classes.items())) for modality, classes in grouped.items()}


def _load_symbol(import_path: str) -> object:
    module_name, attr_name = import_path.split(":")
    return getattr(import_module(module_name), attr_name)


def load(reader_id: str, **kwargs):
    """Instantiate a reader from its canonical ``<modality>/<device_family>`` id."""

    return _load_symbol(_resolve_catalog_entry(reader_id).import_path)(**kwargs)


class _ModalityNamespace:
    def __init__(self, modality: str) -> None:
        self._modality = modality

    def __getattr__(self, name: str) -> object:
        classes = _reader_classes_by_modality().get(self._modality, {})
        import_path = classes.get(name)
        if import_path is None:
            raise AttributeError(f"modality namespace {self._modality!r} has no attribute {name!r}")
        value = _load_symbol(import_path)
        setattr(self, name, value)
        return value

    def __dir__(self) -> list[str]:
        return sorted(_reader_classes_by_modality().get(self._modality, {}))

    def __repr__(self) -> str:
        return f"<octosense.io.readers.{self._modality} namespace>"


@lru_cache(maxsize=None)
def _modality_namespace(modality: str) -> _ModalityNamespace:
    return _ModalityNamespace(modality)


def __getattr__(name: str) -> object:
    if name in _reader_classes_by_modality():
        namespace = _modality_namespace(name)
        globals()[name] = namespace
        return namespace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_reader_classes_by_modality()))


def _public_exports() -> list[str]:
    return ["load", *_reader_classes_by_modality()]


__all__ = _public_exports()
