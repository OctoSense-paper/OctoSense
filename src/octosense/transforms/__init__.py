"""Canonical root public surface for transforms."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "compose": (".api", "compose"),
}


def __getattr__(name: str) -> object:
    module_name, attr_name = _LAZY_EXPORTS.get(name, (None, None))
    if module_name is not None:
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name) if attr_name is not None else module
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = ["compose"]
