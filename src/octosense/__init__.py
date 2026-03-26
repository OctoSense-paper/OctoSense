"""OctoSense: Deep wireless sensing framework."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as pkg_version


def _resolve_version() -> str:
    try:
        return pkg_version("octosense")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()
_LAZY_MODULES = {
    "benchmarks",
    "core",
    "datasets",
    "io",
    "models",
    "pipelines",
    "specs",
    "tasks",
    "transforms",
}
__all__ = [
    "__version__",
    "benchmarks",
    "core",
    "datasets",
    "io",
    "models",
    "pipelines",
    "specs",
    "tasks",
    "transforms",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
