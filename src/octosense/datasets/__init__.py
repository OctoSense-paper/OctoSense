"""Canonical public surface for ``octosense.datasets``.

The package-level API is intentionally small: ``datasets.load(...)`` and
``datasets.from_tensor(...)`` both materialize a metadata-first ``DatasetView``.
Package-local modules such as ``api.py`` and ``registry.py`` remain internal
implementation details rather than competing public owners. Builtin card/schema
sidecars are parsed only via ``catalog.py``; the package no longer exposes a
parallel descriptor helper plane.
"""

from octosense.datasets.api import from_tensor, load
from octosense.datasets.views.dataset_view import DatasetView

__all__ = [
    "DatasetView",
    "from_tensor",
    "load",
]
