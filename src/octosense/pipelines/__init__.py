"""Canonical public surface for ``octosense.pipelines``.

Users should enter this package through ``pipelines.load(...)`` or
``pipelines.infer(...)``. Internal assembly helpers under ``builder.py`` and
``dataloading/`` are not part of the top-level public API.
"""

from octosense.pipelines.api import infer, load

__all__ = ["infer", "load"]
