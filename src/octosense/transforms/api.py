"""Canonical package-internal owner for transform composition."""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn

from octosense.transforms.core.compose import Sequential


def compose(transforms: Iterable[nn.Module]) -> Sequential:
    """Build the canonical sequential transform pipeline."""

    return Sequential(list(transforms))


__all__ = ["Sequential", "compose"]
