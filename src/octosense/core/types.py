"""Semantic type aliases shared across modules."""

from __future__ import annotations

from typing import TypeAlias

import torch

ComplexTensor: TypeAlias = torch.Tensor
RealTensor: TypeAlias = torch.Tensor
TensorLike: TypeAlias = torch.Tensor
DeviceLike: TypeAlias = str | torch.device
Device: TypeAlias = DeviceLike

__all__ = ["ComplexTensor", "Device", "DeviceLike", "RealTensor", "TensorLike"]
