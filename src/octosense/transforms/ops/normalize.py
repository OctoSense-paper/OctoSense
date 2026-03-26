"""Normalization transforms."""

import time
from typing import Literal

import torch
import torch.nn.functional as F

from octosense.core.contracts import AxisContract
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


def _reduction_size(
    x: torch.Tensor,
    dim: int | tuple[int, ...] | None,
) -> int:
    if dim is None:
        return int(x.numel())
    if isinstance(dim, int):
        return int(x.shape[dim])
    size = 1
    for axis in dim:
        size *= int(x.shape[axis])
    return size


def _complex_std_mean(
    x: torch.Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
):
    if dim is None:
        mean = x.mean()
        centered = x - mean
        squared = centered.real.square() + centered.imag.square()
        count = _reduction_size(x, dim)
        if count <= 1:
            std = squared.new_zeros(())
        else:
            std = (squared.sum() / float(count - 1)).sqrt()
        return std, mean

    mean = x.mean(dim=dim, keepdim=True)
    centered = x - mean
    squared = centered.real.square() + centered.imag.square()
    count = _reduction_size(x, dim)
    if count <= 1:
        std = torch.zeros_like(squared.sum(dim=dim, keepdim=True))
    else:
        std = (squared.sum(dim=dim, keepdim=True) / float(count - 1)).sqrt()
    return std, mean


def _minmax_normalize(
    x,
    *,
    range_min: float = 0.0,
    range_max: float = 1.0,
    dim: int | tuple[int, ...] | None = None,
    epsilon: float = 1e-8,
):
    if dim is None:
        x_min, x_max = torch.aminmax(x)
    elif isinstance(dim, int):
        x_min, x_max = torch.aminmax(x, dim=dim, keepdim=True)
    else:
        x_min = x.amin(dim=dim, keepdim=True)
        x_max = x.amax(dim=dim, keepdim=True)
    normalized = (x - x_min) / (x_max - x_min + epsilon)
    return normalized * (range_max - range_min) + range_min


def _zscore_normalize(
    x,
    *,
    dim: int | tuple[int, ...] | None = None,
    epsilon: float = 1e-8,
):
    if torch.is_complex(x):
        std, mean = _complex_std_mean(x, dim=dim)
    elif dim is None:
        std, mean = torch.std_mean(x)
    else:
        std, mean = torch.std_mean(x, dim=dim, keepdim=True)
    return (x - mean) / std.clamp_min(epsilon)


def _l2_normalize(
    x,
    *,
    dim: int | tuple[int, ...] | None = None,
    epsilon: float = 1e-8,
):
    if dim is None:
        flat = x.reshape(1, -1)
        return F.normalize(flat, p=2.0, dim=1, eps=epsilon).reshape_as(x)
    if isinstance(dim, int):
        return F.normalize(x, p=2.0, dim=dim, eps=epsilon)
    denom = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True).clamp_min(epsilon)
    return x / denom


@registered_operator(required_axes=[], description="Min-max or z-score normalization.")
class Normalize(BaseTransform):
    """Normalize RadioTensor amplitude to specified range.

    Supports per-sample or global normalization.
    """

    def __init__(
        self,
        method: Literal["minmax", "zscore", "l2"] = "minmax",
        range: tuple[float, float] = (0.0, 1.0),
        per_sample: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize Normalize transform.

        Args:
            method: Normalization method ("minmax", "zscore", or "l2")
            range: Target range for minmax (default: [0, 1])
            per_sample: Normalize each sample independently (default: True)
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.method = method
        self.range = range
        self.per_sample = per_sample
        self.epsilon = epsilon

    @property
    def input_contract(self) -> AxisContract:
        """Works with any dtype and axes."""
        return AxisContract(required_axes=[])

    @property
    def output_contract(self) -> AxisContract:
        """Output has same axes as input."""
        return AxisContract()

    def forward(self, x: RadioTensor) -> RadioTensor:
        """Normalize RadioTensor.

        Args:
            x: Input RadioTensor

        Returns:
            Normalized RadioTensor with same shape/schema
        """
        data = x.as_tensor()

        # Determine normalization dimensions
        if self.per_sample:
            # Normalize per sample (all dims except batch)
            if "batch" in x.axis_schema.axes:
                batch_idx = x.get_axis_index("batch")
                # Normalize along all dims except batch
                dims = tuple(i for i in range(data.ndim) if i != batch_idx)
            else:
                # No batch dim, normalize globally
                dims = None
        else:
            # Global normalization
            dims = None

        # Apply normalization
        if self.method == "minmax":
            normalized = _minmax_normalize(
                data,
                range_min=self.range[0],
                range_max=self.range[1],
                dim=dims,
                epsilon=self.epsilon,
            )
        elif self.method == "zscore":
            normalized = _zscore_normalize(
                data,
                dim=dims,
                epsilon=self.epsilon,
            )
        elif self.method == "l2":
            normalized = _l2_normalize(
                data,
                dim=dims,
                epsilon=self.epsilon,
            )
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Update metadata (skip when not tracking)
        if is_tracking_meta():
            new_metadata = x.metadata.copy()
            new_metadata.transforms.append(
                TransformRecord(
                    name="Normalize",
                    params={
                        "method": self.method,
                        "range": self.range,
                        "per_sample": self.per_sample,
                    },
                    timestamp=time.time(),
                )
            )
        else:
            new_metadata = x.metadata

        return RadioTensor(
            data=normalized,
            axis_schema=x.axis_schema,
            metadata=new_metadata,
        )


@registered_operator(required_axes=[], description="Z-score each sample while keeping excluded axes intact.")
class SampleZScoreNormalize(BaseTransform):
    """Normalize each sample over all non-excluded axes."""

    def __init__(
        self,
        *,
        exclude_axes: tuple[str, ...] = ("component",),
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.exclude_axes = tuple(exclude_axes)
        self.epsilon = float(epsilon)

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract()

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract()

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        tensor = x.as_tensor().float()
        excluded = set(self.exclude_axes)
        dims = tuple(
            idx for idx, axis_name in enumerate(x.axis_schema.axes) if axis_name not in excluded
        )
        if not dims:
            raise ValueError(
                "SampleZScoreNormalize requires at least one normalization axis; "
                f"exclude_axes={self.exclude_axes} removed all axes from {x.axis_schema.axes}"
            )
        normalized = _zscore_normalize(tensor, dim=dims, epsilon=self.epsilon)
        metadata = x.metadata.copy() if is_tracking_meta() else x.metadata
        if is_tracking_meta():
            metadata.transforms.append(
                TransformRecord(
                    name="SampleZScoreNormalize",
                    params={"exclude_axes": list(self.exclude_axes), "epsilon": self.epsilon},
                    timestamp=time.time(),
                )
            )
        return RadioTensor(
            data=normalized.contiguous(),
            axis_schema=x.axis_schema,
            metadata=metadata,
        )


__all__ = ["Normalize", "SampleZScoreNormalize"]
