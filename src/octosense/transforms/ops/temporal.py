"""Canonical temporal operators.

Temporal operators live here; representation adapters remain under
``octosense.transforms.adapters``.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from octosense.core.contracts import AxisContract
from octosense.io.semantics.metadata import CoordinateAxis
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


@registered_operator(
    required_axes=[],
    description="Resize a named temporal axis to a fixed number of samples.",
)
class TemporalResize(BaseTransform):
    """Resize a temporal axis with torch-native interpolation."""

    def __init__(
        self,
        *,
        axis_name: str = "time",
        target_length: int,
        mode: str = "linear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if int(target_length) < 1:
            raise ValueError("TemporalResize target_length must be >= 1")
        if mode not in {"linear", "nearest"}:
            raise ValueError("TemporalResize only supports mode='linear' or mode='nearest'")
        self.axis_name = axis_name
        self.target_length = int(target_length)
        self.mode = mode
        self.align_corners = bool(align_corners)

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.axis_name])

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.axis_name])

    def _interpolate(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1, 1, int(values.shape[-1]))
        kwargs = {"size": self.target_length, "mode": self.mode}
        if self.mode != "nearest":
            kwargs["align_corners"] = self.align_corners
        return F.interpolate(flat, **kwargs).reshape(*values.shape[:-1], self.target_length)

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        axis_idx = x.get_axis_index(self.axis_name)
        values = x.as_tensor().movedim(axis_idx, -1).contiguous()
        if int(values.shape[-1]) == self.target_length:
            resized = values
        elif torch.is_complex(values):
            real = self._interpolate(values.real.float())
            imag = self._interpolate(values.imag.float())
            resized = torch.complex(real, imag)
        else:
            resized = self._interpolate(values.float()).to(dtype=values.dtype)

        output = resized.movedim(-1, axis_idx).contiguous()
        metadata = x.metadata.copy()
        coord = metadata.get_coord(self.axis_name)
        if coord is not None and coord.values is not None and len(coord.values) > 0:
            raw_values = np.asarray(coord.values, dtype=np.float64)
            if raw_values.size == 1:
                resized_coord = np.repeat(raw_values, self.target_length)
            else:
                resized_coord = np.linspace(
                    float(raw_values[0]),
                    float(raw_values[-1]),
                    self.target_length,
                    dtype=np.float64,
                )
            metadata.coords[self.axis_name] = CoordinateAxis(
                self.axis_name,
                values=resized_coord,
                unit=coord.unit,
            )
        metadata.transforms.append(
            TransformRecord(
                name="TemporalResize",
                params={
                    "axis_name": self.axis_name,
                    "target_length": self.target_length,
                    "mode": self.mode,
                    "align_corners": self.align_corners,
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(output, x.axis_schema, metadata)


__all__ = ["TemporalResize"]
