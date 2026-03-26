"""Axis-aware normalization transforms."""

import time
from typing import Literal

import torch
import torch.nn.functional as F

from octosense.core.contracts import AxisContract
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


@registered_operator(required_axes=[], description="Per-axis L2/max normalization.")
class AxisNormalize(BaseTransform):
    """Normalize along a named axis.

    Supports:
    - L2 normalization along a specific axis
    - Z-score normalization along a specific axis
    """

    def __init__(
        self,
        axis_name: str,
        method: Literal["l2", "zscore"] = "l2",
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.axis_name = axis_name
        self.method = method
        self.epsilon = epsilon

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract()

    def forward(self, x: RadioTensor) -> RadioTensor:
        axis_idx = x.get_axis_index(self.axis_name)
        data = x.as_tensor()

        if self.method == "l2":
            normalized = F.normalize(data, p=2.0, dim=axis_idx, eps=self.epsilon)
        elif self.method == "zscore":
            std, mean = torch.std_mean(data, dim=axis_idx, keepdim=True)
            normalized = (data - mean) / std.clamp_min(self.epsilon)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        new_metadata = x.metadata.copy()
        new_metadata.transforms.append(
            TransformRecord(
                name="AxisNormalize",
                params={
                    "axis_name": self.axis_name,
                    "method": self.method,
                    "epsilon": self.epsilon,
                },
                timestamp=time.time(),
            )
        )

        return RadioTensor(
            data=normalized,
            axis_schema=x.axis_schema,
            metadata=new_metadata,
        )


__all__ = ["AxisNormalize"]
