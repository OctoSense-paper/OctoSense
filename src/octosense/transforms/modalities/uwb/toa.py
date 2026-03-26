"""UWB time-of-arrival estimation transforms."""

from __future__ import annotations

import torch

from octosense.core.contracts import AxisContract, MetadataRequirement
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator

_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


@registered_operator(
    required_axes=[],
    required_meta=["bandwidth"],
    description="Estimate Time-of-Arrival from UWB CIR leading edge.",
)
class ToAEstimation(BaseTransform):
    """Estimate UWB time-of-arrival from a CIR tap axis."""

    def __init__(self, threshold: float = 0.5, axis: str = "tap") -> None:
        super().__init__()
        self.threshold = threshold
        self.axis = axis

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis],
            dtype_constraint="complex",
            required_metadata=[
                MetadataRequirement("bandwidth", "physical", required=True),
            ],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(output_axes=[self.axis])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        dim = x.axis_schema.index(self.axis)
        data = x.as_tensor()
        power = torch.abs(data) ** 2
        peak_power, _ = torch.max(power, dim=dim, keepdim=True)
        threshold_mask = power >= self.threshold * peak_power
        leading_edge_indices = torch.argmax(threshold_mask.int(), dim=dim)

        tap_resolution = 1.0 / x.metadata.bandwidth
        toa = leading_edge_indices.float() * tap_resolution
        range_estimate = _SPEED_OF_LIGHT_M_PER_S * toa / 2.0

        track_meta = is_tracking_meta()
        metadata = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            metadata.extra["toa"] = toa.cpu().numpy()
            metadata.extra["range_estimate"] = range_estimate.cpu().numpy()
            metadata.extra["tap_resolution"] = tap_resolution
            metadata.add_transform(
                "ToAEstimation",
                {"threshold": self.threshold, "axis": self.axis},
            )

        return RadioTensor(data, x.axis_schema, metadata)


__all__ = ["ToAEstimation"]
