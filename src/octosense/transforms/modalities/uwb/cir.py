"""UWB CIR-specific transforms."""

from __future__ import annotations

import torch

from octosense.core.contracts import AxisContract, MetadataRequirement
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


# ---------------------------------------------------------------------------
# LeadingEdgeDetection
# ---------------------------------------------------------------------------


@registered_operator(
    required_axes=[],
    required_meta=["bandwidth"],
    description="Detect leading edge (first path) in UWB CIR.",
)
class LeadingEdgeDetection(BaseTransform):
    """Detect leading edge (first path arrival) in UWB CIR.

    Uses threshold-based detection on the CIR power profile to identify
    the first arriving path, which corresponds to the direct (line-of-sight)
    signal component.

    The output tensor has the same shape as input, with a binary mask
    where 1 indicates taps at or after the detected leading edge.

    Args:
        threshold: Detection threshold as fraction of peak power (default 0.5).
            Lower values detect weaker first paths but increase false alarm rate.
        axis: CIR tap axis name (default ``"tap"``).
    """

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

        # Compute power profile
        power = torch.abs(data) ** 2

        # Find peak power along tap axis
        peak_power, _ = torch.max(power, dim=dim, keepdim=True)

        # Threshold: first tap exceeding threshold * peak_power
        threshold_mask = power >= self.threshold * peak_power

        # Find leading edge index (first True along tap axis)
        # Use cumsum trick: first nonzero after cumsum == 1 marks the leading edge
        cumsum = torch.cumsum(threshold_mask.int(), dim=dim)
        leading_edge_mask = cumsum >= 1

        # Store leading edge indices in metadata
        track_meta = is_tracking_meta()
        meta = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            # Extract per-(time, ant) leading edge tap index
            # argmax on threshold_mask along tap dim gives first True index
            le_indices = torch.argmax(threshold_mask.int(), dim=dim)
            meta.extra["leading_edge_indices"] = le_indices.cpu().numpy()
            meta.add_transform(
                "LeadingEdgeDetection",
                {"threshold": self.threshold, "axis": self.axis},
            )

        # Output: original CIR masked to start from leading edge
        out = data * leading_edge_mask.to(data.dtype)

        return RadioTensor(out, x.axis_schema, meta)


@registered_operator(
    required_axes=[],
    required_meta=[],
    description="Normalize UWB CIR power (max-peak normalization).",
)
class CIRNormalize(BaseTransform):
    """Normalize UWB CIR by peak power.

    Divides the CIR by its peak magnitude along the tap axis so that
    the maximum amplitude is 1.0. This is useful for comparing CIR
    shapes across different measurements.

    Args:
        axis: CIR tap axis name (default ``"tap"``).
        eps: Small constant to avoid division by zero (default 1e-12).
    """

    def __init__(self, axis: str = "tap", eps: float = 1e-12) -> None:
        super().__init__()
        self.axis = axis
        self.eps = eps

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis],
            dtype_constraint="complex",
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(output_axes=[self.axis])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        dim = x.axis_schema.index(self.axis)
        data = x.as_tensor()

        # Peak magnitude along tap axis
        peak_mag, _ = torch.max(torch.abs(data), dim=dim, keepdim=True)
        peak_mag = torch.clamp(peak_mag, min=self.eps)

        # Normalize
        out = data / peak_mag

        # Metadata update
        track_meta = is_tracking_meta()
        meta = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            meta.add_transform(
                "CIRNormalize",
                {"axis": self.axis, "eps": self.eps},
            )

        return RadioTensor(out, x.axis_schema, meta)


__all__ = ["CIRNormalize", "LeadingEdgeDetection"]
