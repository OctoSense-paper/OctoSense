"""Canonical radar projection adapters."""

from __future__ import annotations

from typing import Literal

import torch

from octosense.core.contracts import AxisContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


class _RadarProjectionBase(BaseTransform):
    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=["range", "doppler", "angle"])

    def _require_axes(self, x: RadioTensor, *, output_name: str) -> None:
        if not x.axis_schema.has_axis("range"):
            raise DimensionError(
                f"{output_name} requires 'range' axis.\n"
                f"Current axes: {x.axis_schema.axes}\n"
                "Fix: Apply RangeFFT before this projection."
            )
        if not x.axis_schema.has_axis("doppler"):
            raise DimensionError(
                f"{output_name} requires 'doppler' axis.\n"
                f"Current axes: {x.axis_schema.axes}\n"
                "Fix: Apply DopplerFFT before this projection."
            )
        if not x.axis_schema.has_axis("angle"):
            raise DimensionError(
                f"{output_name} requires 'angle' axis.\n"
                f"Current axes: {x.axis_schema.axes}\n"
                "Fix: Apply AngleFFT before this projection."
            )

    @staticmethod
    def _reduce_axis(
        data: torch.Tensor,
        *,
        dim: int,
        reduce: Literal["sum", "max", "mean"],
    ) -> torch.Tensor:
        power = data.abs().pow(2)
        if reduce == "sum":
            return power.sum(dim=dim)
        if reduce == "max":
            return power.max(dim=dim).values
        if reduce == "mean":
            return power.mean(dim=dim)
        raise ValueError(f"Unknown reduce mode '{reduce}'")

    @staticmethod
    def _project_tensor(
        x: RadioTensor,
        *,
        remove_axis: str,
        reduced: torch.Tensor,
        record_name: str,
        record_params: dict[str, object],
    ) -> RadioTensor:
        new_axes = tuple(axis for axis in x.axis_schema.axes if axis != remove_axis)
        new_axis_meta = {
            axis: meta
            for axis, meta in x.axis_schema.axis_metadata.items()
            if axis != remove_axis
        }
        new_schema = AxisSchema(axes=new_axes, axis_metadata=new_axis_meta)

        meta = x.metadata.copy() if is_tracking_meta() else x.metadata
        if is_tracking_meta():
            stale = [key for key in meta.coords if key not in new_axes]
            for key in stale:
                del meta.coords[key]
            meta.add_transform(
                record_name,
                record_params,
                consumed_axes=[remove_axis],
                produced_axes=list(new_axes),
            )
        return RadioTensor(reduced, new_schema, meta)


@registered_operator(
    description="Reduce the angle axis and return a range-doppler RadioTensor.",
)
class ToRangeDoppler(_RadarProjectionBase):
    """Reduce the angle axis and return a range-doppler ``RadioTensor``."""

    def __init__(
        self,
        reduce: Literal["sum", "max", "mean"] = "sum",
        to_db: bool = False,
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.to_db = bool(to_db)

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(remove_axes=["angle"])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        self._require_axes(x, output_name="ToRangeDoppler")
        reduced = self._reduce_axis(
            x.as_tensor(),
            dim=x.axis_schema.index("angle"),
            reduce=self.reduce,
        )
        if self.to_db:
            reduced = 10.0 * torch.log10(reduced.clamp(min=1e-20))
        projected = self._project_tensor(
            x,
            remove_axis="angle",
            reduced=reduced,
            record_name="ToRangeDoppler",
            record_params={"reduce": self.reduce, "to_db": self.to_db},
        )
        return projected


@registered_operator(
    description="Reduce the doppler axis and return a range-angle RadioTensor.",
)
class ToRangeAngle(_RadarProjectionBase):
    """Reduce the doppler axis and return a range-angle ``RadioTensor``."""

    def __init__(
        self,
        reduce: Literal["sum", "max", "mean"] = "sum",
        to_db: bool = False,
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.to_db = bool(to_db)

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(remove_axes=["doppler"])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        self._require_axes(x, output_name="ToRangeAngle")
        reduced = self._reduce_axis(
            x.as_tensor(),
            dim=x.axis_schema.index("doppler"),
            reduce=self.reduce,
        )
        if self.to_db:
            reduced = 10.0 * torch.log10(reduced.clamp(min=1e-20))
        projected = self._project_tensor(
            x,
            remove_axis="doppler",
            reduced=reduced,
            record_name="ToRangeAngle",
            record_params={"reduce": self.reduce, "to_db": self.to_db},
        )
        return projected

__all__ = [
    "ToRangeAngle",
    "ToRangeDoppler",
]
