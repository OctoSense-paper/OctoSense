"""Canonical complex-view adapters."""

from __future__ import annotations

import time
from typing import Any

from octosense.core.contracts import AxisContract
from octosense.io.semantics.metadata import CoordinateAxis, TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform


def _rename_radiotensor_axes(
    rt: RadioTensor,
    mapping: dict[str, str],
    *,
    record_name: str | None = None,
    record_params: dict[str, Any] | None = None,
) -> RadioTensor:
    new_axes = tuple(mapping.get(axis_name, axis_name) for axis_name in rt.axis_schema.axes)
    axis_metadata: dict[str, AxisMetadata] = {}
    for axis_name in rt.axis_schema.axes:
        metadata = rt.axis_schema.get_metadata(axis_name)
        if metadata is None:
            continue
        renamed = mapping.get(axis_name, axis_name)
        axis_metadata[renamed] = AxisMetadata(
            name=renamed,
            unit=metadata.unit,
            description=metadata.description,
        )

    new_metadata = rt.metadata.copy()
    for old_name, new_name in mapping.items():
        coord = new_metadata.coords.pop(old_name, None)
        if coord is not None:
            new_metadata.coords[new_name] = CoordinateAxis(
                axis_name=new_name,
                values=coord.values,
                unit=coord.unit,
            )

    if record_name is not None:
        new_metadata.transforms.append(
            TransformRecord(
                name=record_name,
                params=record_params or dict(mapping),
                consumed_axes=list(mapping.keys()),
                produced_axes=list(mapping.values()),
                timestamp=time.time(),
            )
        )

    return RadioTensor(rt.as_tensor(), AxisSchema(new_axes, axis_metadata=axis_metadata), new_metadata)


class ConjugateWithReferenceRx(BaseTransform):
    """Broadcast one reference RX conjugate and relabel the derived feature axis."""

    def __init__(
        self,
        rx_axis: str = "rx",
        ref_rx: int = 0,
        axis_name: str = "subc",
        new_axis: str = "conj.subc",
        allow_passthrough: bool = False,
    ) -> None:
        super().__init__()
        self.rx_axis = rx_axis
        self.ref_rx = ref_rx
        self.axis_name = axis_name
        self.new_axis = new_axis
        self.allow_passthrough = allow_passthrough

    def _resolve_effective_rx_axis(self, x: RadioTensor) -> str | None:
        if x.axis_schema.has_axis(self.rx_axis):
            return self.rx_axis
        if self.allow_passthrough and x.axis_schema.has_axis("link"):
            link_dim = x.get_axis_index("link")
            if int(x.shape[link_dim]) == 2:
                return "link"
        return None

    @property
    def input_contract(self) -> AxisContract:
        if self.allow_passthrough:
            return AxisContract(required_axes=[self.axis_name])
        return AxisContract(required_axes=[self.axis_name, self.rx_axis], dtype_constraint="complex")

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.axis_name], output_axes=[self.new_axis])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        if self.allow_passthrough:
            effective_rx_axis = self._resolve_effective_rx_axis(x)
            if effective_rx_axis is None:
                return _rename_radiotensor_axes(
                    RadioTensor(x.as_tensor(), x.axis_schema, x.metadata.copy()),
                    {self.axis_name: self.new_axis},
                    record_name="ConjugateWithReferenceRx",
                    record_params={
                        "rx_axis": self.rx_axis,
                        "ref_rx": self.ref_rx,
                        "axis_name": self.axis_name,
                        "new_axis": self.new_axis,
                        "passthrough": True,
                        "reason": "missing_rx_like_axis",
                    },
                )
        else:
            effective_rx_axis = self.rx_axis

        rx_dim = x.get_axis_index(effective_rx_axis)
        if not 0 <= self.ref_rx < x.shape[rx_dim]:
            raise ValueError(
                f"ref_rx={self.ref_rx} is out of range for axis '{effective_rx_axis}' "
                f"with size {x.shape[rx_dim]}"
            )

        data = x.as_tensor()
        ref = data.select(rx_dim, self.ref_rx).unsqueeze(rx_dim)
        conjugated = data * ref.conj()

        renamed = _rename_radiotensor_axes(
            RadioTensor(conjugated, x.axis_schema, x.metadata.copy()),
            {self.axis_name: self.new_axis},
        )
        renamed.metadata.transforms.append(
            TransformRecord(
                name="ConjugateWithReferenceRx",
                params={
                    "rx_axis": self.rx_axis,
                    "effective_rx_axis": effective_rx_axis,
                    "ref_rx": self.ref_rx,
                    "axis_name": self.axis_name,
                    "new_axis": self.new_axis,
                },
                consumed_axes=[self.axis_name, effective_rx_axis],
                produced_axes=[self.new_axis, effective_rx_axis],
                derivation=(
                    "Multiply each RX channel with the conjugate of one reference RX "
                    "and relabel the derived feature axis."
                ),
                timestamp=time.time(),
            )
        )
        return renamed


__all__ = ["ConjugateWithReferenceRx"]
