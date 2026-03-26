"""Contract-aware model-boundary alignment adapters."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Literal

import torch

from octosense.core.contracts import AxisContract
from octosense.core.contracts.model import ModelInputContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator
from octosense.transforms.core.validators import (
    infer_contract_family,
    resolve_model_input_contract,
    schema_from_model_input_contract,
    validate_aligned_sample,
)

ValueProjection = Literal["auto", "identity", "magnitude", "real", "imag"]
AxisGroup = tuple[str, ...]


class _GroupedAxisAdapter(BaseTransform):
    """Private helper that permutes and flattens named source-axis groups."""

    def __init__(
        self,
        *,
        target_axes: tuple[str, ...],
        axis_groups: tuple[AxisGroup, ...],
        value_projection: Literal["identity", "magnitude", "real", "imag"],
        record_name: str,
        contract: ModelInputContract,
    ) -> None:
        super().__init__()
        self.target_axes = tuple(str(axis) for axis in target_axes)
        self.axis_groups = tuple(tuple(group) for group in axis_groups)
        self.value_projection = value_projection
        self.record_name = record_name
        self.contract = contract
        flattened = [axis for group in self.axis_groups for axis in group]
        if len(self.target_axes) != len(self.axis_groups):
            raise ValueError("target_axes and axis_groups must have the same length")
        if any(not group for group in self.axis_groups):
            raise ValueError("Each target axis must map to at least one source axis")
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"axis_groups must not reuse source axes, got {flattened}")

    @property
    def input_contract(self) -> AxisContract:
        required_axes = [axis for group in self.axis_groups for axis in group]
        return AxisContract(required_axes=required_axes)

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=list(self.target_axes))

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        metadata: dict[str, AxisMetadata] = {}
        for target_axis, group in zip(self.target_axes, self.axis_groups, strict=True):
            if len(group) == 1:
                source_meta = input_rt.axis_schema.get_metadata(group[0])
                if source_meta is not None:
                    metadata[target_axis] = AxisMetadata(
                        name=target_axis,
                        unit=source_meta.unit,
                        description=source_meta.description,
                        semantic_id=source_meta.semantic_id,
                        axis_role=source_meta.axis_role,
                        code=source_meta.code,
                        kind=source_meta.kind,
                        category=source_meta.category,
                        status=source_meta.status,
                    )
                    continue
            metadata[target_axis] = AxisMetadata(
                name=target_axis,
                description=f"Aligned model-entry axis '{target_axis}'",
            )
        return schema_from_model_input_contract(self.contract, axis_metadata=metadata)

    def _project_value(self, data: torch.Tensor) -> torch.Tensor:
        if self.value_projection == "identity":
            return data
        if self.value_projection == "magnitude":
            return data.abs().float() if data.is_complex() else data.float()
        if self.value_projection == "real":
            return data.real.float() if data.is_complex() else data.float()
        if self.value_projection == "imag":
            if not data.is_complex():
                raise DimensionError("value='imag' requires a complex RadioTensor input")
            return data.imag.float()
        raise ValueError(f"Unsupported value projection '{self.value_projection}'")

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        grouped_axes = [axis for group in self.axis_groups for axis in group]
        unresolved = [
            axis_name
            for axis_name in x.axis_schema.axes
            if axis_name not in grouped_axes and axis_name != "batch"
        ]
        if unresolved:
            raise DimensionError(
                "AutoAlign could not resolve all source axes into the model entry contract.\n"
                f"Unresolved axes: {tuple(unresolved)}\n"
                f"Mapped axes: {tuple(grouped_axes)}\n"
                f"Available axes: {x.axis_schema.axes}"
            )

        permutation = [x.get_axis_index(axis_name) for axis_name in grouped_axes]
        data = self._project_value(x.as_tensor().permute(permutation).contiguous())

        group_sizes = []
        for group in self.axis_groups:
            size = 1
            for axis_name in group:
                size *= int(x.shape[x.get_axis_index(axis_name)])
            group_sizes.append(size)
        reshaped = data.reshape(*group_sizes).contiguous()

        metadata = x.metadata.copy()
        new_coords = {}
        for target_axis, group in zip(self.target_axes, self.axis_groups, strict=True):
            if len(group) == 1 and group[0] in metadata.coords:
                coord = metadata.coords[group[0]]
                new_coords[target_axis] = type(coord)(
                    axis_name=target_axis,
                    values=coord.values,
                    unit=coord.unit,
                )
        metadata.coords = dict(new_coords)
        metadata.transforms.append(
            TransformRecord(
                name=self.record_name,
                params={
                    "target_axes": list(self.target_axes),
                    "axis_groups": [list(group) for group in self.axis_groups],
                    "value_projection": self.value_projection,
                    "layout": self.contract.layout,
                },
                consumed_axes=grouped_axes,
                produced_axes=list(self.target_axes),
                timestamp=time.time(),
            )
        )
        return RadioTensor(reshaped, self.expected_output_schema(x), metadata)


@registered_operator(
    description="Align semantic RadioTensor samples to a model input contract.",
)
class AutoAlign(BaseTransform):
    """Align semantic ``RadioTensor`` samples to ``model.get_input_contract()``."""

    def __init__(
        self,
        model_or_contract: object | Mapping[str, object],
        *,
        axis_map: Mapping[str, str | tuple[str, ...]] | None = None,
        channel_axes: tuple[str, ...] | None = None,
        height_axis: str | None = None,
        width_axis: str | None = None,
        time_axis: str | None = None,
        flatten_axes: tuple[str, ...] | None = None,
        stack_other: bool = True,
        value: ValueProjection = "auto",
    ) -> None:
        super().__init__()
        if isinstance(model_or_contract, Mapping):
            self.contract = ModelInputContract(**model_or_contract)
        else:
            self.contract = resolve_model_input_contract(model_or_contract)
        self.axis_map = {str(key): value for key, value in (axis_map or {}).items()}
        self.channel_axes = None if channel_axes is None else tuple(channel_axes)
        self.height_axis = height_axis
        self.width_axis = width_axis
        self.time_axis = time_axis
        self.flatten_axes = None if flatten_axes is None else tuple(flatten_axes)
        self.stack_other = bool(stack_other)
        self.value = value

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract()

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=list(self.contract.axes))

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        del input_rt
        return schema_from_model_input_contract(self.contract)

    def _as_axis_group(self, value: str | tuple[str, ...] | None) -> AxisGroup | None:
        if value is None:
            return None
        if isinstance(value, str):
            return (value,)
        return tuple(str(axis_name) for axis_name in value)

    def _resolve_value_projection(self) -> Literal["identity", "magnitude", "real", "imag"]:
        if self.value != "auto":
            return self.value
        return "identity" if self.contract.dtype_kind == "complex" else "magnitude"

    def _pick_axis(
        self,
        x: RadioTensor,
        *,
        explicit: str | None,
        mapped_key: str,
    ) -> str:
        mapped = self.axis_map.get(mapped_key)
        if mapped is not None:
            group = self._as_axis_group(mapped)
            if group is None or len(group) != 1:
                raise ValueError(f"axis_map['{mapped_key}'] must reference exactly one source axis")
            axis_name = group[0]
        elif explicit is not None:
            axis_name = explicit
        else:
            axis_name = mapped_key if x.axis_schema.has_axis(mapped_key) else ""
        if not axis_name or not x.axis_schema.has_axis(axis_name):
            raise DimensionError(
                f"AutoAlign could not resolve source axis for target '{mapped_key}'.\n"
                f"Available axes: {x.axis_schema.axes}\n"
                "Fix: pass an explicit axis_map or axis override; AutoAlign does not "
                "ship modality-specific fallback candidates in the generic framework."
            )
        return axis_name

    def _resolve_image_groups(self, x: RadioTensor) -> tuple[AxisGroup, ...]:
        height_axis = self._pick_axis(
            x,
            explicit=self.height_axis,
            mapped_key="height",
        )
        width_axis = self._pick_axis(
            x,
            explicit=self.width_axis,
            mapped_key="width",
        )
        channel_group = (
            self._as_axis_group(self.axis_map.get("channel"))
            or self.channel_axes
        )
        if channel_group is None:
            excluded = {height_axis, width_axis, "batch"}
            inferred = tuple(axis_name for axis_name in x.axis_schema.axes if axis_name not in excluded)
            channel_group = inferred if self.stack_other else ()
        if not channel_group:
            raise DimensionError(
                "AutoAlign could not infer image channel axes.\n"
                f"Available axes: {x.axis_schema.axes}\n"
                "Fix: pass channel_axes=... or axis_map={'channel': (...)}."
            )
        return (tuple(channel_group), (height_axis,), (width_axis,))

    def _resolve_sequence_groups(self, x: RadioTensor) -> tuple[AxisGroup, ...]:
        time_axis = self._pick_axis(
            x,
            explicit=self.time_axis,
            mapped_key="time",
        )
        feature_group = (
            self._as_axis_group(self.axis_map.get("feature"))
            or self.flatten_axes
        )
        if feature_group is None:
            excluded = {time_axis, "batch"}
            feature_group = tuple(axis_name for axis_name in x.axis_schema.axes if axis_name not in excluded)
        if not feature_group:
            raise DimensionError(
                "AutoAlign could not infer sequence feature axes.\n"
                f"Available axes: {x.axis_schema.axes}\n"
                "Fix: pass flatten_axes=... or axis_map={'feature': (...)}."
            )
        return ((time_axis,), tuple(feature_group))

    def _resolve_vector_groups(self, x: RadioTensor) -> tuple[AxisGroup, ...]:
        target_axis = self.contract.axes[0]
        explicit = self._as_axis_group(self.axis_map.get(target_axis))
        if explicit is not None:
            return (explicit,)
        grouped = tuple(axis_name for axis_name in x.axis_schema.axes if axis_name != "batch")
        if not grouped:
            raise DimensionError("AutoAlign cannot flatten an empty axis set into a vector contract")
        return (grouped,)

    def _resolve_generic_groups(self, x: RadioTensor) -> tuple[AxisGroup, ...]:
        if not self.axis_map:
            if tuple(x.axis_schema.axes) == tuple(self.contract.axes):
                return tuple((axis_name,) for axis_name in self.contract.axes)
            raise DimensionError(
                "AutoAlign needs an explicit axis_map for generic model contracts.\n"
                f"Contract axes: {self.contract.axes}\n"
                f"Available axes: {x.axis_schema.axes}"
            )
        groups = []
        for target_axis in self.contract.axes:
            mapped = self._as_axis_group(self.axis_map.get(target_axis))
            if mapped is None:
                raise DimensionError(
                    f"AutoAlign axis_map is missing a source mapping for target axis '{target_axis}'."
                )
            groups.append(mapped)
        return tuple(groups)

    def _build_adapter(self, x: RadioTensor) -> _GroupedAxisAdapter:
        family = infer_contract_family(self.contract)
        if family == "image":
            axis_groups = self._resolve_image_groups(x)
        elif family == "sequence":
            axis_groups = self._resolve_sequence_groups(x)
        elif family == "vector":
            axis_groups = self._resolve_vector_groups(x)
        else:
            axis_groups = self._resolve_generic_groups(x)
        return _GroupedAxisAdapter(
            target_axes=self.contract.axes,
            axis_groups=axis_groups,
            value_projection=self._resolve_value_projection(),
            record_name="AutoAlign",
            contract=self.contract,
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        adapter = self._build_adapter(x)
        aligned = adapter(x)
        validate_aligned_sample(aligned, self.contract)
        return aligned

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.__class__.__name__,
            "params": {
                "model_or_contract": self.contract.to_dict(),
                "axis_map": dict(self.axis_map),
                "channel_axes": list(self.channel_axes) if self.channel_axes is not None else None,
                "height_axis": self.height_axis,
                "width_axis": self.width_axis,
                "time_axis": self.time_axis,
                "flatten_axes": list(self.flatten_axes) if self.flatten_axes is not None else None,
                "stack_other": self.stack_other,
                "value": self.value,
            },
        }


__all__ = ["AutoAlign"]
