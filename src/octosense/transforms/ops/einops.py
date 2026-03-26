"""Einops-based transforms for RadioTensor dimension operations.

This module owns both the transform classes and their RadioTensor-aware tensor
logic. The old ``octosense.functional.einops_ops`` layer was redundant because
the implementation already depended on transform semantics, metadata tracking,
and ``RadioTensor`` contracts.
"""

import re
import time
from typing import Any

import einops
import torch

from octosense.core.contracts import AxisContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


def _parse_axis_names(side: str) -> list[str | tuple[str, ...]]:
    """Parse one side of an einops pattern into semantic axis names."""
    result: list[str | tuple[str, ...]] = []
    pattern = r"\(([^)]+)\)|(\w+)"
    for match in re.finditer(pattern, side):
        if match.group(1):
            result.append(tuple(match.group(1).split()))
        elif match.group(2):
            result.append(match.group(2))
    return result


def _validate_pattern_axes(
    schema: AxisSchema,
    lhs_names: list[str | tuple[str, ...]],
) -> None:
    primary_axes = []
    for name in lhs_names:
        if isinstance(name, tuple):
            primary_axes.append("_".join(name))
        else:
            primary_axes.append(name)

    if len(primary_axes) != len(schema.axes):
        raise DimensionError(
            f"Pattern LHS axis count ({len(primary_axes)}) doesn't match "
            f"AxisSchema axis count ({len(schema.axes)}). "
            f"Pattern axes: {primary_axes}, Schema axes: {list(schema.axes)}"
        )

    for i, (lhs_name, schema_axis) in enumerate(zip(lhs_names, schema.axes, strict=True)):
        if isinstance(lhs_name, str) and lhs_name != schema_axis:
            raise DimensionError(
                f"Pattern LHS axis '{lhs_name}' at position {i} doesn't match "
                f"AxisSchema axis '{schema_axis}'. "
                f"Fix: Use '{schema_axis}' in pattern or reorder pattern axes."
            )


def _build_new_schema(
    old_schema: AxisSchema,
    rhs_names: list[str | tuple[str, ...]],
) -> AxisSchema:
    new_axes = []
    new_axis_metadata = {}

    for rhs_name in rhs_names:
        if isinstance(rhs_name, tuple):
            new_axes.append("_".join(rhs_name))
            continue

        new_axes.append(rhs_name)
        if rhs_name in old_schema.axis_metadata:
            new_axis_metadata[rhs_name] = old_schema.axis_metadata[rhs_name]
        elif rhs_name not in old_schema.axes:
            new_axis_metadata[rhs_name] = AxisMetadata(
                name=rhs_name,
                unit=None,
                description=f"Axis '{rhs_name}' created by einops operation",
            )

    return AxisSchema(axes=tuple(new_axes), axis_metadata=new_axis_metadata)


def _update_metadata_for_einops(
    old_metadata: Any,
    lhs_names: list[str | tuple[str, ...]],
    rhs_names: list[str | tuple[str, ...]],
    operation: str,
    pattern: str,
    **kwargs: Any,
) -> Any:
    if not is_tracking_meta():
        return old_metadata

    new_metadata = old_metadata.copy()

    lhs_source_axes = []
    for name in lhs_names:
        if isinstance(name, tuple):
            lhs_source_axes.append("_".join(name))
        else:
            lhs_source_axes.append(name)

    preserved_axes = {
        name for name in rhs_names if isinstance(name, str) and name in lhs_source_axes
    }

    eliminated_axes = set(lhs_source_axes) - preserved_axes
    for axis_name in eliminated_axes:
        if axis_name in new_metadata.coords:
            del new_metadata.coords[axis_name]

    new_metadata.transforms.append(
        TransformRecord(
            name=f"einops.{operation}",
            params={"pattern": pattern, **kwargs},
            timestamp=time.time(),
        )
    )

    return new_metadata


def _is_plain_axis_permutation(names: list[str | tuple[str, ...]]) -> bool:
    return all(isinstance(name, str) for name in names)


def _native_rearrange_tensor(
    tensor: torch.Tensor,
    lhs_names: list[str | tuple[str, ...]],
    rhs_names: list[str | tuple[str, ...]],
) -> torch.Tensor | None:
    if not (_is_plain_axis_permutation(lhs_names) and _is_plain_axis_permutation(rhs_names)):
        return None
    lhs_axes = [str(name) for name in lhs_names]
    rhs_axes = [str(name) for name in rhs_names]
    if sorted(lhs_axes) != sorted(rhs_axes):
        return None
    permutation = [lhs_axes.index(axis_name) for axis_name in rhs_axes]
    return tensor.permute(*permutation)


def _native_reduce_tensor(
    tensor: torch.Tensor,
    lhs_names: list[str | tuple[str, ...]],
    rhs_names: list[str | tuple[str, ...]],
    reduction: str,
) -> torch.Tensor | None:
    if not (_is_plain_axis_permutation(lhs_names) and _is_plain_axis_permutation(rhs_names)):
        return None
    lhs_axes = [str(name) for name in lhs_names]
    rhs_axes = [str(name) for name in rhs_names]
    if len(set(rhs_axes)) != len(rhs_axes):
        return None
    if any(axis_name not in lhs_axes for axis_name in rhs_axes):
        return None
    reduce_dims = tuple(index for index, axis_name in enumerate(lhs_axes) if axis_name not in rhs_axes)
    if not reduce_dims:
        return tensor
    if reduction == "mean":
        return tensor.mean(dim=reduce_dims)
    if reduction == "sum":
        return tensor.sum(dim=reduce_dims)
    if reduction == "prod":
        return tensor.prod(dim=reduce_dims)
    if reduction == "max":
        return torch.amax(tensor, dim=reduce_dims)
    if reduction == "min":
        return torch.amin(tensor, dim=reduce_dims)
    return None


def _native_repeat_tensor(
    tensor: torch.Tensor,
    lhs_names: list[str | tuple[str, ...]],
    rhs_names: list[str | tuple[str, ...]],
    axes_lengths: dict[str, int],
) -> torch.Tensor | None:
    if not (_is_plain_axis_permutation(lhs_names) and _is_plain_axis_permutation(rhs_names)):
        return None
    lhs_axes = [str(name) for name in lhs_names]
    rhs_axes = [str(name) for name in rhs_names]
    if rhs_axes[: len(lhs_axes)] != lhs_axes:
        return None
    if any(axis_name in lhs_axes for axis_name in rhs_axes[len(lhs_axes) :]):
        return None
    expanded = tensor
    expand_shape = list(tensor.shape)
    for axis_name in rhs_axes[len(lhs_axes) :]:
        if axis_name not in axes_lengths:
            return None
        expanded = expanded.unsqueeze(-1)
        expand_shape.append(int(axes_lengths[axis_name]))
    return expanded.expand(*expand_shape)


def _rearrange_tensor(rt: RadioTensor, pattern: str, **axes_lengths: int) -> RadioTensor:
    if "->" not in pattern:
        raise ValueError(f"Pattern must contain '->': {pattern}")

    lhs, rhs = pattern.split("->")
    lhs_names = _parse_axis_names(lhs.strip())
    rhs_names = _parse_axis_names(rhs.strip())
    _validate_pattern_axes(rt.axis_schema, lhs_names)

    data_out = _native_rearrange_tensor(rt.as_tensor(), lhs_names, rhs_names)
    if data_out is None:
        data_out = einops.rearrange(rt.as_tensor(), pattern, **axes_lengths)
    if not data_out.is_contiguous():
        data_out = data_out.contiguous()

    return RadioTensor(
        data=data_out,
        axis_schema=_build_new_schema(rt.axis_schema, rhs_names),
        metadata=_update_metadata_for_einops(
            rt.metadata,
            lhs_names,
            rhs_names,
            operation="rearrange",
            pattern=pattern,
            axes_lengths=axes_lengths,
        ),
    )


def _reduce_tensor(
    rt: RadioTensor,
    pattern: str,
    reduction: str,
    **axes_lengths: int,
) -> RadioTensor:
    if "->" not in pattern:
        raise ValueError(f"Pattern must contain '->': {pattern}")

    lhs, rhs = pattern.split("->")
    lhs_names = _parse_axis_names(lhs.strip())
    rhs_names = _parse_axis_names(rhs.strip())
    _validate_pattern_axes(rt.axis_schema, lhs_names)

    input_tensor = rt.as_tensor()
    native_input = torch.abs(input_tensor) if input_tensor.is_complex() else input_tensor
    data_out = _native_reduce_tensor(native_input, lhs_names, rhs_names, reduction)
    if data_out is None:
        if input_tensor.is_complex():
            reduced = einops.reduce(torch.abs(input_tensor), pattern, reduction, **axes_lengths)
            data_out = reduced.to(dtype=input_tensor.dtype)
        else:
            data_out = einops.reduce(input_tensor, pattern, reduction, **axes_lengths)
    elif input_tensor.is_complex():
        data_out = data_out.to(dtype=input_tensor.dtype)

    if not data_out.is_contiguous():
        data_out = data_out.contiguous()

    return RadioTensor(
        data=data_out,
        axis_schema=_build_new_schema(rt.axis_schema, rhs_names),
        metadata=_update_metadata_for_einops(
            rt.metadata,
            lhs_names,
            rhs_names,
            operation="reduce",
            pattern=pattern,
            reduction=reduction,
            axes_lengths=axes_lengths,
        ),
    )


def _repeat_tensor(rt: RadioTensor, pattern: str, **axes_lengths: int) -> RadioTensor:
    if "->" not in pattern:
        raise ValueError(f"Pattern must contain '->': {pattern}")

    lhs, rhs = pattern.split("->")
    lhs_names = _parse_axis_names(lhs.strip())
    rhs_names = _parse_axis_names(rhs.strip())
    _validate_pattern_axes(rt.axis_schema, lhs_names)

    data_out = _native_repeat_tensor(rt.as_tensor(), lhs_names, rhs_names, axes_lengths)
    if data_out is None:
        data_out = einops.repeat(rt.as_tensor(), pattern, **axes_lengths)
    if not data_out.is_contiguous():
        data_out = data_out.contiguous()

    return RadioTensor(
        data=data_out,
        axis_schema=_build_new_schema(rt.axis_schema, rhs_names),
        metadata=_update_metadata_for_einops(
            rt.metadata,
            lhs_names,
            rhs_names,
            operation="repeat",
            pattern=pattern,
            axes_lengths=axes_lengths,
        ),
    )


@registered_operator(required_axes=[], description="Einops rearrange dimension reorder.")
class Rearrange(BaseTransform):
    """Rearrange RadioTensor dimensions using einops pattern.

    Args:
        pattern: Einops rearrange pattern (e.g., 'batch time subc rx -> batch time (subc rx)')
        **axes_lengths: Explicit axis lengths for split operations
    """

    def __init__(self, pattern: str, **axes_lengths: int) -> None:
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    @property
    def input_contract(self) -> AxisContract:
        if "->" not in self.pattern:
            raise ValueError(f"Pattern must contain '->': {self.pattern}")
        lhs = self.pattern.split("->")[0].strip()
        lhs_names = _parse_axis_names(lhs)
        required_axes = []
        for name in lhs_names:
            if isinstance(name, tuple):
                required_axes.append("_".join(name))
            else:
                required_axes.append(name)
        return AxisContract(required_axes=required_axes)

    @property
    def output_contract(self) -> AxisContract:
        if "->" not in self.pattern:
            raise ValueError(f"Pattern must contain '->': {self.pattern}")
        rhs = self.pattern.split("->")[1].strip()
        rhs_names = _parse_axis_names(rhs)
        output_axes = []
        for name in rhs_names:
            if isinstance(name, tuple):
                output_axes.append("_".join(name))
            else:
                output_axes.append(name)
        return AxisContract(output_axes=output_axes)

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        rhs = self.pattern.split("->")[1].strip()
        rhs_names = _parse_axis_names(rhs)
        output_axes = []
        for name in rhs_names:
            if isinstance(name, tuple):
                output_axes.append("_".join(name))
            else:
                output_axes.append(name)
        axis_metadata = {
            axis_name: input_rt.axis_schema.axis_metadata[axis_name]
            for axis_name in output_axes
            if axis_name in input_rt.axis_schema.axis_metadata
        }
        return AxisSchema(tuple(output_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        return _rearrange_tensor(x, self.pattern, **self.axes_lengths)

    def to_dict(self) -> dict:
        """Serialize with axes_lengths flattened into params."""
        params = {"pattern": self.pattern}
        # Flatten axes_lengths into params (no nesting)
        params.update(self.axes_lengths)
        return {"name": self.__class__.__name__, "params": params, "version": "1.0"}


@registered_operator(required_axes=[], description="Einops reduce dimension collapse.")
class Reduce(BaseTransform):
    """Reduce RadioTensor dimensions using einops pattern.

    Args:
        pattern: Einops reduce pattern (e.g., 'batch time subc rx -> batch time subc')
        reduction: Reduction operation ('mean', 'max', 'min', 'sum', 'prod')
        **axes_lengths: Explicit axis lengths for pooling operations
    """

    def __init__(self, pattern: str, reduction: str = "mean", **axes_lengths: int) -> None:
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths

    @property
    def input_contract(self) -> AxisContract:
        if "->" not in self.pattern:
            raise ValueError(f"Pattern must contain '->': {self.pattern}")
        lhs = self.pattern.split("->")[0].strip()
        lhs_names = _parse_axis_names(lhs)
        required_axes = []
        for name in lhs_names:
            if isinstance(name, tuple):
                required_axes.append("_".join(name))
            else:
                required_axes.append(name)
        return AxisContract(required_axes=required_axes)

    @property
    def output_contract(self) -> AxisContract:
        if "->" not in self.pattern:
            raise ValueError(f"Pattern must contain '->': {self.pattern}")
        rhs = self.pattern.split("->")[1].strip()
        rhs_names = _parse_axis_names(rhs)
        output_axes = []
        for name in rhs_names:
            if isinstance(name, tuple):
                output_axes.append("_".join(name))
            else:
                output_axes.append(name)
        return AxisContract(output_axes=output_axes)

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        rhs = self.pattern.split("->")[1].strip()
        rhs_names = _parse_axis_names(rhs)
        output_axes = []
        for name in rhs_names:
            if isinstance(name, tuple):
                output_axes.append("_".join(name))
            else:
                output_axes.append(name)
        axis_metadata = {
            axis_name: input_rt.axis_schema.axis_metadata[axis_name]
            for axis_name in output_axes
            if axis_name in input_rt.axis_schema.axis_metadata
        }
        return AxisSchema(tuple(output_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        return _reduce_tensor(x, self.pattern, self.reduction, **self.axes_lengths)

    def to_dict(self) -> dict:
        """Serialize with axes_lengths flattened into params."""
        params = {"pattern": self.pattern, "reduction": self.reduction}
        params.update(self.axes_lengths)
        return {"name": self.__class__.__name__, "params": params, "version": "1.0"}


@registered_operator(required_axes=[], description="Einops repeat dimension expansion.")
class Repeat(BaseTransform):
    """Repeat RadioTensor dimensions using einops pattern.

    Args:
        pattern: Einops repeat pattern (e.g., 'batch time subc -> batch time subc rx')
        **axes_lengths: Required lengths for new axes
    """

    def __init__(self, pattern: str, **axes_lengths: int) -> None:
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    @property
    def input_contract(self) -> AxisContract:
        if "->" not in self.pattern:
            raise ValueError(f"Pattern must contain '->': {self.pattern}")
        lhs = self.pattern.split("->")[0].strip()
        lhs_names = _parse_axis_names(lhs)
        required_axes = []
        for name in lhs_names:
            if isinstance(name, tuple):
                required_axes.append("_".join(name))
            else:
                required_axes.append(name)
        return AxisContract(required_axes=required_axes)

    @property
    def output_contract(self) -> AxisContract:
        if "->" not in self.pattern:
            raise ValueError(f"Pattern must contain '->': {self.pattern}")
        rhs = self.pattern.split("->")[1].strip()
        rhs_names = _parse_axis_names(rhs)
        output_axes = []
        for name in rhs_names:
            if isinstance(name, tuple):
                output_axes.append("_".join(name))
            else:
                output_axes.append(name)
        return AxisContract(output_axes=output_axes)

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        rhs = self.pattern.split("->")[1].strip()
        rhs_names = _parse_axis_names(rhs)
        output_axes = []
        for name in rhs_names:
            if isinstance(name, tuple):
                output_axes.append("_".join(name))
            else:
                output_axes.append(name)
        axis_metadata = {
            axis_name: input_rt.axis_schema.axis_metadata[axis_name]
            for axis_name in output_axes
            if axis_name in input_rt.axis_schema.axis_metadata
        }
        return AxisSchema(tuple(output_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        return _repeat_tensor(x, self.pattern, **self.axes_lengths)

    def to_dict(self) -> dict:
        """Serialize with axes_lengths flattened into params."""
        params = {"pattern": self.pattern}
        params.update(self.axes_lengths)
        return {"name": self.__class__.__name__, "params": params, "version": "1.0"}


__all__ = ["Rearrange", "Reduce", "Repeat", "_parse_axis_names"]
