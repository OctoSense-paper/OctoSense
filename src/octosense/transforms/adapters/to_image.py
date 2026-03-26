"""RadioTensor reshaping utilities for model input."""

import time

from octosense.core.contracts import AxisContract
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.registry import registered_operator
from octosense.transforms.core.base import BaseTransform


@registered_operator(
    required_axes=[],
    description="Reshape RadioTensor to image layout (C,H,W).",
)
class ToImage(BaseTransform):
    """Convert RadioTensor to NCHW image layout.

    Expected input axes include `frame` and `freq`. Selected axes are merged into
    the channel dimension in the given order.
    """

    def __init__(
        self,
        channel_axes: tuple[str, ...] | None = None,
        height_axis: str = "freq",
        width_axis: str = "frame",
        stack_other: bool = False,
    ) -> None:
        super().__init__()
        self.channel_axes = channel_axes
        self.height_axis = height_axis
        self.width_axis = width_axis
        self.stack_other = bool(stack_other)

    def _resolve_channel_axes(self, x: RadioTensor) -> tuple[str, ...]:
        resolved = tuple(self.channel_axes or ())
        if self.stack_other:
            excluded = {self.height_axis, self.width_axis, "batch", *resolved}
            resolved = resolved + tuple(
                axis_name for axis_name in x.axis_schema.axes if axis_name not in excluded
            )
        if not resolved:
            raise ValueError(
                "ToImage could not resolve channel axes. Specify channel_axes=... or set stack_other=True "
                "with non-spatial axes available."
            )
        return resolved

    @property
    def input_contract(self) -> AxisContract:
        required = [self.height_axis, self.width_axis]
        if self.channel_axes is not None:
            required = list(self.channel_axes) + required
        return AxisContract(required_axes=required)

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract()

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        axes = list(input_rt.axis_schema.axes)
        height_idx = input_rt.get_axis_index(self.height_axis)
        width_idx = input_rt.get_axis_index(self.width_axis)
        channel_axes = self._resolve_channel_axes(input_rt)
        channel_indices = [input_rt.get_axis_index(ax) for ax in channel_axes]
        remaining = [
            i
            for i in range(len(axes))
            if i not in channel_indices + [height_idx, width_idx]
        ]
        new_axes = [axes[i] for i in remaining] + ["channel", self.height_axis, self.width_axis]
        axis_metadata = {
            k: v for k, v in input_rt.axis_schema.axis_metadata.items() if k in new_axes
        }
        axis_metadata.setdefault("channel", AxisMetadata("channel", None, "Merged channel axis"))
        return AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        axes = list(x.axis_schema.axes)
        height_idx = x.get_axis_index(self.height_axis)
        width_idx = x.get_axis_index(self.width_axis)
        channel_axes = self._resolve_channel_axes(x)
        channel_indices = [x.get_axis_index(ax) for ax in channel_axes]

        # Determine remaining axes (e.g., batch)
        remaining = [
            i
            for i in range(len(axes))
            if i not in channel_indices + [height_idx, width_idx]
        ]

        # Permute to (remaining..., channel_axes..., height, width)
        perm = remaining + channel_indices + [height_idx, width_idx]
        data = x.as_tensor().permute(perm)

        # Merge channel axes into a single channel dim
        channel_size = 1
        for idx in channel_indices:
            channel_size *= x.shape[idx]

        if remaining:
            leading_shape = [x.shape[i] for i in remaining]
            data = data.reshape(
                *leading_shape,
                channel_size,
                x.shape[height_idx],
                x.shape[width_idx],
            )
        else:
            data = data.reshape(channel_size, x.shape[height_idx], x.shape[width_idx])
        data = data.contiguous()

        # Build new schema
        new_axes = []
        if remaining:
            new_axes.extend([axes[i] for i in remaining])
        new_axes.extend(["channel", self.height_axis, self.width_axis])
        axis_metadata = {k: v for k, v in x.axis_schema.axis_metadata.items() if k in new_axes}
        axis_metadata.setdefault(
            "channel",
            AxisMetadata("channel", None, "Merged channel axis"),
        )
        new_schema = AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

        new_metadata = x.metadata.copy()
        for ax in channel_axes:
            if ax in new_metadata.coords:
                del new_metadata.coords[ax]
        new_metadata.transforms.append(
            TransformRecord(
                name="ToImage",
                params={
                    "channel_axes": channel_axes,
                    "height_axis": self.height_axis,
                    "width_axis": self.width_axis,
                    "stack_other": self.stack_other,
                },
                timestamp=time.time(),
            )
        )

        return RadioTensor(data=data, axis_schema=new_schema, metadata=new_metadata)
