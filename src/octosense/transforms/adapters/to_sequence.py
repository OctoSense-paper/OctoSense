"""Sequence-oriented transforms for recurrent RF models."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch

from octosense.core.contracts import AxisContract
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.registry import registered_operator
from octosense.transforms.core.base import BaseTransform


@registered_operator(
    required_axes=[],
    description="Flatten named RF axes into a feature dimension for sequence models.",
)
class ToSequence(BaseTransform):
    """Convert a RadioTensor into a ``(time, feature)``-style sequence."""

    def __init__(
        self,
        flatten_axes: tuple[str, ...],
        *,
        time_axis: str = "time",
        value: Literal["magnitude", "real", "imag"] = "magnitude",
    ) -> None:
        super().__init__()
        if not flatten_axes:
            raise ValueError("flatten_axes must not be empty")
        self.flatten_axes = flatten_axes
        self.time_axis = time_axis
        self.value = value

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.time_axis, *self.flatten_axes])

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract()

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        axis_names = list(input_rt.axis_schema.axes)
        time_idx = input_rt.get_axis_index(self.time_axis)
        flatten_indices = [input_rt.get_axis_index(name) for name in self.flatten_axes]
        remaining = [
            idx for idx in range(len(axis_names)) if idx not in flatten_indices and idx != time_idx
        ]
        new_axes = [axis_names[idx] for idx in remaining] + [self.time_axis, "feature"]
        axis_metadata = {
            key: value
            for key, value in input_rt.axis_schema.axis_metadata.items()
            if key in new_axes
        }
        axis_metadata["feature"] = AxisMetadata("feature", None, "Flattened RF feature axis")
        return AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        axis_names = list(x.axis_schema.axes)
        time_idx = x.get_axis_index(self.time_axis)
        flatten_indices = [x.get_axis_index(name) for name in self.flatten_axes]
        remaining = [
            idx for idx in range(len(axis_names)) if idx not in flatten_indices and idx != time_idx
        ]

        perm = remaining + [time_idx] + flatten_indices
        data = x.as_tensor().permute(perm)
        if self.value == "magnitude":
            data = data.abs().float()
        elif self.value == "real":
            data = data.real.float()
        elif self.value == "imag":
            data = data.imag.float()
        else:
            raise ValueError(f"Unsupported value projection: {self.value}")

        feature_size = 1
        for idx in flatten_indices:
            feature_size *= int(x.shape[idx])

        if remaining:
            leading_shape = [int(x.shape[idx]) for idx in remaining]
            data = data.reshape(*leading_shape, int(x.shape[time_idx]), feature_size)
        else:
            data = data.reshape(int(x.shape[time_idx]), feature_size)
        data = data.contiguous()

        new_axes = [axis_names[idx] for idx in remaining] + [self.time_axis, "feature"]
        axis_metadata = {
            key: value for key, value in x.axis_schema.axis_metadata.items() if key in new_axes
        }
        axis_metadata["feature"] = AxisMetadata("feature", None, "Flattened RF feature axis")

        metadata = x.metadata.copy()
        for axis_name in self.flatten_axes:
            metadata.coords.pop(axis_name, None)
        metadata.transforms.append(
            TransformRecord(
                name="ToSequence",
                params={
                    "flatten_axes": list(self.flatten_axes),
                    "time_axis": self.time_axis,
                    "value": self.value,
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(data, AxisSchema(tuple(new_axes), axis_metadata=axis_metadata), metadata)


@registered_operator(
    required_axes=[],
    description="Pool variable-size point clouds into fixed-width sequence features.",
)
class PointStatsSequence(BaseTransform):
    """Aggregate a ``(time, point, feature)`` point cloud into ``(time, feature')``.

    Each frame is pooled over the point axis with a fixed set of statistics so RFNet
    can consume MM-Fi mmWave point clouds without pretending they are ADC tensors.
    """

    def __init__(
        self,
        *,
        time_axis: str = "time",
        point_axis: str = "point",
        feature_axis: str = "feature",
        stats: tuple[str, ...] = ("mean", "max", "std"),
    ) -> None:
        super().__init__()
        if not stats:
            raise ValueError("stats must not be empty")
        invalid = set(stats) - {"mean", "max", "std"}
        if invalid:
            raise ValueError(f"Unsupported point-pooling stats: {sorted(invalid)}")
        self.time_axis = time_axis
        self.point_axis = point_axis
        self.feature_axis = feature_axis
        self.stats = stats

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.time_axis, self.point_axis, self.feature_axis])

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract()

    def expected_output_schema(self, input_rt: RadioTensor) -> AxisSchema:
        axis_names = list(input_rt.axis_schema.axes)
        remaining = [
            idx
            for idx in range(len(axis_names))
            if axis_names[idx] not in {self.time_axis, self.point_axis, self.feature_axis}
        ]
        new_axes = [axis_names[idx] for idx in remaining] + [self.time_axis, "feature"]
        axis_metadata = {
            key: value
            for key, value in input_rt.axis_schema.axis_metadata.items()
            if key in new_axes
        }
        axis_metadata["feature"] = AxisMetadata("feature", None, "Pooled point-cloud feature axis")
        return AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        axis_names = list(x.axis_schema.axes)
        time_idx = x.get_axis_index(self.time_axis)
        point_idx = x.get_axis_index(self.point_axis)
        feature_idx = x.get_axis_index(self.feature_axis)
        remaining = [
            idx
            for idx in range(len(axis_names))
            if idx not in {time_idx, point_idx, feature_idx}
        ]

        perm = remaining + [time_idx, point_idx, feature_idx]
        data = x.as_tensor().permute(perm)
        if torch.is_complex(data):
            data = data.real
        data = data.float().contiguous()

        leading_shape = data.shape[:-3]
        time_steps = int(data.shape[-3])
        point_dim = int(data.shape[-2])
        feature_dim = int(data.shape[-1])
        flat = data.reshape(-1, time_steps, point_dim, feature_dim)

        if point_dim == 0:
            pooled_feature_dim = feature_dim * len(self.stats)
            pooled_shape = (*leading_shape, time_steps, pooled_feature_dim)
            if not leading_shape:
                pooled_shape = (time_steps, pooled_feature_dim)
            pooled = torch.zeros(pooled_shape, dtype=data.dtype, device=data.device)
            new_axes = [axis_names[idx] for idx in remaining] + [self.time_axis, "feature"]
            axis_metadata = {
                key: value for key, value in x.axis_schema.axis_metadata.items() if key in new_axes
            }
            axis_metadata["feature"] = AxisMetadata(
                "feature",
                None,
                "Pooled point-cloud feature axis",
            )
            metadata = x.metadata.copy()
            metadata.coords.pop(self.point_axis, None)
            metadata.set_coord("feature", np.arange(pooled_feature_dim), unit="index")
            metadata.transforms.append(
                TransformRecord(
                    name="PointStatsSequence",
                    params={
                        "time_axis": self.time_axis,
                        "point_axis": self.point_axis,
                        "feature_axis": self.feature_axis,
                        "stats": list(self.stats),
                    },
                    timestamp=time.time(),
                )
            )
            return RadioTensor(
                pooled,
                AxisSchema(tuple(new_axes), axis_metadata=axis_metadata),
                metadata,
            )

        raw_point_counts = x.metadata.extra.get("point_counts")
        if raw_point_counts is None:
            counts = torch.full((time_steps,), point_dim, dtype=torch.long, device=flat.device)
        else:
            counts = torch.as_tensor(
                raw_point_counts,
                dtype=torch.long,
                device=flat.device,
            ).reshape(-1)
            if counts.numel() != time_steps:
                raise ValueError(
                    "point_counts length must match the time axis length: "
                    f"{counts.numel()} != {time_steps}"
                )
            counts = counts.clamp(min=0, max=point_dim)

        mask = (
            torch.arange(point_dim, device=flat.device)
            .view(1, 1, point_dim, 1)
            .expand(flat.shape[0], time_steps, point_dim, 1)
        ) < counts.view(1, time_steps, 1, 1)
        valid = mask.expand_as(flat)
        safe_counts = counts.clamp(min=1).view(1, time_steps, 1)

        masked = torch.where(valid, flat, torch.zeros_like(flat))
        mean = masked.sum(dim=2) / safe_counts

        neg_inf = torch.full_like(flat, float("-inf"))
        max_values = torch.where(valid, flat, neg_inf).amax(dim=2)
        max_values = torch.where(
            counts.view(1, time_steps, 1) > 0,
            max_values,
            torch.zeros_like(max_values),
        )

        second_moment = (masked * masked).sum(dim=2) / safe_counts
        variance = (second_moment - (mean * mean)).clamp_min(0.0)
        std = torch.sqrt(variance)

        pooled_parts: list[torch.Tensor] = []
        for stat in self.stats:
            if stat == "mean":
                pooled_parts.append(mean)
            elif stat == "max":
                pooled_parts.append(max_values)
            elif stat == "std":
                pooled_parts.append(std)
        pooled = torch.cat(pooled_parts, dim=-1).contiguous()

        if leading_shape:
            pooled = pooled.reshape(*leading_shape, time_steps, pooled.shape[-1])
        else:
            pooled = pooled.reshape(time_steps, pooled.shape[-1])

        new_axes = [axis_names[idx] for idx in remaining] + [self.time_axis, "feature"]
        axis_metadata = {
            key: value for key, value in x.axis_schema.axis_metadata.items() if key in new_axes
        }
        axis_metadata["feature"] = AxisMetadata("feature", None, "Pooled point-cloud feature axis")

        metadata = x.metadata.copy()
        metadata.coords.pop(self.point_axis, None)
        metadata.set_coord("feature", np.arange(int(pooled.shape[-1])), unit="index")
        metadata.transforms.append(
            TransformRecord(
                name="PointStatsSequence",
                params={
                    "time_axis": self.time_axis,
                    "point_axis": self.point_axis,
                    "feature_axis": self.feature_axis,
                    "stats": list(self.stats),
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(
            pooled,
            AxisSchema(tuple(new_axes), axis_metadata=axis_metadata),
            metadata,
        )
