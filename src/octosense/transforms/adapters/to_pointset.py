"""Canonical point-set adapter and point-set container types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.registry import registered_runtime_operator

_BATCH_LIKE_AXES = {"batch", "sample"}

__all__ = ["FeatureField", "FeatureSchema", "BatchInfo", "PointSet", "ToPointSet"]


@dataclass(frozen=True)
class FeatureField:
    """Schema for a single point-set feature column."""

    name: str
    unit: str = ""
    coord_system: str = "sensor"
    dtype: str = "float32"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureField name cannot be empty")


@dataclass(frozen=True)
class FeatureSchema:
    """Schema describing point-set feature columns."""

    fields: list[FeatureField] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate feature names: {duplicates}")

    def get_field(self, name: str) -> FeatureField | None:
        for feature in self.fields:
            if feature.name == name:
                return feature
        return None

    def has_field(self, name: str) -> bool:
        return self.get_field(name) is not None

    def __len__(self) -> int:
        return len(self.fields)


@dataclass
class BatchInfo:
    """Batch metadata for variable-length point sets."""

    strategy: str = "offsets_lengths"
    offsets: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    mask: torch.Tensor | None = None
    max_points: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("offsets_lengths", "pad_mask"):
            raise ValueError(
                f"Invalid strategy '{self.strategy}', must be 'offsets_lengths' or 'pad_mask'"
            )


class PointSet:
    """Variable-length point-set container owned by the point-set adapter."""

    def __init__(
        self,
        data: torch.Tensor,
        feature_schema: FeatureSchema,
        variable_dim: str = "point",
        coordinate_frames: dict[str, Any] | None = None,
        batch_info: BatchInfo | None = None,
    ) -> None:
        if data.ndim not in (2, 3):
            raise ValueError(
                "PointSet data must be 2D (points, features) or "
                f"3D (batch, points, features), got shape {data.shape}"
            )
        num_features = data.shape[-1]
        if num_features != len(feature_schema):
            raise ValueError(
                f"Data has {num_features} features but schema defines {len(feature_schema)} features"
            )
        self.data = data
        self.feature_schema = feature_schema
        self.variable_dim = variable_dim
        self.coordinate_frames = coordinate_frames or {}
        self.batch_info = batch_info

    def to_tensor(self) -> torch.Tensor:
        return self.data

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        feature_schema: FeatureSchema,
        variable_dim: str = "point",
    ) -> "PointSet":
        return cls(tensor, feature_schema, variable_dim)

    def batch(self, pointsets: list["PointSet"], strategy: str = "offsets_lengths") -> "PointSet":
        if strategy == "offsets_lengths":
            all_data = [pointset.data for pointset in pointsets]
            batched_data = torch.cat(all_data, dim=0)
            lengths = [pointset.data.shape[0] for pointset in pointsets]
            offsets = [0]
            for length in lengths[:-1]:
                offsets.append(offsets[-1] + length)
            batch_info = BatchInfo(strategy="offsets_lengths", offsets=offsets, lengths=lengths)
            return PointSet(
                batched_data,
                self.feature_schema,
                self.variable_dim,
                self.coordinate_frames,
                batch_info,
            )

        if strategy == "pad_mask":
            max_points = max(pointset.data.shape[0] for pointset in pointsets)
            num_features = len(self.feature_schema)
            batch_size = len(pointsets)
            padded_data = torch.zeros(batch_size, max_points, num_features, dtype=self.data.dtype)
            mask = torch.zeros(batch_size, max_points, dtype=torch.bool)
            for index, pointset in enumerate(pointsets):
                n_points = pointset.data.shape[0]
                padded_data[index, :n_points] = pointset.data
                mask[index, :n_points] = True
            batch_info = BatchInfo(strategy="pad_mask", mask=mask, max_points=max_points)
            return PointSet(
                padded_data,
                self.feature_schema,
                self.variable_dim,
                self.coordinate_frames,
                batch_info,
            )

        raise ValueError(f"Unknown strategy '{strategy}'")

    def unbatch(self) -> list["PointSet"]:
        if self.batch_info is None:
            return [self]

        if self.batch_info.strategy == "offsets_lengths":
            pointsets: list[PointSet] = []
            for offset, length in zip(
                self.batch_info.offsets,
                self.batch_info.lengths,
                strict=True,
            ):
                data_slice = self.data[offset : offset + length]
                pointsets.append(
                    PointSet(
                        data_slice,
                        self.feature_schema,
                        self.variable_dim,
                        self.coordinate_frames,
                        None,
                    )
                )
            return pointsets

        if self.batch_info.strategy == "pad_mask":
            pointsets = []
            for index in range(self.data.shape[0]):
                if self.batch_info.mask is not None:
                    valid_mask = self.batch_info.mask[index]
                    data_slice = self.data[index][valid_mask]
                else:
                    data_slice = self.data[index]
                pointsets.append(
                    PointSet(
                        data_slice,
                        self.feature_schema,
                        self.variable_dim,
                        self.coordinate_frames,
                        None,
                    )
                )
            return pointsets

        raise ValueError(f"Unknown strategy '{self.batch_info.strategy}'")

    def __repr__(self) -> str:
        if self.data.ndim == 2:
            n_points, n_features = self.data.shape
            return (
                f"PointSet(points={n_points}, features={n_features}, "
                f"variable_dim='{self.variable_dim}')"
            )
        batch_size, max_points, n_features = self.data.shape
        return (
            f"PointSet(batch={batch_size}, max_points={max_points}, "
            f"features={n_features}, variable_dim='{self.variable_dim}')"
        )


@registered_runtime_operator(
    description="Convert a semantic dense grid into a sparse point-set terminal payload.",
)
class ToPointSet(nn.Module):
    """Convert a semantic dense grid into a sparse point-set representation."""

    def __init__(
        self,
        threshold: float = 0.3,
        feature_names: list[str] | None = None,
        keep_batch_dim: bool = False,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.feature_names = None if feature_names is None else list(feature_names)
        self.keep_batch_dim = bool(keep_batch_dim)

    def _to_pointset(self, x: RadioTensor) -> PointSet:
        tensor = x.as_tensor()
        values = tensor.abs() if tensor.is_complex() else tensor

        max_val = values.max()
        threshold_abs = self.threshold * max_val
        peak_mask = values > threshold_abs
        peak_indices = torch.nonzero(peak_mask, as_tuple=False)
        peak_values = values[peak_mask].unsqueeze(1)

        coordinate_axes = list(x.axis_schema.axes)
        batch_indices: torch.Tensor | None = None
        coordinate_values = peak_indices
        if (
            not self.keep_batch_dim
            and coordinate_axes
            and coordinate_axes[0] in _BATCH_LIKE_AXES
        ):
            batch_indices = peak_indices[:, 0].clone()
            coordinate_axes = coordinate_axes[1:]
            coordinate_values = peak_indices[:, 1:]

        point_data = torch.cat([coordinate_values.float(), peak_values], dim=1)
        point_dtype = str(point_data.dtype).replace("torch.", "")
        resolved_feature_names = list(self.feature_names or [*coordinate_axes, "value"])

        schema_fields: list[FeatureField] = []
        for index, axis_name in enumerate(coordinate_axes):
            field_name = (
                resolved_feature_names[index]
                if index < len(resolved_feature_names)
                else axis_name
            )
            schema_fields.append(
                FeatureField(
                    name=field_name,
                    unit="index",
                    coord_system="grid",
                    dtype=point_dtype,
                    description=f"Index along {axis_name} axis",
                )
            )

        value_name = (
            resolved_feature_names[len(coordinate_axes)]
            if len(resolved_feature_names) > len(coordinate_axes)
            else "value"
        )
        value_unit = ""
        if x.axis_schema.axes:
            first_axis_meta = x.axis_schema.get_metadata(x.axis_schema.axes[0])
            if first_axis_meta and first_axis_meta.unit:
                value_unit = first_axis_meta.unit
        schema_fields.append(
            FeatureField(
                name=value_name,
                unit=value_unit,
                coord_system="grid",
                dtype=point_dtype,
                description="Peak amplitude value",
            )
        )

        coordinate_frames: dict[str, object] = {
            "source": "peak_detection",
            "input_axes": list(x.axis_schema.axes),
            "output_axes": coordinate_axes,
        }
        if batch_indices is not None:
            coordinate_frames["batch_axis"] = x.axis_schema.axes[0]
            coordinate_frames["batch_indices"] = batch_indices.tolist()
        if is_tracking_meta():
            coordinate_frames["provenance"] = {
                "transform": "ToPointSet",
                "threshold": self.threshold,
                "keep_batch_dim": self.keep_batch_dim,
                "n_peaks": int(peak_indices.shape[0]),
            }

        return PointSet(
            point_data,
            FeatureSchema(fields=schema_fields),
            variable_dim="peak",
            coordinate_frames=coordinate_frames,
        )

    def forward(self, x: RadioTensor) -> PointSet:
        return self._to_pointset(x)

    def __repr__(self) -> str:
        return (
            "ToPointSet("
            f"threshold={self.threshold}, keep_batch_dim={self.keep_batch_dim}"
            ")"
        )
