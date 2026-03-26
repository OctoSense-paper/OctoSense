"""Axis-label helpers for display-only use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from octosense.io.semantics.metadata import CoordinateAxis
    from octosense.io.tensor import RadioTensor


def axis_label_from_metadata(coord: CoordinateAxis | None, axis_name: str) -> str:
    display_name = axis_name.capitalize()
    if coord is None:
        return display_name
    if coord.unit:
        return f"{display_name} ({coord.unit})"
    return display_name


def get_axis_values(tensor: RadioTensor, axis_name: str) -> np.ndarray:
    if axis_name not in tensor.axis_schema.axes:
        available = list(tensor.axis_schema.axes)
        raise ValueError(f"Axis '{axis_name}' not found. Available: {available}")
    coord = tensor.metadata.get_coord(axis_name)
    if coord is not None and coord.values is not None:
        return np.asarray(coord.values)
    axis_idx = tensor.get_axis_index(axis_name)
    return np.arange(tensor.shape[axis_idx])


def get_axis_label(tensor: RadioTensor, axis_name: str) -> str:
    if axis_name not in tensor.axis_schema.axes:
        available = list(tensor.axis_schema.axes)
        raise ValueError(f"Axis '{axis_name}' not found. Available: {available}")
    return axis_label_from_metadata(tensor.metadata.get_coord(axis_name), axis_name)


def get_axis_info(tensor: RadioTensor, axis_name: str) -> tuple[np.ndarray, str]:
    return get_axis_values(tensor, axis_name), get_axis_label(tensor, axis_name)
