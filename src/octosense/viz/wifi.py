"""WiFi CSI visualization helpers backed by ``RadioTensor`` metadata."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from octosense.io.profiles.wifi import (
    get_wifi_subcarrier_axis_info,
    get_wifi_subcarrier_indices,
)
from octosense.io.tensor import RadioTensor
from octosense.viz.axes import get_axis_info


def _resolve_axis_index(axis_name: str, size: int, index: int | None) -> int:
    resolved = 0 if index is None else int(index)
    if not 0 <= resolved < int(size):
        raise IndexError(f"Axis {axis_name!r} index {resolved} is out of bounds for size {size}")
    return resolved


def _select_axis(
    array: np.ndarray,
    axis_names: tuple[str, ...],
    axis_name: str,
    index: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    axis_idx = axis_names.index(axis_name)
    selected = np.take(array, indices=int(index), axis=axis_idx)
    new_axes = axis_names[:axis_idx] + axis_names[axis_idx + 1 :]
    return selected, new_axes


def _prepare_csi_array(tensor: RadioTensor) -> tuple[np.ndarray, tuple[str, ...]]:
    data = tensor.as_tensor()
    if data.is_complex():
        data = torch.abs(data)
    return data.detach().cpu().numpy(), tuple(tensor.axis_schema.axes)


def _ensure_axes_present(tensor: RadioTensor, *axis_names: str) -> None:
    missing = [axis_name for axis_name in axis_names if axis_name not in tensor.axis_schema.axes]
    if missing:
        available = ", ".join(tensor.axis_schema.axes)
        raise ValueError(
            f"Tensor is missing required axes {missing}; available axes are [{available}]"
        )


def _format_coord_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _axis_label_or_default(tensor: RadioTensor, axis_name: str) -> tuple[np.ndarray, str]:
    values, label = get_axis_info(tensor, axis_name)
    return np.asarray(values), label


def _default_subcarrier_index(tensor: RadioTensor) -> int:
    subc_values = get_wifi_subcarrier_indices(tensor)
    if subc_values.size == 0:
        return 0
    if np.issubdtype(subc_values.dtype, np.number):
        numeric_values = subc_values.astype(np.float64)
        if np.any(numeric_values < 0.0) and np.any(numeric_values > 0.0):
            return int(np.argmin(np.abs(numeric_values)))
        return int(len(numeric_values) // 2)
    return 0


def plot_csi_amplitude_by_subcarrier(
    tensor: RadioTensor,
    *,
    time_index: int = 0,
    tx_index: int | None = 0,
    rx_index: int | None = 0,
    subc_unit: str = "index",
    ax: Any | None = None,
    label: str | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot CSI amplitude against the subcarrier coordinate for one time slice."""

    _ensure_axes_present(tensor, "time", "subc")
    array, axis_names = _prepare_csi_array(tensor)

    if "tx" in axis_names:
        axis_idx = axis_names.index("tx")
        array, axis_names = _select_axis(
            array,
            axis_names,
            "tx",
            _resolve_axis_index("tx", array.shape[axis_idx], tx_index),
        )
    if "rx" in axis_names:
        axis_idx = axis_names.index("rx")
        array, axis_names = _select_axis(
            array,
            axis_names,
            "rx",
            _resolve_axis_index("rx", array.shape[axis_idx], rx_index),
        )

    time_axis_idx = axis_names.index("time")
    array, axis_names = _select_axis(
        array,
        axis_names,
        "time",
        _resolve_axis_index("time", array.shape[time_axis_idx], time_index),
    )

    if axis_names != ("subc",):
        raise ValueError(
            "plot_csi_amplitude_by_subcarrier expected a single remaining 'subc' axis after "
            f"selection, got {axis_names}"
        )

    subc_values, subc_label = get_wifi_subcarrier_axis_info(tensor, unit=subc_unit)
    time_values, time_label = _axis_label_or_default(tensor, "time")
    selected_time_value = time_values[_resolve_axis_index("time", len(time_values), time_index)]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.plot(subc_values, array, linewidth=2.0, label=label)
    ax.set_xlabel(subc_label)
    ax.set_ylabel("CSI amplitude")
    ax.set_title(
        title
        or f"CSI amplitude at {time_label}={_format_coord_value(selected_time_value)}"
    )
    if label is not None:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_csi_amplitude_over_time(
    tensor: RadioTensor,
    *,
    subc_index: int | None = None,
    tx_index: int | None = 0,
    rx_index: int | None = 0,
    subc_unit: str = "index",
    ax: Any | None = None,
    label: str | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Plot CSI amplitude against the time coordinate for one subcarrier trace."""

    _ensure_axes_present(tensor, "time")
    array, axis_names = _prepare_csi_array(tensor)

    selected_subc_index = subc_index
    if "subc" in axis_names:
        axis_idx = axis_names.index("subc")
        if selected_subc_index is None:
            selected_subc_index = _default_subcarrier_index(tensor)
        array, axis_names = _select_axis(
            array,
            axis_names,
            "subc",
            _resolve_axis_index("subc", array.shape[axis_idx], selected_subc_index),
        )

    if "tx" in axis_names:
        axis_idx = axis_names.index("tx")
        array, axis_names = _select_axis(
            array,
            axis_names,
            "tx",
            _resolve_axis_index("tx", array.shape[axis_idx], tx_index),
        )

    if "rx" in axis_names:
        axis_idx = axis_names.index("rx")
        array, axis_names = _select_axis(
            array,
            axis_names,
            "rx",
            _resolve_axis_index("rx", array.shape[axis_idx], rx_index),
        )

    if axis_names != ("time",):
        raise ValueError(
            "plot_csi_amplitude_over_time expected a single remaining 'time' axis after "
            f"selection, got {axis_names}"
        )

    time_values, time_label = _axis_label_or_default(tensor, "time")

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.plot(time_values, array, linewidth=2.0, label=label)
    ax.set_xlabel(time_label)
    ax.set_ylabel("CSI amplitude")
    if title is None and "subc" in tensor.axis_schema.axes:
        subc_values, _ = get_wifi_subcarrier_axis_info(tensor, unit=subc_unit)
        selected_value = subc_values[
            _resolve_axis_index("subc", len(subc_values), selected_subc_index)
        ]
        title = f"CSI amplitude over time at subc={_format_coord_value(selected_value)}"
    ax.set_title(title or "CSI amplitude over time")
    if label is not None:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_csi_stft_spectrogram(
    tensor: RadioTensor,
    *,
    subc_index: int | None = None,
    tx_index: int | None = 0,
    rx_index: int | None = 0,
    subc_unit: str = "index",
    freq_limit_hz: float | None = None,
    fftshift: bool = True,
    cmap: str = "magma",
    ax: Any | None = None,
    title: str | None = None,
) -> tuple[Any, Any, Any]:
    """Plot one STFT spectrogram using metadata-backed frame/frequency axes."""

    _ensure_axes_present(tensor, "freq", "frame")
    array, axis_names = _prepare_csi_array(tensor)

    selected_subc_index = subc_index
    if "subc" in axis_names:
        axis_idx = axis_names.index("subc")
        if selected_subc_index is None:
            selected_subc_index = _default_subcarrier_index(tensor)
        array, axis_names = _select_axis(
            array,
            axis_names,
            "subc",
            _resolve_axis_index("subc", array.shape[axis_idx], selected_subc_index),
        )

    if "tx" in axis_names:
        axis_idx = axis_names.index("tx")
        array, axis_names = _select_axis(
            array,
            axis_names,
            "tx",
            _resolve_axis_index("tx", array.shape[axis_idx], tx_index),
        )

    if "rx" in axis_names:
        axis_idx = axis_names.index("rx")
        array, axis_names = _select_axis(
            array,
            axis_names,
            "rx",
            _resolve_axis_index("rx", array.shape[axis_idx], rx_index),
        )

    if set(axis_names) != {"freq", "frame"}:
        raise ValueError(
            "plot_csi_stft_spectrogram expected only 'freq' and 'frame' axes after selection, "
            f"got {axis_names}"
        )

    freq_axis_idx = axis_names.index("freq")
    frame_axis_idx = axis_names.index("frame")
    plot_array = np.moveaxis(array, (freq_axis_idx, frame_axis_idx), (0, 1))

    freq_values, freq_label = _axis_label_or_default(tensor, "freq")
    frame_values, frame_label = _axis_label_or_default(tensor, "frame")

    if fftshift:
        plot_array = np.fft.fftshift(plot_array, axes=0)
        freq_values = np.fft.fftshift(freq_values)

    if freq_limit_hz is not None:
        limit = float(freq_limit_hz)
        freq_mask = np.abs(freq_values.astype(np.float64)) <= limit
        if not np.any(freq_mask):
            raise ValueError(
                f"freq_limit_hz={freq_limit_hz!r} removed every STFT bin from the plot."
            )
        plot_array = plot_array[freq_mask, :]
        freq_values = freq_values[freq_mask]

    if np.issubdtype(freq_values.dtype, np.number):
        sort_order = np.argsort(freq_values.astype(np.float64))
        plot_array = plot_array[sort_order, :]
        freq_values = freq_values[sort_order]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    extent = None
    if plot_array.shape[0] == len(freq_values) and plot_array.shape[1] == len(frame_values):
        if (
            len(freq_values) > 1
            and len(frame_values) > 1
            and np.issubdtype(freq_values.dtype, np.number)
            and np.issubdtype(frame_values.dtype, np.number)
        ):
            extent = [
                float(frame_values[0]),
                float(frame_values[-1]),
                float(freq_values[0]),
                float(freq_values[-1]),
            ]

    image = ax.imshow(
        plot_array,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=extent,
    )
    ax.set_xlabel(frame_label)
    ax.set_ylabel(freq_label)

    if title is None and "subc" in tensor.axis_schema.axes:
        subc_values, _ = get_wifi_subcarrier_axis_info(tensor, unit=subc_unit)
        selected_value = subc_values[
            _resolve_axis_index("subc", len(subc_values), selected_subc_index)
        ]
        title = f"STFT spectrogram at subc={_format_coord_value(selected_value)}"
    ax.set_title(title or "STFT spectrogram")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig, ax, image


__all__ = [
    "plot_csi_amplitude_by_subcarrier",
    "plot_csi_amplitude_over_time",
    "plot_csi_stft_spectrogram",
]
