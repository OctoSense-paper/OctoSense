"""WiFi STFNet-owned dense DFS projection operators."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import numpy as np
import torch

from octosense.core.contracts import AxisContract
from octosense.io.semantics.metadata import CoordinateAxis, TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.backends.cuda.tensor import (
    AxisMappedTensorRunner,
    gaussian_window,
    spectrogram_tensor_batched,
)
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator

_DENSE_DFS_INPUT_AXES = ("time", "subc", "tx", "rx")
_DENSE_DFS_KERNEL_SOURCE_AXES = ("time_bin", "freq_bin", "subc", "link")


def _as_int_tuple(values: list[object] | tuple[object, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _stft_frame_count(signal_length: int, hop_length: int) -> int:
    if signal_length < 0:
        raise ValueError(f"signal_length must be non-negative, got {signal_length}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    # torch.stft(..., center=True) pads by n_fft // 2 on both sides, yielding 1 + floor(L / hop).
    return 1 + (int(signal_length) // int(hop_length))


def _build_time_bin_ticks(
    *,
    raw_time_bins: int,
    time_step: float,
    target_time_bins: int | None,
) -> np.ndarray:
    ticks = np.arange(int(raw_time_bins), dtype=np.float64) * float(time_step)
    if target_time_bins is None:
        return ticks
    target = int(target_time_bins)
    if target <= 0:
        raise ValueError(f"target_time_bins must be positive when provided, got {target_time_bins}")
    if ticks.shape[0] > target:
        start = (ticks.shape[0] - target) // 2
        stop = start + target
        return ticks[start:stop]
    if ticks.shape[0] < target:
        if ticks.size == 0:
            return np.arange(target, dtype=np.float64) * float(time_step)
        extra = ticks[-1] + float(time_step) * np.arange(
            1,
            target - ticks.shape[0] + 1,
            dtype=np.float64,
        )
        return np.concatenate([ticks, extra], axis=0)
    return ticks


def _dense_dfs_projector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(-1).float()


def _project_metadata_coords(metadata: Any, axes: tuple[str, ...] | list[str]) -> Any:
    projected = metadata.copy()
    keep = set(str(axis_name) for axis_name in axes)
    projected.coords = {
        name: coord
        for name, coord in projected.coords.items()
        if name in keep
    }
    return projected


def _resolve_selected_indices(
    *,
    ppe_layout: Mapping[str, Any],
    axis_name: str,
    expected_size: int,
    total_size: int,
) -> tuple[int, ...]:
    raw_indices = ppe_layout.get(f"{axis_name}_index")
    if raw_indices is None:
        indices = tuple(range(expected_size))
    else:
        if not isinstance(raw_indices, (list, tuple)):
            raise TypeError(f"WiFi STFNet PPE expects {axis_name}_index to be a list or tuple")
        indices = _as_int_tuple(raw_indices)
    if len(indices) != expected_size:
        raise ValueError(
            f"WiFi STFNet PPE expects {axis_name}_index to have {expected_size} entries, "
            f"got {len(indices)}"
        )
    if indices and (min(indices) < 0 or max(indices) >= total_size):
        raise ValueError(
            f"WiFi STFNet PPE {axis_name}_index falls outside the declared lattice: "
            f"{min(indices)}..{max(indices)} vs total_{axis_name}={total_size}"
        )
    return indices


def _build_dense_dfs_ppe_embedding(
    *,
    time_bins: int,
    freq_bins: int,
    subc_bins: int,
    link_bins: int,
    device: torch.device,
    ppe_layout: Mapping[str, Any],
) -> torch.Tensor:
    total_dims = _as_int_tuple(ppe_layout["total_dims"])
    if len(total_dims) != 4:
        raise ValueError(f"WiFi STFNet PPE expects total_dims with 4 entries, got {list(total_dims)}")
    total_link, total_subc, total_freq, total_time = total_dims
    link_index = _resolve_selected_indices(
        ppe_layout=ppe_layout,
        axis_name="link",
        expected_size=link_bins,
        total_size=total_link,
    )
    subcarrier_index = _resolve_selected_indices(
        ppe_layout=ppe_layout,
        axis_name="subcarrier",
        expected_size=subc_bins,
        total_size=total_subc,
    )
    freq_index = _resolve_selected_indices(
        ppe_layout=ppe_layout,
        axis_name="freq",
        expected_size=freq_bins,
        total_size=total_freq,
    )
    time_index = _resolve_selected_indices(
        ppe_layout=ppe_layout,
        axis_name="time",
        expected_size=time_bins,
        total_size=total_time,
    )
    scale = math.pi / float(total_link * total_subc * total_freq * total_time - 1)
    link_axis = torch.tensor(
        link_index,
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 1, link_bins)
    subc_index = torch.tensor(
        subcarrier_index,
        device=device,
        dtype=torch.float32,
    ).view(1, 1, subc_bins, 1)
    freq_axis = torch.tensor(
        freq_index,
        device=device,
        dtype=torch.float32,
    ).view(1, freq_bins, 1, 1)
    time_axis = torch.tensor(
        time_index,
        device=device,
        dtype=torch.float32,
    ).view(time_bins, 1, 1, 1)
    flat_index = (((link_axis * total_subc) + subc_index) * total_freq + freq_axis) * total_time + time_axis
    return flat_index * scale


@registered_operator(
    required_axes=["time", "subc", "tx", "rx"],
    description="Project WiFi CSI into dense DFS amplitude tensors.",
)
class DenseDFSAmplitude(BaseTransform):
    """Project WiFi CSI into dense DFS amplitude tensors."""

    def __init__(
        self,
        *,
        sample_rate: int,
        window_size: int,
        window_step: int,
        nfft: int,
        max_doppler_hz: int,
        target_time_bins: int | None = None,
        metadata_axes_to_keep: tuple[str, ...] | list[str] = ("subc",),
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.window_size = int(window_size)
        self.window_step = int(window_step)
        self.nfft = int(nfft)
        self.max_doppler_hz = int(max_doppler_hz)
        self.target_time_bins = int(target_time_bins) if target_time_bins is not None else None
        self.metadata_axes_to_keep = tuple(str(axis_name) for axis_name in metadata_axes_to_keep)
        self._spectrogram_runner: AxisMappedTensorRunner | None = None
        self._spectrogram_runner_device: tuple[str, int | None] | None = None

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=["time", "subc", "tx", "rx"], dtype_constraint="complex")

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(remove_axes=["time", "tx", "rx"], add_axes=["time_bin", "freq_bin", "link", "component"])

    def expected_output_schema(self, _input_rt: RadioTensor) -> AxisSchema:
        return AxisSchema(
            ("time_bin", "freq_bin", "subc", "link", "component"),
            axis_metadata={
                "time_bin": AxisMetadata("time_bin", "s", "Dense DFS frame index"),
                "freq_bin": AxisMetadata("freq_bin", "Hz", "Dense DFS Doppler bin"),
                "subc": AxisMetadata("subc", "index", "WiFi subcarrier"),
                "link": AxisMetadata("link", "index", "Flattened tx-rx link"),
                "component": AxisMetadata("component", None, "Amplitude-only singleton component"),
            },
        )

    def _runner_device_key(self, device: torch.device) -> tuple[str, int | None]:
        return (device.type, device.index)

    def _build_spectrogram_runner(self, device: torch.device) -> AxisMappedTensorRunner:
        freq_values = torch.fft.fftfreq(self.nfft, d=1.0 / float(self.sample_rate))
        freq_mask = (freq_values >= -self.max_doppler_hz) & (freq_values <= self.max_doppler_hz)
        freq_indices = torch.nonzero(freq_mask, as_tuple=False).flatten().to(device=device, dtype=torch.long)
        window = gaussian_window(
            self.window_size,
            std=float(self.window_size),
            dtype=torch.float32,
            device=device,
        )

        def kernel(batch: torch.Tensor) -> torch.Tensor:
            return spectrogram_tensor_batched(
                batch,
                input_axes=_DENSE_DFS_INPUT_AXES,
                temporal_axis="time",
                spectral_axis="subc",
                link_axes=("rx", "tx"),
                n_fft=self.nfft,
                hop_length=self.window_step,
                win_length=self.window_size,
                window=window,
                freq_indices=freq_indices,
                roll_shift=self.max_doppler_hz,
                target_time_bins=self.target_time_bins,
            )

        return AxisMappedTensorRunner(
            input_axes=_DENSE_DFS_INPUT_AXES,
            source_axes=_DENSE_DFS_KERNEL_SOURCE_AXES,
            kernel=kernel,
            project=_dense_dfs_projector,
        )

    def _get_spectrogram_runner(self, device: torch.device) -> AxisMappedTensorRunner:
        device_key = self._runner_device_key(device)
        if self._spectrogram_runner is None or self._spectrogram_runner_device != device_key:
            self._spectrogram_runner = self._build_spectrogram_runner(device)
            self._spectrogram_runner_device = device_key
        return self._spectrogram_runner

    def _selected_freq_values(self) -> np.ndarray:
        freq_values = torch.fft.fftfreq(self.nfft, d=1.0 / float(self.sample_rate))
        freq_mask = (freq_values >= -self.max_doppler_hz) & (freq_values <= self.max_doppler_hz)
        freq_indices = torch.nonzero(freq_mask, as_tuple=False).flatten()
        return torch.roll(freq_values[freq_indices], shifts=self.max_doppler_hz).cpu().numpy().astype(
            np.float64,
            copy=False,
        )

    def _runtime_backend_name(self, device: torch.device) -> str:
        return f"{device.type}.axis_mapped_spectrogram"

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        data = x.as_tensor().to(torch.complex64)
        time_len, _subc_len, tx_len, rx_len = data.shape
        amplitude = self._get_spectrogram_runner(data.device)(data.unsqueeze(0))[0].contiguous()

        time_step = float(self.window_step) / float(self.sample_rate)
        cropped_ticks = _build_time_bin_ticks(
            raw_time_bins=_stft_frame_count(time_len, self.window_step),
            time_step=time_step,
            target_time_bins=self.target_time_bins,
        )
        selected_freq = self._selected_freq_values()

        metadata = _project_metadata_coords(x.metadata, self.metadata_axes_to_keep)
        metadata.coords["time_bin"] = CoordinateAxis("time_bin", cropped_ticks, unit="s")
        metadata.coords["freq_bin"] = CoordinateAxis("freq_bin", selected_freq, unit="Hz")
        metadata.coords["link"] = CoordinateAxis("link", np.arange(rx_len * tx_len, dtype=np.int64), unit="index")
        metadata.coords["component"] = CoordinateAxis("component", np.arange(1, dtype=np.int64), unit="index")
        metadata.transforms.append(
            TransformRecord(
                name="DenseDFSAmplitude",
                params={
                    "sample_rate": self.sample_rate,
                    "window_size": self.window_size,
                    "window_step": self.window_step,
                    "nfft": self.nfft,
                    "max_doppler_hz": self.max_doppler_hz,
                    "target_time_bins": self.target_time_bins,
                    "metadata_axes_to_keep": list(self.metadata_axes_to_keep),
                    "backend": self._runtime_backend_name(data.device),
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(
            data=amplitude.float(),
            axis_schema=self.expected_output_schema(x),
            metadata=metadata,
        )


@registered_operator(
    required_axes=["time_bin", "freq_bin", "subc", "link", "component"],
    description="Convert dense DFS amplitudes into STFNet complex inputs.",
)
class WiFiDenseDFSComplexProjector(BaseTransform):
    """Convert dense DFS amplitudes into complex STFNet image inputs."""

    def __init__(
        self,
        *,
        apply_ppe: bool = True,
        ppe_layout: Mapping[str, Any] | None = None,
        metadata_axes_to_keep: tuple[str, ...] | list[str] = ("time_bin", "freq_bin", "subc", "link"),
    ) -> None:
        super().__init__()
        self.apply_ppe = bool(apply_ppe)
        self.ppe_layout = dict(ppe_layout or {})
        self.metadata_axes_to_keep = tuple(str(axis_name) for axis_name in metadata_axes_to_keep)

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=["time_bin", "freq_bin", "subc", "link", "component"])

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=["time_bin", "freq_bin", "subc", "link"], dtype_constraint="complex")

    def expected_output_schema(self, _input_rt: RadioTensor) -> AxisSchema:
        return AxisSchema(
            ("time_bin", "freq_bin", "subc", "link"),
            axis_metadata={
                "time_bin": AxisMetadata("time_bin", "s", "Dense DFS frame index"),
                "freq_bin": AxisMetadata("freq_bin", "Hz", "Dense DFS Doppler bin"),
                "subc": AxisMetadata("subc", "index", "WiFi subcarrier"),
                "link": AxisMetadata("link", "index", "Flattened tx-rx link"),
            },
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        tensor = x.as_tensor()
        if tensor.shape[-1] != 1:
            raise ValueError(
                "WiFiDenseDFSComplexProjector expects amplitude-only dense DFS input with component=1, "
                f"got {tuple(tensor.shape)}"
            )
        magnitude = tensor[..., 0].float()
        if self.apply_ppe:
            rotary = _build_dense_dfs_ppe_embedding(
                time_bins=int(magnitude.shape[0]),
                freq_bins=int(magnitude.shape[1]),
                subc_bins=int(magnitude.shape[2]),
                link_bins=int(magnitude.shape[3]),
                device=magnitude.device,
                ppe_layout=self.ppe_layout,
            )
        else:
            rotary = torch.zeros(magnitude.shape, dtype=torch.float32, device=magnitude.device)
        complex_tensor = torch.polar(magnitude, rotary)
        metadata = _project_metadata_coords(x.metadata, self.metadata_axes_to_keep)
        metadata.transforms.append(
            TransformRecord(
                name="WiFiDenseDFSComplexProjector",
                params={
                    "apply_ppe": self.apply_ppe,
                    "metadata_axes_to_keep": list(self.metadata_axes_to_keep),
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(
            data=complex_tensor.contiguous(),
            axis_schema=self.expected_output_schema(x),
            metadata=metadata,
        )

__all__ = [
    "DenseDFSAmplitude",
    "WiFiDenseDFSComplexProjector",
]
