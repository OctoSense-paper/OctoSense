"""WiFi spectral operators."""

import time

import numpy as np
import torch
import torch.nn.functional as F

from octosense.core.contracts import AxisContract, MetadataRequirement
from octosense.core.errors import MetadataError
from octosense.io.semantics.metadata import CoordinateAxis, TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


@registered_operator(
    required_axes=[],
    required_meta=["sample_rate"],
    description="Compute temporal autocorrelation spectra over lag.",
)
class AutocorrelationSpectrum(BaseTransform):
    """Compute autocorrelation over a named time axis and emit a lag spectrum."""

    def __init__(
        self,
        axis_name: str = "time",
        *,
        max_lag: int | None = None,
        normalized: bool = True,
        value: str = "real",
    ) -> None:
        super().__init__()
        if value not in {"magnitude", "real", "imag"}:
            raise ValueError(f"Unsupported ACF projection '{value}'")
        self.axis_name = axis_name
        self.max_lag = max_lag
        self.normalized = normalized
        self.value = value

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
            required_metadata=[MetadataRequirement("sample_rate", "physical", required=True)],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(remove_axes=[self.axis_name], add_axes=["lag"])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        if x.metadata.sample_rate is None:
            raise MetadataError("AutocorrelationSpectrum requires metadata.sample_rate")

        axis_idx = x.get_axis_index(self.axis_name)
        data = x.as_tensor()
        if torch.is_complex(data):
            if self.value == "magnitude":
                data = data.abs().float()
            elif self.value == "real":
                data = data.real.float()
            else:
                data = data.imag.float()
        else:
            data = data.float()

        data = data.movedim(axis_idx, -1).contiguous()
        time_len = int(data.shape[-1])
        max_lag = time_len - 1 if self.max_lag is None else min(self.max_lag, time_len - 1)
        fft_len = 1
        while fft_len < 2 * time_len:
            fft_len *= 2

        flat = data.reshape(-1, time_len)
        spectrum = torch.fft.rfft(flat, n=fft_len, dim=-1)
        acf = torch.fft.irfft(spectrum.conj() * spectrum, n=fft_len, dim=-1)[..., : max_lag + 1]
        if self.normalized:
            denom = acf[..., :1].clamp_min(1e-8)
            acf = acf / denom
        acf = acf.reshape(*data.shape[:-1], max_lag + 1).contiguous()

        axis_names = list(x.axis_schema.axes)
        new_axes = [ax for ax in axis_names if ax != self.axis_name] + ["lag"]
        axis_metadata = {
            ax: meta for ax, meta in x.axis_schema.axis_metadata.items() if ax != self.axis_name
        }
        axis_metadata["lag"] = AxisMetadata("lag", "s", "Autocorrelation lag")

        metadata = x.metadata.copy()
        metadata.coords.pop(self.axis_name, None)
        lag_values = np.arange(max_lag + 1, dtype=np.float64) / float(x.metadata.sample_rate)
        metadata.coords["lag"] = CoordinateAxis("lag", values=lag_values, unit="s")
        metadata.transforms.append(
            TransformRecord(
                name="AutocorrelationSpectrum",
                params={
                    "axis_name": self.axis_name,
                    "max_lag": self.max_lag,
                    "normalized": self.normalized,
                    "value": self.value,
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(
            data=acf.float(),
            axis_schema=AxisSchema(tuple(new_axes), axis_metadata=axis_metadata),
            metadata=metadata,
        )


@registered_operator(
    required_axes=[],
    required_meta=["sample_rate"],
    description="Compute a wavelet scalogram over pseudo-frequency and time.",
)
class WaveletScalogram(BaseTransform):
    """Compute a Morlet-style scalogram along a named time axis."""

    def __init__(
        self,
        axis_name: str = "time",
        *,
        num_frequencies: int = 32,
        min_frequency_hz: float = 1.0,
        max_frequency_hz: float | None = None,
        wavelet_cycles: float = 6.0,
        value: str = "real",
    ) -> None:
        super().__init__()
        if value not in {"magnitude", "real", "imag"}:
            raise ValueError(f"Unsupported wavelet projection '{value}'")
        self.axis_name = axis_name
        self.num_frequencies = num_frequencies
        self.min_frequency_hz = min_frequency_hz
        self.max_frequency_hz = max_frequency_hz
        self.wavelet_cycles = wavelet_cycles
        self.value = value

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
            required_metadata=[MetadataRequirement("sample_rate", "physical", required=True)],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(remove_axes=[self.axis_name], add_axes=["pseudo_freq", "frame"])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        sample_rate = x.metadata.sample_rate
        if sample_rate is None:
            raise MetadataError("WaveletScalogram requires metadata.sample_rate")
        nyquist = float(sample_rate) / 2.0
        max_freq = nyquist if self.max_frequency_hz is None else min(self.max_frequency_hz, nyquist)
        if self.min_frequency_hz <= 0 or max_freq <= self.min_frequency_hz:
            raise ValueError("WaveletScalogram requires 0 < min_frequency_hz < max_frequency_hz")

        axis_idx = x.get_axis_index(self.axis_name)
        data = x.as_tensor()
        if torch.is_complex(data):
            if self.value == "magnitude":
                data = data.abs().float()
            elif self.value == "real":
                data = data.real.float()
            else:
                data = data.imag.float()
        else:
            data = data.float()

        data = data.movedim(axis_idx, -1).contiguous()
        time_len = int(data.shape[-1])
        flat = data.reshape(-1, 1, time_len)
        freq_values = np.geomspace(
            float(self.min_frequency_hz),
            float(max_freq),
            self.num_frequencies,
            dtype=np.float64,
        )
        freqs = torch.from_numpy(freq_values).to(device=flat.device, dtype=torch.float32)
        responses: list[torch.Tensor] = []
        for freq in freqs.tolist():
            sigma_t = self.wavelet_cycles / (2.0 * np.pi * float(freq))
            half_width = max(1, int(round(4.0 * sigma_t * float(sample_rate))))
            t = torch.arange(-half_width, half_width + 1, device=flat.device, dtype=torch.float32)
            t = t / float(sample_rate)
            envelope = torch.exp(-0.5 * (t / sigma_t) ** 2)
            kernel_cos = torch.cos(2.0 * np.pi * float(freq) * t) * envelope
            kernel_sin = torch.sin(2.0 * np.pi * float(freq) * t) * envelope
            kernel_cos = F.normalize(kernel_cos, p=1.0, dim=0, eps=1e-8).view(1, 1, -1)
            kernel_sin = F.normalize(kernel_sin, p=1.0, dim=0, eps=1e-8).view(1, 1, -1)
            resp_cos = F.conv1d(flat, kernel_cos, padding=kernel_cos.shape[-1] // 2)
            resp_sin = F.conv1d(flat, kernel_sin, padding=kernel_sin.shape[-1] // 2)
            responses.append(torch.hypot(resp_cos, resp_sin).squeeze(1))

        scalogram = torch.stack(responses, dim=1)
        scalogram = scalogram.reshape(*data.shape[:-1], self.num_frequencies, time_len).contiguous()

        axis_names = list(x.axis_schema.axes)
        new_axes = [ax for ax in axis_names if ax != self.axis_name] + ["pseudo_freq", "frame"]
        axis_metadata = {
            ax: meta for ax, meta in x.axis_schema.axis_metadata.items() if ax != self.axis_name
        }
        axis_metadata["pseudo_freq"] = AxisMetadata(
            "pseudo_freq",
            "Hz",
            "Wavelet pseudo-frequency bins",
        )
        axis_metadata["frame"] = AxisMetadata("frame", "s", "Wavelet time frames")

        metadata = x.metadata.copy()
        metadata.coords.pop(self.axis_name, None)
        metadata.coords["pseudo_freq"] = CoordinateAxis(
            "pseudo_freq",
            values=freq_values,
            unit="Hz",
        )
        time_coord = x.metadata.get_coord(self.axis_name)
        if time_coord is not None and time_coord.values is not None:
            frame_values = np.asarray(time_coord.values, dtype=np.float64)
        else:
            frame_values = np.arange(time_len, dtype=np.float64) / float(sample_rate)
        metadata.coords["frame"] = CoordinateAxis("frame", values=frame_values, unit="s")
        metadata.transforms.append(
            TransformRecord(
                name="WaveletScalogram",
                params={
                    "axis_name": self.axis_name,
                    "num_frequencies": self.num_frequencies,
                    "min_frequency_hz": self.min_frequency_hz,
                    "max_frequency_hz": max_freq,
                    "wavelet_cycles": self.wavelet_cycles,
                    "value": self.value,
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(
            data=scalogram.float(),
            axis_schema=AxisSchema(tuple(new_axes), axis_metadata=axis_metadata),
            metadata=metadata,
        )


__all__ = ["AutocorrelationSpectrum", "WaveletScalogram"]
