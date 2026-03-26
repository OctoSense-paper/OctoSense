"""Acoustic FMCW demodulation transforms."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from octosense.core.contracts import AxisContract, MetadataRequirement
from octosense.core.errors import MetadataError
from octosense.io.semantics.metadata import CoordinateAxis, TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


@registered_operator(
    required_axes=["sample"],
    required_meta=["sample_rate", "center_freq", "bandwidth"],
    description="Acoustic FMCW demodulation: mix + FFT for range estimation.",
)
class FMCWDemod(BaseTransform):
    """Demodulate acoustic FMCW signals into a range profile."""

    def __init__(
        self,
        n_fft: int | None = None,
        window: str = "hann",
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.window = window

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=["sample"],
            required_metadata=[
                MetadataRequirement("sample_rate", "physical", required=True),
                MetadataRequirement("center_freq", "physical", required=True),
                MetadataRequirement("bandwidth", "physical", required=True),
            ],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            remove_axes=["sample"],
            add_axes=["range"],
        )

    def expected_output_schema(self, x: RadioTensor) -> AxisSchema:
        old_axes = list(x.axis_schema.axes)
        new_axes = ["range" if ax == "sample" else ax for ax in old_axes]
        axis_metadata = {
            key: value for key, value in x.axis_schema.axis_metadata.items() if key != "sample"
        }
        axis_metadata["range"] = AxisMetadata("range", "m", "Range bins")
        return AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        sample_rate = x.metadata.sample_rate
        center_freq = x.metadata.center_freq
        bandwidth = x.metadata.bandwidth

        if sample_rate is None or sample_rate <= 0:
            raise MetadataError("sample_rate must be positive for FMCWDemod")
        if center_freq <= 0:
            raise MetadataError("center_freq must be positive for FMCWDemod")
        if bandwidth <= 0:
            raise MetadataError("bandwidth must be positive for FMCWDemod")

        sample_dim = x.get_axis_index("sample")
        data = x.as_tensor()
        n_samples = data.shape[sample_dim]
        n_fft = self.n_fft or n_samples

        t = torch.arange(n_samples, device=data.device, dtype=data.dtype) / sample_rate
        chirp_rate = bandwidth / (n_samples / sample_rate)
        phase = 2.0 * math.pi * (center_freq * t + 0.5 * chirp_rate * t**2)
        ref_chirp = torch.exp(1j * phase.to(torch.float32)).to(torch.complex64)

        ref_shape = [1] * data.ndim
        ref_shape[sample_dim] = n_samples
        ref_chirp = ref_chirp.reshape(ref_shape)

        beat_signal = data.to(torch.complex64) * ref_chirp.conj()

        if self.window == "hann":
            window = torch.hann_window(n_samples, device=data.device)
        elif self.window == "hamming":
            window = torch.hamming_window(n_samples, device=data.device)
        else:
            window = torch.ones(n_samples, device=data.device)

        window_shape = [1] * data.ndim
        window_shape[sample_dim] = n_samples
        beat_signal = beat_signal * window.reshape(window_shape)

        range_profile = torch.fft.fft(beat_signal, n=n_fft, dim=sample_dim)
        new_schema = self.expected_output_schema(x)

        speed_of_sound = 343.0
        range_resolution = speed_of_sound / (2 * bandwidth)
        max_range = range_resolution * n_fft

        if is_tracking_meta():
            metadata = x.metadata.copy()
            metadata.coords.pop("sample", None)
            metadata.coords["range"] = CoordinateAxis(
                "range",
                values=np.arange(n_fft) * range_resolution,
                unit="m",
            )
            metadata.extra["range_resolution"] = range_resolution
            metadata.extra["max_range"] = max_range
            metadata.transforms.append(
                TransformRecord(
                    name="FMCWDemod",
                    params={"n_fft": n_fft, "window": self.window},
                    timestamp=time.time(),
                )
            )
        else:
            metadata = x.metadata

        return RadioTensor(range_profile, new_schema, metadata)


__all__ = ["FMCWDemod"]
