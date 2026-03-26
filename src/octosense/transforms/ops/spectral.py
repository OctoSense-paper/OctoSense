"""Frequency domain transforms (FFT, IFFT)."""

import math
import time
from typing import Literal

import numpy as np
import torch

from octosense.core.contracts import AxisContract, MetadataRequirement
from octosense.core.errors import DimensionError, MetadataError
from octosense.io.semantics.metadata import CoordinateAxis, SignalMetadata, TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


def _fft(
    x: torch.Tensor,
    *,
    dim: int,
    norm: Literal["ortho", "forward", "backward"],
) -> torch.Tensor:
    return torch.fft.fft(x, dim=dim, norm=norm)


def _ifft(
    x: torch.Tensor,
    *,
    dim: int,
    norm: Literal["ortho", "forward", "backward"],
) -> torch.Tensor:
    return torch.fft.ifft(x, dim=dim, norm=norm)


def _validate_sample_rate(metadata: SignalMetadata) -> float:
    """Validate and return sample_rate, raising MetadataError if invalid.

    This validation is always performed regardless of TrackMeta state.
    """
    sample_rate = metadata.sample_rate
    if sample_rate is None:
        raise MetadataError(
            "sample_rate required for FFT frequency coordinate calculation. "
            "Ensure your Reader sets this metadata field."
        )
    if isinstance(sample_rate, float) and math.isnan(sample_rate):
        raise MetadataError(
            "sample_rate is NaN. FFT requires a valid numeric sample_rate. "
            "Fix: Set metadata.sample_rate to a finite positive value."
        )
    return sample_rate


@registered_operator(
    required_axes=[],
    required_meta=["sample_rate"],
    description="FFT along time axis, produces freq axis output.",
)
class FFT(BaseTransform):
    """FFT transform for RadioTensor with time axis.

    Transforms time axis -> freq axis with metadata updates.
    """

    def __init__(
        self,
        axis_name: str = "time",
        norm: Literal["ortho", "forward", "backward"] = "ortho",
    ) -> None:
        super().__init__()
        self.axis_name = axis_name
        self.norm = norm

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
            dtype_constraint="complex",
            required_metadata=[
                MetadataRequirement("sample_rate", "physical", required=True),
            ],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            output_axes=["freq"],
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        # 1. Validate input
        if not x.dtype.is_complex:
            raise DimensionError(f"FFT requires complex dtype, got {x.dtype}")
        self._validate_input(x)

        # 2. Always validate sample_rate (regardless of TrackMeta)
        _validate_sample_rate(x.metadata)

        # 3. Get axis dimension
        axis_dim = x.get_axis_index(self.axis_name)

        # 4. Apply FFT using the internal tensor kernel
        data_fft = _fft(
            x.as_tensor(),
            dim=axis_dim,
            norm=self.norm,
        )

        # 5. Update axis schema
        new_axes = list(x.axis_schema.axes)
        new_axes[axis_dim] = "freq"
        axis_metadata = dict(x.axis_schema.axis_metadata)
        axis_metadata.pop(self.axis_name, None)
        axis_metadata["freq"] = AxisMetadata(
            "freq",
            "Hz",
            "Frequency domain (after FFT)",
        )
        new_schema = AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

        # 6. Update metadata (only when tracking)
        if is_tracking_meta():
            new_metadata = self._update_metadata_for_fft(
                x.metadata,
                axis_dim,
                data_fft.shape[axis_dim],
            )
        else:
            # Skip expensive metadata copy when not tracking
            new_metadata = x.metadata

        # 7. Create output RadioTensor
        return RadioTensor(
            data=data_fft,
            axis_schema=new_schema,
            metadata=new_metadata,
        )

    def _update_metadata_for_fft(
        self,
        metadata: SignalMetadata,
        axis_dim: int,
        fft_size: int,
    ) -> SignalMetadata:
        sample_rate = metadata.sample_rate

        # Use torch.fft.fftfreq to match unshifted FFT output
        freq_coords = torch.fft.fftfreq(fft_size, d=1.0 / sample_rate).numpy()

        # Create new metadata (copy Layer A and C, update Layer B)
        new_metadata = metadata.copy()

        # Remove old axis coordinate
        if self.axis_name in new_metadata.coords:
            del new_metadata.coords[self.axis_name]

        # Update coordinate axes (Layer B)
        new_metadata.coords["freq"] = CoordinateAxis(
            axis_name="freq",
            values=freq_coords,
            unit="Hz",
        )

        # Append transform provenance (Layer C)
        new_metadata.transforms.append(
            TransformRecord(
                name="FFT",
                params={"axis_name": self.axis_name, "norm": self.norm},
                timestamp=time.time(),
            )
        )

        return new_metadata


@registered_operator(
    required_axes=[],
    required_meta=["sample_rate"],
    description="Inverse FFT along freq axis, produces time axis output.",
)
class InverseFFT(BaseTransform):
    """Inverse FFT transform for RadioTensor with freq axis."""

    def __init__(
        self,
        axis_name: str = "freq",
        norm: Literal["ortho", "forward", "backward"] = "ortho",
    ) -> None:
        super().__init__()
        self.axis_name = axis_name
        self.norm = norm

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
            dtype_constraint="complex",
            required_metadata=[
                MetadataRequirement("sample_rate", "physical", required=True),
            ],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            output_axes=["time"],
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        if not x.dtype.is_complex:
            raise DimensionError(f"InverseFFT requires complex dtype, got {x.dtype}")
        self._validate_input(x)

        axis_dim = x.get_axis_index(self.axis_name)

        data_ifft = _ifft(
            x.as_tensor(),
            dim=axis_dim,
            norm=self.norm,
        )

        new_axes = list(x.axis_schema.axes)
        new_axes[axis_dim] = "time"
        axis_metadata = dict(x.axis_schema.axis_metadata)
        axis_metadata.pop(self.axis_name, None)
        axis_metadata["time"] = AxisMetadata(
            "time",
            "s",
            "Temporal samples (after IFFT)",
        )
        new_schema = AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

        if is_tracking_meta():
            new_metadata = self._update_metadata_for_ifft(
                x.metadata,
                axis_dim,
                data_ifft.shape[axis_dim],
            )
        else:
            new_metadata = x.metadata

        return RadioTensor(
            data=data_ifft,
            axis_schema=new_schema,
            metadata=new_metadata,
        )

    def _update_metadata_for_ifft(
        self,
        metadata: SignalMetadata,
        axis_dim: int,
        time_size: int,
    ) -> SignalMetadata:
        sample_rate = metadata.sample_rate
        if sample_rate is None:
            raise MetadataError("sample_rate required for IFFT time coordinate calculation")

        time_coords = np.arange(time_size) / sample_rate

        new_metadata = metadata.copy()

        if "freq" in new_metadata.coords:
            del new_metadata.coords["freq"]

        new_metadata.coords["time"] = CoordinateAxis(
            axis_name="time",
            values=time_coords,
            unit="s",
        )

        new_metadata.transforms.append(
            TransformRecord(
                name="InverseFFT",
                params={"axis_name": self.axis_name, "norm": self.norm},
                timestamp=time.time(),
            )
        )

        return new_metadata


@registered_operator(
    required_axes=[],
    required_meta=["sample_rate"],
    description="STFT along time axis, produces freq and frame axes.",
)
class STFT(BaseTransform):
    """STFT transform for RadioTensor with time axis."""

    def __init__(
        self,
        axis_name: str = "time",
        n_fft: int = 256,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool | None = None,
    ) -> None:
        super().__init__()
        self.axis_name = axis_name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
            required_metadata=[MetadataRequirement("sample_rate", "physical", required=True)],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
            remove_axes=[self.axis_name],
            add_axes=["freq", "frame"],
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        # 1. Early check: onesided=True with complex input is not supported
        if self.onesided is True and x.dtype.is_complex:
            raise DimensionError(
                "STFT with onesided=True is not supported for complex input. "
                "Complex signals require full-spectrum (two-sided) output. "
                "Fix: Use onesided=False or onesided=None (auto-detect)."
            )

        self._validate_input(x)

        # 2. Move time axis to last
        axis_dim = x.get_axis_index(self.axis_name)
        data = x.as_tensor()
        if axis_dim != data.ndim - 1:
            perm = list(range(data.ndim))
            perm.pop(axis_dim)
            perm.append(axis_dim)
            data = data.permute(perm)

        # 3. Build window
        win_length = self.win_length or self.n_fft
        if self.window is None:
            window = torch.hann_window(win_length, device=data.device, dtype=data.real.dtype)
        else:
            window = self.window.to(device=data.device, dtype=data.real.dtype)

        hop_length = self.hop_length if self.hop_length is not None else self.n_fft // 4

        # Determine effective onesided: default to False for complex input and True for real input
        effective_onesided = self.onesided if self.onesided is not None else (not x.dtype.is_complex)

        # 4. Apply STFT
        leading_shape = data.shape[:-1]
        time_len = data.shape[-1]
        data_2d = data.reshape(-1, time_len)
        stft_out = torch.stft(
            data_2d,
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=effective_onesided,
            return_complex=True,
        )
        if leading_shape:
            stft_out = stft_out.reshape(*leading_shape, stft_out.shape[-2], stft_out.shape[-1])
        else:
            stft_out = stft_out.reshape(stft_out.shape[-2], stft_out.shape[-1])
        stft_out = stft_out.contiguous()

        # 5. Update axis schema
        new_axes = [ax for ax in x.axis_schema.axes if ax != self.axis_name]
        new_axes.extend(["freq", "frame"])
        axis_metadata = {
            k: v for k, v in x.axis_schema.axis_metadata.items() if k != self.axis_name
        }
        axis_metadata["freq"] = AxisMetadata("freq", "Hz", "Frequency bins (STFT)")
        axis_metadata["frame"] = AxisMetadata("frame", "s", "Time frames (STFT)")
        new_schema = AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

        # 6. Update metadata coordinates
        new_metadata = x.metadata.copy()
        if self.axis_name in new_metadata.coords:
            del new_metadata.coords[self.axis_name]

        sample_rate = new_metadata.sample_rate
        if sample_rate is None:
            raise MetadataError("sample_rate required for STFT coordinate calculation")

        if effective_onesided:
            freq_coords = torch.fft.rfftfreq(self.n_fft, d=1.0 / sample_rate).numpy()
        else:
            freq_coords = torch.fft.fftfreq(self.n_fft, d=1.0 / sample_rate).numpy()
        new_metadata.coords["freq"] = CoordinateAxis("freq", values=freq_coords, unit="Hz")

        frames = stft_out.shape[-1]
        frame_times = (np.arange(frames) * hop_length) / sample_rate
        new_metadata.coords["frame"] = CoordinateAxis("frame", values=frame_times, unit="s")

        new_metadata.transforms.append(
            TransformRecord(
                name="STFT",
                params={
                    "axis_name": self.axis_name,
                    "n_fft": self.n_fft,
                    "hop_length": hop_length,
                    "win_length": win_length,
                    "center": self.center,
                    "pad_mode": self.pad_mode,
                    "normalized": self.normalized,
                    "onesided": effective_onesided,
                },
                timestamp=time.time(),
            )
        )

        return RadioTensor(data=stft_out, axis_schema=new_schema, metadata=new_metadata)


__all__ = ["FFT", "InverseFFT", "STFT"]
