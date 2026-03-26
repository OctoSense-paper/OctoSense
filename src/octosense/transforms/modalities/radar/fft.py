"""Radar FFT transforms: RangeFFT, DopplerFFT, AngleFFT.

Three GPU-native, differentiable nn.Module transforms implementing the
FMCW radar processing chain.  All transforms are modality-agnostic
(axis-driven, no ``if modality`` branches).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from octosense.core.contracts import AxisContract, MetadataRequirement
from octosense.core.errors import DimensionError
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator

# Speed of light (m/s)
_C = 299_792_458.0

# Supported window functions
_WINDOWS = {
    "hann": torch.hann_window,
    "hamming": torch.hamming_window,
    "blackman": torch.blackman_window,
    "none": None,
}


def _get_window(
    name: str, size: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    """Return a 1-D window tensor or None for rectangular."""
    if name == "none":
        return None
    fn = _WINDOWS.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown window '{name}'. Choose from {list(_WINDOWS)}")
    return fn(size, device=device).to(dtype=dtype.to_real() if dtype.is_complex else dtype)


# ---------------------------------------------------------------------------
# RangeFFT
# ---------------------------------------------------------------------------


@registered_operator(
    required_axes=[],
    required_meta=["bandwidth"],
    description="FFT along fast-time (ADC) to compute range profile.",
)
class RangeFFT(BaseTransform):
    """FFT along fast-time dimension to compute range profile.

    Transforms the ``adc`` axis to ``range`` axis.

    Physical relationship::

        range_resolution = c / (2 * bandwidth)       (metres)
        range_bins[k]    = k * range_resolution       (metres)

    Args:
        axis: Input axis name to transform (default ``"adc"``).
        window: Window function (``"hann"``, ``"hamming"``, ``"blackman"``, ``"none"``).
        n_fft: FFT size. ``None`` uses input size.
    """

    def __init__(self, axis: str = "adc", window: str = "hann", n_fft: int | None = None) -> None:
        super().__init__()
        self.axis = axis
        self.window = window
        self.n_fft = n_fft

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis],
            output_axes=["range"],
            dtype_constraint="complex",
            required_metadata=[
                MetadataRequirement("bandwidth", "physical", required=True),
            ],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(output_axes=["range"])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        dim = x.axis_schema.index(self.axis)
        data = x.as_tensor()
        n = self.n_fft or data.shape[dim]

        # Window
        win = _get_window(
            self.window, data.shape[dim], device=data.device, dtype=data.dtype)
        if win is not None:
            shape = [1] * data.ndim
            shape[dim] = -1
            data = data * win.reshape(shape)

        # FFT
        out = torch.fft.fft(data, n=n, dim=dim, norm="ortho")

        # New schema
        new_axes = list(x.axis_schema.axes)
        new_axes[dim] = "range"
        meta_dict = dict(x.axis_schema.axis_metadata)
        meta_dict.pop(self.axis, None)
        meta_dict["range"] = AxisMetadata("range", "m", "Range bins")
        new_schema = AxisSchema(axes=tuple(new_axes), axis_metadata=meta_dict)

        # Metadata update
        track_meta = is_tracking_meta()
        meta = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            bandwidth = meta.bandwidth
            range_res = _C / (2.0 * bandwidth)
            range_coords = np.arange(n, dtype=np.float64) * range_res
            meta.set_coord("range", range_coords, "m")
            meta.extra["range_resolution"] = range_res
            meta.add_transform(
                "RangeFFT", {"axis": self.axis,
                             "window": self.window, "n_fft": self.n_fft}
            )

        return RadioTensor(out, new_schema, meta)


# ---------------------------------------------------------------------------
# DopplerFFT
# ---------------------------------------------------------------------------


@registered_operator(
    required_axes=[],
    required_meta=["chirp_period", "center_freq"],
    description="FFT along slow-time (chirp) to compute Doppler profile.",
)
class DopplerFFT(BaseTransform):
    """FFT along slow-time dimension to compute Doppler profile.

    Transforms the ``chirp`` axis to ``doppler`` axis with ``fftshift``
    to centre zero-velocity.

    Physical relationship::

        wavelength         = c / center_freq
        doppler_resolution = wavelength / (2 * N_chirps * chirp_period)  (m/s)

    Args:
        axis: Input axis name (default ``"chirp"``).
        window: Window function name.
        fft_shift: Apply fftshift to centre zero-velocity (default ``True``).
    """

    def __init__(self, axis: str = "chirp", window: str = "hann", fft_shift: bool = True) -> None:
        super().__init__()
        self.axis = axis
        self.window = window
        self.fft_shift = fft_shift

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis],
            output_axes=["doppler"],
            required_metadata=[
                MetadataRequirement("chirp_period", "physical", required=True),
                MetadataRequirement("center_freq", "physical", required=True),
            ],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(output_axes=["doppler"])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        # Always validate chirp_period regardless of TrackMeta/PerformanceMode
        chirp_period = x.metadata.chirp_period
        if chirp_period is None:
            raise DimensionError(
                "DopplerFFT requires chirp_period in metadata but got None. "
                "Fix: Set metadata.chirp_period to the chirp repetition interval in seconds."
            )
        if isinstance(chirp_period, float) and math.isnan(chirp_period):
            raise DimensionError(
                "DopplerFFT requires valid chirp_period but got NaN. "
                "Fix: Set metadata.chirp_period to a finite positive value."
            )

        dim = x.axis_schema.index(self.axis)
        data = x.as_tensor()
        n = data.shape[dim]

        # Window
        win = _get_window(self.window, n, device=data.device, dtype=data.dtype)
        if win is not None:
            shape = [1] * data.ndim
            shape[dim] = -1
            data = data * win.reshape(shape)

        # FFT + optional shift
        out = torch.fft.fft(data, dim=dim, norm="ortho")
        if self.fft_shift:
            out = torch.fft.fftshift(out, dim=dim)

        # Schema
        new_axes = list(x.axis_schema.axes)
        new_axes[dim] = "doppler"
        meta_dict = dict(x.axis_schema.axis_metadata)
        meta_dict.pop(self.axis, None)
        meta_dict["doppler"] = AxisMetadata(
            "doppler", "m/s", "Doppler velocity bins")
        new_schema = AxisSchema(axes=tuple(new_axes), axis_metadata=meta_dict)

        # Metadata
        track_meta = is_tracking_meta()
        meta = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            wavelength = _C / meta.center_freq
            chirp_period = meta.chirp_period
            doppler_res = wavelength / (2.0 * n * chirp_period)
            doppler_coords = np.arange(-n // 2, n - n //
                                       2, dtype=np.float64) * doppler_res
            meta.set_coord("doppler", doppler_coords, "m/s")
            meta.extra["doppler_resolution"] = doppler_res
            meta.add_transform(
                "DopplerFFT",
                {"axis": self.axis, "window": self.window, "fft_shift": self.fft_shift},
            )

        return RadioTensor(out, new_schema, meta)


# ---------------------------------------------------------------------------
# AngleFFT
# ---------------------------------------------------------------------------


@registered_operator(
    required_axes=[],
    required_meta=[],
    description="FFT-based angle estimation along virtual antenna dimension.",
)
class AngleFFT(BaseTransform):
    """FFT-based angle estimation along virtual antenna dimension.

    Transforms the ``ant`` axis to ``angle`` axis using zero-padded FFT
    (conventional Bartlett beamforming).

    Physical relationship::

        angle_bins = arcsin(linspace(-1, 1, num_angle_bins))   (radians)
        angle_resolution ≈ 2 / num_angle_bins                  (radians)

    Args:
        axis: Input axis name (default ``"ant"``).
        num_angle_bins: FFT size for angle dimension (default 64).
        fft_shift: Apply fftshift to centre broadside (default ``True``).
    """

    def __init__(self, axis: str = "ant", num_angle_bins: int = 64, fft_shift: bool = True) -> None:
        super().__init__()
        self.axis = axis
        self.num_angle_bins = num_angle_bins
        self.fft_shift = fft_shift

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis],
            output_axes=["angle"],
            required_extra_fields=["antenna_positions"],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(output_axes=["angle"])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        # Check antenna_positions in metadata.extra
        if "antenna_positions" not in x.metadata.extra:
            raise DimensionError(
                "AngleFFT requires 'antenna_positions' in metadata.extra.\n"
                "Current extra keys: " +
                str(list(x.metadata.extra.keys())) + "\n"
                "Fix: Ensure the reader sets metadata.extra['antenna_positions'] "
                "to a list of (x, y, z) tuples."
            )

        dim = x.axis_schema.index(self.axis)
        data = x.as_tensor()

        # Zero-padded FFT along antenna dimension
        out = torch.fft.fft(data, n=self.num_angle_bins, dim=dim, norm="ortho")
        if self.fft_shift:
            out = torch.fft.fftshift(out, dim=dim)

        # Schema
        new_axes = list(x.axis_schema.axes)
        new_axes[dim] = "angle"
        meta_dict = dict(x.axis_schema.axis_metadata)
        meta_dict.pop(self.axis, None)
        meta_dict["angle"] = AxisMetadata("angle", "deg", "Angle bins")
        new_schema = AxisSchema(axes=tuple(new_axes), axis_metadata=meta_dict)

        # Metadata
        track_meta = is_tracking_meta()
        meta = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            n = self.num_angle_bins
            angle_res_rad = 2.0 / n
            angle_res_deg = float(np.rad2deg(angle_res_rad))
            # Angle coordinates in degrees via arcsin mapping
            u = np.linspace(-1, 1, n, endpoint=True)
            angle_coords = np.rad2deg(np.arcsin(np.clip(u, -1, 1)))
            meta.set_coord("angle", angle_coords, "deg")
            meta.extra["angle_resolution"] = angle_res_deg
            meta.add_transform(
                "AngleFFT",
                {"axis": self.axis, "num_angle_bins": n, "fft_shift": self.fft_shift},
            )

        return RadioTensor(out, new_schema, meta)

__all__ = ["AngleFFT", "DopplerFFT", "RangeFFT"]
