"""Acoustic-specific transforms."""

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
from octosense.transforms.ops.spectral import STFT


@registered_operator(
    required_axes=["sample"],
    required_meta=["sample_rate"],
    description="Acoustic waveform STFT along sample axis, produces freq and frame axes.",
)
class AcousticSTFT(STFT):
    """Acoustic-specific STFT wrapper with waveform axis defaults."""

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: torch.Tensor | None = None,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool | None = None,
    ) -> None:
        super().__init__(
            axis_name="sample",
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        out = super().forward(x)
        if out.metadata.transforms:
            out.metadata.transforms[-1].name = "AcousticSTFT"
        return out


@registered_operator(
    required_axes=["sample", "mic"],
    required_meta=["sample_rate"],
    description="Delay-and-sum beamforming for microphone arrays.",
)
class AcousticBeamforming(BaseTransform):
    """Delay-and-sum beamforming for microphone arrays.

    Performs time-domain delay-and-sum beamforming given microphone positions
    and a target steering direction. This is a frequency-domain implementation
    using phase shifts.

    Input: (time, sample, mic) — multi-channel audio
    Output: (time, sample, angle) — steered beamformed output

    The mic axis is replaced by an angle axis representing steering directions.
    """

    def __init__(
        self,
        mic_positions: list[float] | None = None,
        angles_deg: list[float] | None = None,
        mic_spacing: float = 0.01,  # 1 cm default
        speed_of_sound: float = 343.0,
    ) -> None:
        super().__init__()
        # Default: linear array with 2 mics
        self.mic_positions = mic_positions
        self.mic_spacing = mic_spacing
        self.angles_deg = angles_deg or list(range(-90, 91, 5))
        self.speed_of_sound = speed_of_sound

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=["sample", "mic"],
            required_metadata=[
                MetadataRequirement("sample_rate", "physical", required=True),
            ],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            remove_axes=["mic"],
            add_axes=["angle"],
        )

    def expected_output_schema(self, x: RadioTensor) -> AxisSchema:
        old_axes = list(x.axis_schema.axes)
        new_axes = ["angle" if ax == "mic" else ax for ax in old_axes]
        axis_metadata = {
            k: v for k, v in x.axis_schema.axis_metadata.items() if k != "mic"
        }
        axis_metadata["angle"] = AxisMetadata("angle", "deg", "Steering angle")
        return AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)

        sample_rate = x.metadata.sample_rate
        if sample_rate is None or sample_rate <= 0:
            raise MetadataError("sample_rate must be positive for AcousticBeamforming")

        data = x.as_tensor()
        sample_dim = x.get_axis_index("sample")
        mic_dim = x.get_axis_index("mic")
        n_mics = data.shape[mic_dim]
        n_samples = data.shape[sample_dim]

        # Determine mic positions
        if self.mic_positions is not None:
            positions = torch.tensor(self.mic_positions, dtype=torch.float32,
                                     device=data.device)
        else:
            positions = torch.arange(n_mics, dtype=torch.float32,
                                     device=data.device) * self.mic_spacing

        angles_rad = torch.tensor(
            [a * math.pi / 180.0 for a in self.angles_deg],
            dtype=torch.float32,
            device=data.device,
        )

        # FFT along sample axis for frequency-domain beamforming
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / sample_rate, device=data.device)
        data_fft = torch.fft.rfft(data, dim=sample_dim)

        # Build steering vectors: (n_angles, n_mics, n_freqs)
        # delay[a, m] = positions[m] * sin(angle[a]) / speed_of_sound
        delays = torch.outer(
            torch.sin(angles_rad), positions
        ) / self.speed_of_sound  # (n_angles, n_mics)

        # Phase shift: exp(-j * 2*pi * f * delay)
        # steering: (n_angles, n_mics, n_freqs)
        steering = torch.exp(
            -1j * 2 * math.pi * delays.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        ).to(torch.complex64)

        # Move mic and sample dims for einsum
        # data_fft shape: (..., sample_fft, ...) with mic_dim
        # We need to perform: output[..., angle, freq] = sum_mic steering[angle, mic, freq] * data_fft[..., freq, mic]

        # Move data to standard form: put mic last, sample(fft) second-to-last
        perm = list(range(data_fft.ndim))
        # Remove sample and mic dims
        remaining = [i for i in range(data_fft.ndim) if i not in (sample_dim, mic_dim)]
        # Rebuild: remaining + [sample_fft, mic]
        perm = remaining + [sample_dim, mic_dim]
        data_perm = data_fft.permute(perm)  # (..., freq, mic)

        leading_shape = data_perm.shape[:-2]
        n_freq = data_perm.shape[-2]
        data_flat = data_perm.reshape(-1, n_freq, n_mics)  # (batch, freq, mic)

        # Beamform: sum over mic, result: (batch, n_angles, freq)
        data_complex = data_flat.to(torch.complex64)
        # steering: (n_angles, n_mics, n_freq)
        # data_flat: (batch, n_freq, n_mics)
        # output: (batch, n_angles, n_freq)
        output = torch.einsum(
            "amf,bfm->baf", steering, data_complex
        )

        # IFFT back to time domain: (batch, n_angles, sample)
        output_time = torch.fft.irfft(output, n=n_samples, dim=-1)

        # Reshape back to leading dims + (angle, sample)
        output_time = output_time.reshape(*leading_shape, len(self.angles_deg), n_samples)

        # Swap to (..., sample, angle) to match convention
        output_time = output_time.transpose(-2, -1)

        # Build new schema: replace mic with angle, keep order
        old_axes = list(x.axis_schema.axes)
        new_axes = ["angle" if ax == "mic" else ax for ax in old_axes]
        axis_metadata = {
            k: v for k, v in x.axis_schema.axis_metadata.items() if k != "mic"
        }
        axis_metadata["angle"] = AxisMetadata("angle", "deg", "Steering angle")
        new_schema = AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

        if is_tracking_meta():
            new_metadata = x.metadata.copy()
            new_metadata.coords["angle"] = CoordinateAxis(
                "angle",
                values=np.array(self.angles_deg, dtype=np.float32),
                unit="deg",
            )

            new_metadata.transforms.append(
                TransformRecord(
                    name="AcousticBeamforming",
                    params={
                        "mic_spacing": self.mic_spacing,
                        "n_angles": len(self.angles_deg),
                        "speed_of_sound": self.speed_of_sound,
                    },
                    timestamp=time.time(),
                )
            )
        else:
            new_metadata = x.metadata

        return RadioTensor(
            data=output_time, axis_schema=new_schema, metadata=new_metadata
        )


__all__ = ["AcousticSTFT", "AcousticBeamforming"]
