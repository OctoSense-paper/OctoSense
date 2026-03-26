"""Parametric RF spectrum estimation transforms."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch

from octosense.core.contracts import AxisContract, MetadataRequirement
from octosense.core.errors import DimensionError, MetadataError
from octosense.io.semantics.metadata import CoordinateAxis, TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator

_C = 299_792_458.0


@registered_operator(
    required_axes=[],
    description="Estimate AoA/ToF pseudo-spectra with MUSIC subspace decomposition.",
)
class MUSICSpectrum(BaseTransform):
    """Estimate 1D MUSIC pseudo-spectra for AoA or ToF."""

    def __init__(
        self,
        *,
        mode: Literal["aoa", "tof"] = "aoa",
        sensor_axis: str | None = None,
        snapshot_axis: str = "time",
        num_sources: int = 1,
        num_scan_bins: int = 181,
        aoa_scan_range_deg: tuple[float, float] = (-90.0, 90.0),
        tof_scan_range_s: tuple[float, float] = (0.0, 200e-9),
    ) -> None:
        super().__init__()
        self.mode = mode
        self.sensor_axis = sensor_axis or ("rx" if mode == "aoa" else "subc")
        self.snapshot_axis = snapshot_axis
        self.num_sources = num_sources
        self.num_scan_bins = num_scan_bins
        self.aoa_scan_range_deg = aoa_scan_range_deg
        self.tof_scan_range_s = tof_scan_range_s

    @property
    def input_contract(self) -> AxisContract:
        required_meta: list[MetadataRequirement] = []
        required_extra_fields: list[str] = []
        if self.mode == "aoa":
            required_meta.append(
                MetadataRequirement("center_freq", "physical", required=True)
            )
            required_extra_fields.append("antenna_positions")
        else:
            required_meta.append(
                MetadataRequirement("subcarrier_spacing", "physical", required=True)
            )
        return AxisContract(
            required_axes=[self.snapshot_axis, self.sensor_axis],
            dtype_constraint="complex",
            required_metadata=required_meta,
            required_extra_fields=required_extra_fields,
        )

    @property
    def output_contract(self) -> AxisContract:
        output_axis = "aoa_bin" if self.mode == "aoa" else "tof_bin"
        return AxisContract(
            remove_axes=[self.snapshot_axis, self.sensor_axis],
            add_axes=[output_axis],
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        if not x.dtype.is_complex:
            raise DimensionError(f"MUSICSpectrum requires complex dtype, got {x.dtype}")
        if self.sensor_axis == self.snapshot_axis:
            raise ValueError("sensor_axis and snapshot_axis must be different")

        sensor_idx = x.get_axis_index(self.sensor_axis)
        snapshot_idx = x.get_axis_index(self.snapshot_axis)
        axis_names = list(x.axis_schema.axes)
        remaining = [
            idx
            for idx in range(len(axis_names))
            if idx not in {sensor_idx, snapshot_idx}
        ]
        perm = remaining + [snapshot_idx, sensor_idx]
        data = x.as_tensor().permute(perm).contiguous()
        snapshots = int(data.shape[-2])
        num_sensors = int(data.shape[-1])
        if snapshots < 2:
            raise DimensionError("MUSICSpectrum requires at least 2 snapshots")
        if self.num_sources >= num_sensors:
            raise DimensionError("num_sources must be smaller than the sensor-axis size")

        flat = data.reshape(-1, snapshots, num_sensors)
        # Each snapshot is a sensor vector in the last dimension. Complex
        # covariance must preserve that orientation: R = X^T X* / N.
        cov = flat.transpose(-1, -2) @ flat.conj() / float(snapshots)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        noise_dim = num_sensors - self.num_sources
        noise_space = eigvecs[..., :noise_dim]

        if self.mode == "aoa":
            scan_values = torch.linspace(
                self.aoa_scan_range_deg[0],
                self.aoa_scan_range_deg[1],
                self.num_scan_bins,
                dtype=torch.float32,
                device=flat.device,
            )
            steering = self._build_aoa_steering(x, num_sensors, scan_values, flat.device)
            output_axis = "aoa_bin"
            output_unit = "deg"
            output_values = scan_values.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            scan_values = torch.linspace(
                self.tof_scan_range_s[0],
                self.tof_scan_range_s[1],
                self.num_scan_bins,
                dtype=torch.float32,
                device=flat.device,
            )
            steering = self._build_tof_steering(x, num_sensors, scan_values, flat.device)
            output_axis = "tof_bin"
            output_unit = "s"
            output_values = scan_values.detach().cpu().numpy().astype(np.float64, copy=False)

        noise_projection = noise_space @ noise_space.conj().transpose(-1, -2)
        a = steering.unsqueeze(0).expand(flat.shape[0], -1, -1)
        numerator = torch.einsum("bfs,bst,bft->bf", a.conj(), noise_projection, a).real
        spectrum = (1.0 / numerator.clamp_min(1e-8)).reshape(
            *data.shape[:-2],
            self.num_scan_bins,
        )

        new_axes = [axis_names[idx] for idx in remaining] + [output_axis]
        axis_metadata = {
            key: value
            for key, value in x.axis_schema.axis_metadata.items()
            if key in new_axes
        }
        axis_metadata[output_axis] = AxisMetadata(
            output_axis,
            output_unit,
            "MUSIC pseudo-spectrum scan bins",
        )

        metadata = x.metadata.copy()
        metadata.coords.pop(self.sensor_axis, None)
        metadata.coords.pop(self.snapshot_axis, None)
        metadata.coords[output_axis] = CoordinateAxis(
            output_axis,
            values=output_values,
            unit=output_unit,
        )
        metadata.transforms.append(
            TransformRecord(
                name="MUSICSpectrum",
                params={
                    "mode": self.mode,
                    "sensor_axis": self.sensor_axis,
                    "snapshot_axis": self.snapshot_axis,
                    "num_sources": self.num_sources,
                    "num_scan_bins": self.num_scan_bins,
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(
            data=spectrum.float(),
            axis_schema=AxisSchema(tuple(new_axes), axis_metadata=axis_metadata),
            metadata=metadata,
        )

    def _build_aoa_steering(
        self,
        x: RadioTensor,
        num_sensors: int,
        scan_values_deg: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        antenna_positions = x.metadata.extra.get("antenna_positions")
        if antenna_positions is None:
            raise MetadataError(
                "MUSICSpectrum(mode='aoa') requires metadata.extra['antenna_positions']"
            )
        positions = torch.as_tensor(antenna_positions, dtype=torch.float32, device=device)
        if positions.shape[0] != num_sensors:
            raise DimensionError(
                "Antenna geometry length must match the selected sensor axis size for MUSIC AoA"
            )
        wavelength = _C / float(x.metadata.center_freq)
        theta = torch.deg2rad(scan_values_deg)
        x_pos = positions[:, 0]
        phase = 2.0 * np.pi / wavelength * torch.sin(theta).unsqueeze(-1) * x_pos.unsqueeze(0)
        return torch.exp(-1j * phase.to(torch.float32)).to(torch.complex64)

    def _build_tof_steering(
        self,
        x: RadioTensor,
        num_sensors: int,
        scan_values_s: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        spacing = x.metadata.subcarrier_spacing
        if spacing is None:
            raise MetadataError(
                "MUSICSpectrum(mode='tof') requires metadata.subcarrier_spacing"
            )
        indices = x.metadata.subcarrier_indices
        if len(indices) != num_sensors:
            raise DimensionError(
                "subcarrier_indices length must match the selected sensor axis size for MUSIC ToF"
            )
        freq_offsets = torch.as_tensor(indices, dtype=torch.float32, device=device) * float(spacing)
        phase = 2.0 * np.pi * scan_values_s.unsqueeze(-1) * freq_offsets.unsqueeze(0)
        return torch.exp(-1j * phase.to(torch.float32)).to(torch.complex64)


__all__ = ["MUSICSpectrum"]
