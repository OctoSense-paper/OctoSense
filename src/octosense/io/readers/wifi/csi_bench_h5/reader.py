"""Sample-level reader for CSI-Bench H5 payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from octosense.io.readers.wifi.base import (
    BaseWiFiReader,
    ReaderError,
    _compact_metadata_kwargs,
    _optional_float,
)
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor, SignalMetadata


def load_csi_bench_h5_payload(path: str | Path) -> tuple[str, np.ndarray]:
    """Load the first declared CSI payload from one CSI-Bench H5 sample."""

    with h5py.File(Path(path), "r") as handle:
        if "CSI_amps" in handle and isinstance(handle["CSI_amps"], h5py.Dataset):
            return "CSI_amps", np.asarray(handle["CSI_amps"])
        for key in handle.keys():
            value = handle[key]
            if isinstance(value, h5py.Dataset):
                return key, np.asarray(value)
    raise ValueError(f"Could not find any dataset payload inside H5 file: {path}")


def _axes_for_payload(array: np.ndarray) -> tuple[str, ...]:
    if array.ndim == 1:
        return ("time",)
    if array.ndim == 2:
        return ("time", "feature")
    if array.ndim == 3:
        return ("time", "feature", "channel")
    return tuple(f"axis_{index}" for index in range(array.ndim))


class CSIBenchH5Reader(BaseWiFiReader):
    """Reader that materializes one CSI-Bench H5 sample into a RadioTensor."""

    modality = "wifi"
    device_family = "csi_bench_h5"
    device_name = "WiFi Device"
    reader_version = "1.0"

    def __init__(self) -> None:
        super().__init__()
        config = self.reader_definition_bundle.config
        self._file_extensions = tuple(str(ext) for ext in config.get("file_extensions", (".h5", ".hdf5")))
        coord_units = dict(config.get("coord_units", {}))
        self._coord_units = {str(axis): str(unit) for axis, unit in coord_units.items()}

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"
        if path.suffix.lower() not in self._file_extensions:
            return False, (
                f"Invalid file extension: {path.suffix}. "
                f"CSIBenchH5Reader expects {', '.join(self._file_extensions)} files."
            )
        try:
            load_csi_bench_h5_payload(path)
        except Exception as exc:
            return False, f"Invalid CSI-Bench H5 payload: {exc}"
        return True, ""

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        raise ReaderError(
            "CSIBenchH5Reader is sample-oriented. Use read(file_path, ...).",
            context={"file_path": str(file_path)},
        )

    def read(
        self,
        file_path: str | Path,
        *,
        capture_device: str | None = None,
        center_freq_hz: float | None = None,
        bandwidth_hz: float | None = None,
        sample_rate_hz: float | None = None,
        sample_id: str | None = None,
        label_name: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RadioTensor:
        path = Path(file_path)
        is_valid, message = self.validate_format(path)
        if not is_valid:
            raise ReaderError(message)

        data_key, payload = load_csi_bench_h5_payload(path)
        tensor = torch.from_numpy(np.asarray(payload))
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        elif tensor.dtype == torch.complex128:
            tensor = tensor.to(torch.complex64)
        tensor = tensor.contiguous()

        axes = _axes_for_payload(np.asarray(payload))
        resolved_capture_device = capture_device or self.device_name
        center_freq = _optional_float(center_freq_hz)
        bandwidth = _optional_float(bandwidth_hz)
        sample_rate = _optional_float(sample_rate_hz)
        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality=self.modality,
                center_freq=center_freq,
                bandwidth=bandwidth,
                sample_rate=sample_rate,
                reader_id=self.reader_id,
                capture_device=resolved_capture_device,
                extra=dict(extra_metadata or {}),
            )
        )
        metadata.extra.setdefault("h5_key", data_key)
        if sample_id is not None:
            metadata.extra["sample_id"] = sample_id
        if label_name is not None:
            metadata.extra["label_name"] = label_name

        for axis_index, axis_name in enumerate(axes[:3]):
            metadata.set_coord(
                axis_name,
                np.arange(int(tensor.shape[axis_index]), dtype=np.float64),
                self._coord_units.get(axis_name, "index"),
            )

        binding_payload: dict[str, Any] = {
            "dimension_names": axes,
            "data_format": f"{str(tensor.dtype).replace('torch.', '')}[{','.join(axes)}]",
        }
        binding_payload.update(
            _compact_metadata_kwargs(
                bandwidth_hz=bandwidth,
                center_freq_hz=center_freq,
                sample_rate_hz=sample_rate,
            )
        )
        self._finalize_runtime_contract(metadata, raw_payload=binding_payload)
        return RadioTensor.from_reader(tensor, AxisSchema(axes), metadata=metadata)


__all__ = ["CSIBenchH5Reader", "load_csi_bench_h5_payload"]
