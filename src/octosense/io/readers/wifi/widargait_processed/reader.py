"""Reader for WidarGait processed CSI numpy tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from octosense.io.profiles.wifi import (
    BASE_WIFI_SUBCARRIER_SPACING_HZ,
    IWL5300_20MHZ_SUBCARRIER_INDICES,
)
from octosense.io.readers.wifi.base import (
    BaseWiFiReader,
    ReaderError,
    _compact_metadata_kwargs,
    _optional_float,
)
from octosense.io.tensor import RadioTensor, SignalMetadata


class WidargaitProcessedReader(BaseWiFiReader):
    """Materialize one canonical sample from WidarGait ``CSI_Processed`` tensors."""

    modality = "wifi"
    device_family = "widargait_processed"
    device_name = "WidarGaitProcessedCSI"
    reader_version = "1.0"

    def __init__(self) -> None:
        super().__init__()
        config = self.reader_definition_bundle.config
        self._file_extensions = tuple(str(ext) for ext in config.get("file_extensions", (".npy",)))

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        try:
            path = self._coerce_input_path(file_path)
            self._load_array(path)
            return True, ""
        except ReaderError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Invalid WidarGait processed tensor: {exc}"

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        return [self.read(file_path)]

    def read(
        self,
        file_path: str | Path,
        *,
        sample_id: str | None = None,
        dataset_name: str | None = None,
        signal_modality: str = "wifi",
        capture_device: str | None = None,
        center_freq_hz: float | None = None,
        bandwidth_hz: float | None = None,
        sample_rate_hz: float | None = None,
    ) -> RadioTensor:
        path = self._coerce_input_path(file_path)
        array = self._load_array(path)
        data = torch.from_numpy(array).to(torch.complex64).permute(0, 1, 3, 2).contiguous()
        schema = self._canonical_wifi_axis_schema()
        sample_rate = _optional_float(sample_rate_hz)
        center_freq = _optional_float(center_freq_hz)
        bandwidth = _optional_float(bandwidth_hz)
        subcarrier_indices = list(IWL5300_20MHZ_SUBCARRIER_INDICES[: data.shape[1]])
        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality=str(signal_modality),
                center_freq=center_freq,
                bandwidth=bandwidth,
                sample_rate=sample_rate,
                subcarrier_indices=subcarrier_indices,
                reader_id=self.reader_id,
                capture_device=str(capture_device or self.device_name),
                extra={
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "sample_path": str(path),
                },
            )
        )
        time_values = np.arange(data.shape[0], dtype=np.float64)
        time_unit = "frame"
        if sample_rate is not None:
            time_values = time_values / sample_rate
            time_unit = "s"
        metadata.set_coord("time", time_values, unit=time_unit)
        metadata.set_coord(
            "subc",
            np.asarray(subcarrier_indices, dtype=np.float64) * BASE_WIFI_SUBCARRIER_SPACING_HZ,
            unit="Hz",
        )
        binding_payload = {
            "dimension_names": tuple(schema.axes),
            "data_format": "complex64[time,subc,tx,rx]",
            "num_subc": int(data.shape[1]),
            "num_tx": int(data.shape[2]),
            "num_rx": int(data.shape[3]),
        }
        binding_payload.update(
            _compact_metadata_kwargs(
                center_freq_hz=center_freq,
                bandwidth_hz=bandwidth,
                sample_rate_hz=sample_rate,
            )
        )
        self._finalize_runtime_contract(metadata, raw_payload=binding_payload)
        return RadioTensor.from_reader(data, schema, metadata=metadata)

    def _coerce_input_path(self, file_path: str | Path) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise ReaderError(f"File not found: {path}")
        if path.suffix.lower() not in self._file_extensions:
            raise ReaderError(
                f"Invalid file extension: {path.suffix}. "
                f"WidargaitProcessedReader expects {', '.join(self._file_extensions)} files."
            )
        return path

    def _load_array(self, file_path: str | Path) -> np.ndarray:
        array = np.asarray(np.load(file_path), dtype=np.complex64)
        if array.ndim != 4:
            raise ReaderError(
                "WidarGait processed tensor must have 4 dimensions "
                "(time, subc, rx, tx)",
                context={"shape": list(array.shape), "file_path": str(file_path)},
            )
        if array.shape[1] > len(IWL5300_20MHZ_SUBCARRIER_INDICES):
            raise ReaderError(
                "WidarGait processed tensor uses more subcarriers than the IWL5300 profile supports",
                context={"num_subc": int(array.shape[1]), "file_path": str(file_path)},
            )
        return array


__all__ = ["WidargaitProcessedReader"]
