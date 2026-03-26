"""HuPR radar-map reader that materializes one runtime sample tensor."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from octosense.io.readers.mmwave.base import BaseRadarReader, ReaderError
from octosense.io.semantics.metadata import SignalMetadata
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor

_AXIS_SCHEMA = AxisSchema(("range", "azimuth", "elevation", "sensor"))


class HUPRRadarMapReader(BaseRadarReader):
    """Canonical runtime reader for HuPR radar-map pickle bundles."""

    modality = "mmwave"
    device_family = "hupr_radar_map"
    device_name = "AWR1843"
    reader_version = "1.0"

    def read_file(
        self,
        file_path: str | Path,
        config: object | None = None,
    ) -> RadioTensor:
        raise ReaderError(
            "HUPRRadarMapReader requires sample-level read(..., sample_index=...) calls; "
            "file-level read_file() is not supported."
        )

    def validate_format(
        self,
        file_path: str | Path,
        config: object | None = None,
    ) -> tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"
        if path.suffix.lower() != ".pkl":
            return False, f"HuPR radar-map reader expects .pkl files, got {path.suffix or '<none>'}"
        return True, ""

    def read(
        self,
        file_path: str | Path,
        *,
        sample_index: int,
        horizontal_key: str = "hori",
        vertical_key: str = "vert",
        signal_modality: str = "mmwave",
        capture_device: str | None = None,
        center_freq_hz: float = 0.0,
        bandwidth_hz: float = 0.0,
        coord_units: Mapping[str, str] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> RadioTensor:
        """Read one HuPR sample from the pickle bundle into a canonical RadioTensor."""
        path = Path(file_path)
        is_valid, message = self.validate_format(path)
        if not is_valid:
            raise ReaderError(message)

        payload = self._load_payload(path)
        stacked_signal = self._load_signal_sample(
            payload,
            sample_index=sample_index,
            horizontal_key=horizontal_key,
            vertical_key=vertical_key,
            file_path=path,
        )
        tensor = torch.from_numpy(stacked_signal).to(torch.complex64).contiguous()
        metadata = self._build_metadata(
            tensor_shape=tuple(tensor.shape),
            signal_modality=signal_modality,
            capture_device=capture_device or self.device_name,
            center_freq_hz=center_freq_hz,
            bandwidth_hz=bandwidth_hz,
            coord_units=coord_units,
            extra_metadata=extra_metadata,
        )
        return RadioTensor.from_reader(tensor, _AXIS_SCHEMA, metadata)

    def _load_payload(self, file_path: Path) -> dict[str, Any]:
        try:
            with open(file_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            raise ReaderError(f"Failed to load HuPR pickle bundle {file_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReaderError(f"HuPR pickle payload must be a mapping, got {type(payload)!r}")
        return payload

    def _load_signal_sample(
        self,
        payload: Mapping[str, Any],
        *,
        sample_index: int,
        horizontal_key: str,
        vertical_key: str,
        file_path: Path,
    ) -> np.ndarray:
        horizontal_series = self._require_series(payload, horizontal_key, file_path=file_path)
        vertical_series = self._require_series(payload, vertical_key, file_path=file_path)
        if len(horizontal_series) != len(vertical_series):
            raise ReaderError(
                "HuPR horizontal/vertical radar series length mismatch",
                context={
                    "file_path": str(file_path),
                    "horizontal_key": horizontal_key,
                    "vertical_key": vertical_key,
                    "horizontal_count": len(horizontal_series),
                    "vertical_count": len(vertical_series),
                },
            )
        if sample_index < 0 or sample_index >= len(horizontal_series):
            raise ReaderError(
                "HuPR sample_index out of range",
                context={
                    "file_path": str(file_path),
                    "sample_index": sample_index,
                    "sample_count": len(horizontal_series),
                },
            )
        horizontal = np.asarray(horizontal_series[sample_index])
        vertical = np.asarray(vertical_series[sample_index])
        if horizontal.shape != vertical.shape:
            raise ReaderError(
                "HuPR horizontal/vertical radar sample shape mismatch",
                context={
                    "file_path": str(file_path),
                    "sample_index": sample_index,
                    "horizontal_shape": tuple(horizontal.shape),
                    "vertical_shape": tuple(vertical.shape),
                },
            )
        return np.stack([horizontal, vertical], axis=-1)

    def _require_series(
        self,
        payload: Mapping[str, Any],
        key: str,
        *,
        file_path: Path,
    ) -> list[Any]:
        if key not in payload:
            raise ReaderError(
                f"HuPR pickle payload is missing key {key!r}",
                context={"file_path": str(file_path), "available_keys": sorted(str(name) for name in payload)},
            )
        series = payload[key]
        if not isinstance(series, list):
            raise ReaderError(
                f"HuPR payload field {key!r} must be a list, got {type(series)!r}",
                context={"file_path": str(file_path)},
            )
        return series

    def _build_metadata(
        self,
        *,
        tensor_shape: tuple[int, ...],
        signal_modality: str,
        capture_device: str,
        center_freq_hz: float,
        bandwidth_hz: float,
        coord_units: Mapping[str, str] | None,
        extra_metadata: Mapping[str, Any] | None,
    ) -> SignalMetadata:
        units = dict(coord_units or {})
        metadata = SignalMetadata(
            modality=signal_modality,
            center_freq=center_freq_hz,
            bandwidth=bandwidth_hz,
            reader_id=self.reader_id,
            capture_device=capture_device,
            extra=dict(extra_metadata or {}),
        )
        metadata.set_coord("range", np.arange(tensor_shape[0]), unit=units.get("range", "bin"))
        metadata.set_coord("azimuth", np.arange(tensor_shape[1]), unit=units.get("azimuth", "bin"))
        metadata.set_coord("elevation", np.arange(tensor_shape[2]), unit=units.get("elevation", "bin"))
        metadata.set_coord("sensor", np.arange(tensor_shape[3]), unit=units.get("sensor", "index"))
        return metadata
