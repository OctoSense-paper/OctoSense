"""XRFV2 WiFi reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from octosense.io.profiles.wifi import (
    build_centered_wifi_subcarrier_indices,
    resolve_wifi_subcarrier_spacing_hz,
)
from octosense.io.readers.wifi.base import BaseWiFiReader, ReaderError
from octosense.io.tensor import RadioTensor, SignalMetadata, build_reader_axis_schema


def _canonicalize_wifi_clip(sample: np.ndarray) -> np.ndarray:
    array = np.asarray(sample, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if array.ndim == 0:
        raise ReaderError("XRFV2 WiFi clip cannot be scalar")
    if array.ndim == 2 and array.shape[1] == 270:
        return array.reshape(int(array.shape[0]), 3, 3, 30).transpose(0, 3, 2, 1)
    if array.ndim == 4 and array.shape[1:] == (3, 3, 30):
        return array.transpose(0, 3, 2, 1)
    if array.ndim == 4 and array.shape[1:] == (30, 3, 3):
        return array
    raise ReaderError(f"Unsupported XRFV2 WiFi clip shape: {array.shape}")


class XRFV2Reader(BaseWiFiReader):
    """Reader for XRFV2 WiFi clips stored in HDF5 files."""

    modality = "wifi"
    device_family = "xrfv2"
    device_name = "Wi-Fi"
    reader_version = "1.0"

    def __init__(self) -> None:
        super().__init__()
        config = self.reader_definition_bundle.config
        self._file_extensions = tuple(
            str(ext).lower() for ext in config.get("file_extensions", (".h5", ".hdf5"))
        )

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"
        if path.suffix.lower() not in self._file_extensions:
            return False, (
                f"Invalid file extension: {path.suffix}. "
                f"XRFV2Reader expects {', '.join(self._file_extensions)} files."
            )
        return True, ""

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        return [self.read(file_path)]

    def read(
        self,
        file_path: str | Path,
        *,
        dataset_key: str | None = None,
        base_index: int | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> RadioTensor:
        path = Path(file_path)
        is_valid, msg = self.validate_format(path)
        if not is_valid:
            raise ReaderError(msg)

        raw_array, resolved_key, source_layout = self._load_clip_array(
            path,
            dataset_key=dataset_key,
            base_index=base_index,
            start=start,
            end=end,
        )
        canonical = _canonicalize_wifi_clip(raw_array)
        schema = build_reader_axis_schema(self.reader_definition_bundle)
        data = torch.from_numpy(canonical).float().contiguous()
        metadata = self._build_metadata(
            path,
            canonical,
            dataset_key=resolved_key,
            source_layout=source_layout,
            base_index=base_index,
            start=start,
            end=end if end is not None else int(start + canonical.shape[0]),
        )
        return RadioTensor.from_reader(data, schema, metadata=metadata)

    def _load_clip_array(
        self,
        path: Path,
        *,
        dataset_key: str | None,
        base_index: int | None,
        start: int,
        end: int | None,
    ) -> tuple[np.ndarray, str, str]:
        preferred_keys = [dataset_key] if dataset_key else ["wifi", "amp"]
        with h5py.File(path, "r") as handle:
            resolved_key = next((key for key in preferred_keys if key and key in handle), None)
            if resolved_key is None:
                known = ", ".join(sorted(handle.keys()))
                raise ReaderError(
                    f"Unable to resolve XRFV2 payload dataset in {path}. Known keys: {known}"
                )
            dataset = handle[resolved_key]
            layout = "processed" if resolved_key == "wifi" else "raw"
            if base_index is None:
                base = np.asarray(dataset)
            else:
                base = np.asarray(dataset[int(base_index)])
        clip_end = None if end is None else int(end)
        return np.asarray(base[int(start) : clip_end]), resolved_key, layout

    def _build_metadata(
        self,
        path: Path,
        clip: np.ndarray,
        *,
        dataset_key: str,
        source_layout: str,
        base_index: int | None,
        start: int,
        end: int,
    ) -> SignalMetadata:
        num_time, num_subc, num_tx, num_rx = (int(dim) for dim in clip.shape)
        subcarrier_indices = build_centered_wifi_subcarrier_indices(num_subc)
        metadata = SignalMetadata(
            modality="wifi",
            subcarrier_spacing=resolve_wifi_subcarrier_spacing_hz(),
            subcarrier_indices=subcarrier_indices.tolist(),
            reader_id=self.reader_id,
            capture_device=self.device_name,
            extra={
                "sample_path": str(path),
                "source_layout": source_layout,
                "source_key": dataset_key,
                "base_index": base_index,
                "clip_start": int(start),
                "clip_end": int(end),
                "raw_shape": [int(dim) for dim in clip.shape],
                "raw_dtype": str(clip.dtype),
            },
        )
        metadata.set_coord("time", np.arange(num_time, dtype=np.float64), unit="frame")
        metadata.set_coord("subc", subcarrier_indices, unit="index")
        metadata.set_coord("tx", np.arange(num_tx, dtype=np.float64), unit="index")
        metadata.set_coord("rx", np.arange(num_rx, dtype=np.float64), unit="index")
        self._finalize_runtime_contract(
            metadata,
            raw_payload={
                "dimension_names": ("time", "subc", "tx", "rx"),
                "data_format": "float32[time,subc,tx,rx]",
                "num_subc": num_subc,
                "num_tx": num_tx,
                "num_rx": num_rx,
            },
        )
        return metadata


__all__ = ["XRFV2Reader"]
