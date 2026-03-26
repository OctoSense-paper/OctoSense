"""OctoNet WiFi pickle reader."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from octosense.io.profiles.wifi import (
    build_centered_wifi_subcarrier_indices,
    resolve_wifi_subcarrier_spacing_hz,
)
from octosense.io.readers.wifi.base import BaseWiFiReader, ReaderError
from octosense.io.tensor import RadioTensor, SignalMetadata, build_reader_axis_schema


def parse_datetime_text(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    formats = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H.%M.%S.%f",
        "%Y-%m-%d %H.%M.%S",
        "%Y-%m-%d %I.%M.%S.%f %p",
        "%Y-%m-%d %I.%M.%S %p",
        "%Y%m%d%H%M%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def timestamp_to_seconds(value: object) -> float:
    if isinstance(value, str):
        dt = parse_datetime_text(value)
        if dt is not None:
            return float(dt.timestamp())
        return float(value)
    if isinstance(value, datetime):
        return float(value.timestamp())
    if hasattr(value, "timestamp") and callable(value.timestamp):
        return float(value.timestamp())
    return float(value)


class OctonetPickleReader(BaseWiFiReader):
    """Reader for sample-level OctoNet WiFi pickle files."""

    modality = "wifi"
    device_family = "octonet_pickle"
    device_name = "AX200"
    reader_version = "1.0"

    def __init__(self) -> None:
        super().__init__()
        config = self.reader_definition_bundle.config
        self._file_extensions = tuple(str(ext) for ext in config.get("file_extensions", (".pickle", ".pkl")))
        repair_by_subc = config.get("pilot_indices_by_num_subcarriers", {})
        self._pilot_indices_by_num_subc = {
            int(key): tuple(int(index) for index in value)
            for key, value in dict(repair_by_subc).items()
        }

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"
        if path.suffix.lower() not in self._file_extensions:
            return False, (
                f"Invalid file extension: {path.suffix}. "
                f"OctonetPickleReader expects {', '.join(self._file_extensions)} files."
            )
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            self._extract_payload(payload)
            return True, ""
        except Exception as exc:
            return False, f"Invalid OctoNet WiFi pickle payload: {exc}"

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        path = Path(file_path)
        is_valid, msg = self.validate_format(path)
        if not is_valid:
            raise ReaderError(msg)

        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        frames, timestamps, user_id = self._extract_payload(payload)
        if len(timestamps) != int(frames.shape[0]):
            raise ReaderError(
                "OctoNet WiFi payload must provide one timestamp per frame",
                context={
                    "frame_count": int(frames.shape[0]),
                    "timestamp_count": len(timestamps),
                    "file_path": str(path),
                },
            )

        frame_timestamp_seconds = [timestamp_to_seconds(timestamp) for timestamp in timestamps]
        first_timestamp = str(timestamps[0]) if timestamps else ""
        num_subc = int(frames.shape[2])
        num_rx = int(frames.shape[1])
        subcarrier_indices = build_centered_wifi_subcarrier_indices(num_subc)
        pilot_indices = list(self._pilot_indices_by_num_subc.get(num_subc, ()))
        schema = build_reader_axis_schema(self.reader_definition_bundle)
        signals: list[RadioTensor] = []
        binding_common = {
            "dimension_names": tuple(schema.axes),
            "data_format": "complex64[time,subc,tx,rx]",
            "num_subc": num_subc,
            "num_tx": 1,
            "num_rx": num_rx,
        }

        for frame_index, timestamp_seconds in enumerate(frame_timestamp_seconds):
            frame = np.asarray(frames[frame_index], dtype=np.complex64)
            data = (
                torch.from_numpy(frame)
                .to(torch.complex64)
                .permute(1, 0)
                .unsqueeze(0)
                .unsqueeze(2)
                .contiguous()
            )
            metadata = SignalMetadata(
                modality="wifi",
                subcarrier_spacing=resolve_wifi_subcarrier_spacing_hz(),
                timestamp_start=float(timestamp_seconds),
                subcarrier_indices=subcarrier_indices.tolist(),
                reader_id=self.reader_id,
                capture_device=self.device_name,
                extra={
                    "sample_id": path.stem,
                    "sample_path": str(path),
                    "user_id": user_id,
                    "timestamp": first_timestamp,
                    "raw_shape": [int(dim) for dim in frames.shape],
                    "raw_dtype": str(frames.dtype),
                    "pilot_indices": pilot_indices,
                    "pilot_interpolated": False,
                },
            )
            metadata.set_coord("time", np.array([0.0], dtype=np.float64), unit="s")
            metadata.set_coord("subc", subcarrier_indices, unit="index")
            self._finalize_runtime_contract(
                metadata,
                raw_payload=binding_common | {"timestamp": float(timestamp_seconds)},
            )
            signals.append(RadioTensor.from_reader(data, schema, metadata=metadata))
        self._assign_stream_sample_rate(signals)
        return signals

    def _extract_payload(
        self,
        payload: Any,
    ) -> tuple[np.ndarray, list[object], int | None]:
        if not isinstance(payload, dict):
            raise ReaderError(f"Expected pickle payload to be a mapping, got {type(payload)!r}")

        if "data" in payload:
            frames = np.asarray(payload["data"], dtype=np.complex64)
            timestamps = list(payload.get("timestamp", ()))
            user_id = None
        else:
            modality_data = payload.get("modality_data")
            if not isinstance(modality_data, dict):
                raise ReaderError("Missing modality_data in OctoNet WiFi bundle")
            wifi_entries = modality_data.get("wifi")
            if not isinstance(wifi_entries, list) or not wifi_entries:
                raise ReaderError("Missing modality_data['wifi'][0] in OctoNet WiFi bundle")
            wifi_entry = wifi_entries[0]
            if not isinstance(wifi_entry, dict):
                raise ReaderError("WiFi bundle entry must be a mapping")
            frames = np.asarray(wifi_entry["frames"], dtype=np.complex64)
            timestamps = list(wifi_entry.get("timestamps", ()))
            user_id = int(payload["user_id"]) if payload.get("user_id") is not None else None

        if frames.ndim != 3:
            raise ReaderError(f"OctoNet WiFi frames must have shape (time, rx, subc), got {frames.shape}")
        return frames, timestamps, user_id


__all__ = ["OctonetPickleReader", "parse_datetime_text", "timestamp_to_seconds"]
