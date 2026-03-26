"""Sample-level reader for SignFi MAT archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as scio
import torch

from octosense.io.readers.wifi.base import BaseWiFiReader, ReaderError
from octosense.io.tensor import RadioTensor, SignalMetadata, build_reader_axis_schema


class SignFiMATReader(BaseWiFiReader):
    """Reader that materializes one SignFi sample into a canonical RadioTensor."""

    modality = "wifi"
    device_family = "signfi_mat"
    device_name = "Intel 5300"
    reader_version = "1.0"

    def __init__(self) -> None:
        super().__init__()
        config = self.reader_definition_bundle.config
        self._file_extensions = tuple(str(ext) for ext in config.get("file_extensions", (".mat",)))
        coord_units = dict(config.get("coord_units", {}))
        self._coord_units = {str(axis): str(unit) for axis, unit in coord_units.items()}
        self._schema = build_reader_axis_schema(self.reader_definition_bundle)

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"
        if path.suffix.lower() not in self._file_extensions:
            return False, (
                f"Invalid file extension: {path.suffix}. "
                f"SignFiMATReader expects {', '.join(self._file_extensions)} files."
            )
        try:
            scio.whosmat(path)
        except Exception as exc:
            return False, f"Invalid SignFi MAT payload: {exc}"
        return True, ""

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        raise ReaderError(
            "SignFiMATReader is sample-indexed. Use read(..., data_key=..., sample_index=...).",
            context={"file_path": str(file_path)},
        )

    def read(
        self,
        file_path: str | Path,
        *,
        data_key: str,
        sample_index: int,
        sample_id: str | None = None,
        label_name: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RadioTensor:
        path = Path(file_path)
        is_valid, message = self.validate_format(path)
        if not is_valid:
            raise ReaderError(message)

        payload = scio.loadmat(path, variable_names=[data_key])
        if data_key not in payload:
            raise ReaderError(
                "Requested SignFi MAT key is missing.",
                context={"file_path": str(path), "data_key": data_key},
            )
        raw_tensor = np.asarray(payload[data_key])
        if raw_tensor.ndim != 4:
            raise ReaderError(
                "Expected SignFi tensor with shape (time, subc, rx, sample).",
                context={"file_path": str(path), "data_key": data_key, "shape": tuple(raw_tensor.shape)},
            )
        if sample_index < 0 or sample_index >= int(raw_tensor.shape[-1]):
            raise ReaderError(
                "SignFi sample_index out of range.",
                context={
                    "file_path": str(path),
                    "data_key": data_key,
                    "sample_index": sample_index,
                    "sample_count": int(raw_tensor.shape[-1]),
                },
            )

        frame = np.asarray(raw_tensor[..., sample_index])
        if frame.ndim != 3:
            raise ReaderError(
                "Expected SignFi sample frame with shape (time, subc, rx).",
                context={"file_path": str(path), "data_key": data_key, "sample_index": sample_index},
            )
        tensor = torch.from_numpy(frame)
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        elif tensor.dtype == torch.complex128:
            tensor = tensor.to(torch.complex64)
        tensor = tensor.unsqueeze(2).contiguous()

        metadata = SignalMetadata(
            modality=self.modality,
            reader_id=self.reader_id,
            capture_device=self.device_name,
            extra=dict(extra_metadata or {}),
        )
        metadata.extra.setdefault("source_key", data_key)
        metadata.extra.setdefault("sample_index", int(sample_index))
        if sample_id is not None:
            metadata.extra["sample_id"] = sample_id
        if label_name is not None:
            metadata.extra["label_name"] = label_name
        metadata.set_coord(
            "time",
            np.arange(int(tensor.shape[0]), dtype=np.float64),
            self._coord_units.get("time", "frame"),
        )
        metadata.set_coord(
            "subc",
            np.arange(int(tensor.shape[1]), dtype=np.float64),
            self._coord_units.get("subc", "index"),
        )
        metadata.set_coord(
            "tx",
            np.arange(int(tensor.shape[2]), dtype=np.float64),
            self._coord_units.get("tx", "index"),
        )
        metadata.set_coord(
            "rx",
            np.arange(int(tensor.shape[3]), dtype=np.float64),
            self._coord_units.get("rx", "index"),
        )

        binding_payload = {
            "data": frame,
            "dimension_names": tuple(self._schema.axes),
            "data_format": f"{str(tensor.dtype).replace('torch.', '')}[time,subc,tx,rx]",
            "num_subc": int(tensor.shape[1]),
            "num_tx": int(tensor.shape[2]),
            "num_rx": int(tensor.shape[3]),
        }
        self._finalize_runtime_contract(metadata, raw_payload=binding_payload)
        return RadioTensor.from_reader(tensor, self._schema, metadata=metadata)


__all__ = ["SignFiMATReader"]
