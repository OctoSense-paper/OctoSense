"""XRF55 WiFi reader."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io as scio

from octosense.io.profiles.wifi import (
    build_centered_wifi_subcarrier_indices,
    resolve_wifi_subcarrier_spacing_hz,
)
from octosense.io.readers.wifi.base import BaseWiFiReader, ReaderError
from octosense.io.tensor import RadioTensor, SignalMetadata, build_reader_axis_schema

_WIFI_SUBC_CANDIDATES = {30, 56, 57, 90, 114, 121}
_FLAT_FEATURE_LAYOUTS: dict[int, tuple[int, int, int]] = {
    90: (3, 1, 30),
    270: (3, 3, 30),
}


def _unwrap_object_scalar(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        current = current.item()
    return current


def _scalar_from_any(value: Any, default: int | None = None) -> int | None:
    current = _unwrap_object_scalar(value)
    array = np.asarray(current)
    if array.size == 0:
        return default
    item = array.reshape(-1)[0]
    return int(item.item() if hasattr(item, "item") else item)


def _normalize_rx_permutation(perm: np.ndarray, n_rx: int) -> np.ndarray | None:
    if n_rx <= 1:
        return None
    values = np.asarray(perm).reshape(-1)
    if values.size < n_rx:
        return None
    indices = values[:n_rx].astype(np.int64, copy=False) - 1
    if np.any(indices < 0) or sorted(indices.tolist()) != list(range(n_rx)):
        return None
    return indices


def _canonicalize_csi_packet(
    packet: Any,
    *,
    n_tx: int,
    n_rx: int,
    rx_permutation: np.ndarray | None = None,
) -> np.ndarray:
    array = np.asarray(_unwrap_object_scalar(packet))
    array = np.squeeze(array)
    if array.ndim == 0:
        raise ReaderError("XRF55 CSI packet cannot be scalar")

    canonical: np.ndarray
    if array.ndim == 2:
        if array.shape[1] in _WIFI_SUBC_CANDIDATES:
            spatial, subc = array.shape
            if n_tx > 1 and n_rx > 1 and spatial == n_tx * n_rx:
                canonical = np.transpose(array.reshape(n_tx, n_rx, subc), (2, 0, 1))
            elif spatial == n_rx:
                canonical = np.transpose(array, (1, 0))[:, None, :]
            elif spatial == n_tx:
                canonical = np.transpose(array, (1, 0))[:, :, None]
            else:
                canonical = np.transpose(array, (1, 0))[:, None, :]
        elif array.shape[0] in _WIFI_SUBC_CANDIDATES:
            subc, spatial = array.shape
            if n_tx > 1 and n_rx > 1 and spatial == n_tx * n_rx:
                canonical = array.reshape(subc, n_tx, n_rx)
            elif spatial == n_rx:
                canonical = array[:, None, :]
            elif spatial == n_tx:
                canonical = array[:, :, None]
            else:
                canonical = array[:, None, :]
        else:
            raise ReaderError(f"Unsupported XRF55 CSI packet shape: {array.shape}")
    elif array.ndim == 3:
        if array.shape[2] in _WIFI_SUBC_CANDIDATES:
            if array.shape[0] == n_tx and array.shape[1] == n_rx:
                canonical = np.transpose(array, (2, 0, 1))
            elif array.shape[0] == n_rx and array.shape[1] == n_tx:
                canonical = np.transpose(array, (2, 1, 0))
            else:
                canonical = np.transpose(array, (2, 0, 1))
        elif array.shape[0] in _WIFI_SUBC_CANDIDATES:
            if array.shape[1] == n_tx and array.shape[2] == n_rx:
                canonical = array
            elif array.shape[1] == n_rx and array.shape[2] == n_tx:
                canonical = np.transpose(array, (0, 2, 1))
            else:
                canonical = array
        elif array.shape[1] in _WIFI_SUBC_CANDIDATES:
            if array.shape[0] == n_tx and array.shape[2] == n_rx:
                canonical = np.transpose(array, (1, 0, 2))
            elif array.shape[0] == n_rx and array.shape[2] == n_tx:
                canonical = np.transpose(array, (1, 2, 0))
            else:
                canonical = np.transpose(array, (1, 0, 2))
        else:
            raise ReaderError(f"Unsupported XRF55 CSI packet shape: {array.shape}")
    else:
        raise ReaderError(f"Unsupported XRF55 CSI packet rank: {array.shape}")

    if rx_permutation is not None and canonical.shape[2] == rx_permutation.size:
        canonical = canonical[:, :, rx_permutation]

    return canonical.astype(np.complex64 if np.iscomplexobj(canonical) else np.float32, copy=False)


def _extract_xrf55_packet_from_struct(cell: np.ndarray) -> np.ndarray:
    record = np.asarray(cell).reshape(-1)[0]
    dtype_names = getattr(record, "dtype", None).names
    if not dtype_names or "csi" not in dtype_names:
        raise ReaderError("Unsupported XRF55 MAT record without 'csi' field")

    n_rx = _scalar_from_any(record["Nrx"], default=1) or 1
    n_tx = _scalar_from_any(record["Ntx"], default=1) or 1
    rx_permutation = _normalize_rx_permutation(np.asarray(record["perm"]), n_rx)
    return _canonicalize_csi_packet(
        record["csi"],
        n_tx=n_tx,
        n_rx=n_rx,
        rx_permutation=rx_permutation,
    )


def _coerce_payload_to_array(payload: Any) -> np.ndarray:
    if isinstance(payload, dict):
        for key in ("csi", "CSI", "data", "amp", "wifi"):
            if key in payload:
                return _coerce_payload_to_array(payload[key])
        raise ReaderError("Unsupported XRF55 payload dict without CSI-like keys")

    value = _unwrap_object_scalar(payload)
    array = np.asarray(value)

    if array.dtype.names and "csi" in array.dtype.names:
        packets = [_extract_xrf55_packet_from_struct(entry) for entry in array.reshape(-1)]
        return np.stack(packets, axis=0)

    if array.dtype == object:
        decoded_packets: list[np.ndarray] = []
        for item in array.reshape(-1):
            item_value = _unwrap_object_scalar(item)
            item_array = np.asarray(item_value)
            if item_array.dtype.names and "csi" in item_array.dtype.names:
                decoded_packets.append(_extract_xrf55_packet_from_struct(item_array))
        if decoded_packets:
            return np.stack(decoded_packets, axis=0)
        if array.size == 1:
            return _coerce_payload_to_array(array.item())

    return array


def _load_structured_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        for key in ("csi", "CSI", "data", "amp", "wifi"):
            if key in payload:
                return payload[key]
        return payload[payload.files[0]]
    if suffix == ".mat":
        payload = scio.loadmat(path)
        for key in ("csi", "CSI", "csi_data", "data", "CSIamp", "wifi"):
            if key in payload:
                return payload[key]
        raise ReaderError(f"Unsupported MAT payload in {path}")
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            for key in ("csi", "CSI", "data", "amp", "wifi"):
                if key in handle:
                    return np.asarray(handle[key])
            first_key = next(iter(handle.keys()))
            return np.asarray(handle[first_key])
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            for key in ("csi", "CSI", "data", "amp", "wifi"):
                if key in payload:
                    return payload[key]
        return payload
    raise ReaderError(f"Unsupported XRF55 sample suffix: {path.suffix}")


def _canonicalize_sample_array(payload: np.ndarray) -> np.ndarray:
    array = np.asarray(payload)
    if array.ndim == 0:
        raise ReaderError("XRF55 WiFi payload cannot be scalar")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    if array.ndim == 2:
        time_dim, feature_dim = int(array.shape[0]), int(array.shape[1])
        if feature_dim in _WIFI_SUBC_CANDIDATES:
            return array.reshape(time_dim, feature_dim, 1, 1)
        flat_layout = _FLAT_FEATURE_LAYOUTS.get(feature_dim)
        if flat_layout is not None:
            rx_dim, tx_dim, subc_dim = flat_layout
            return array.reshape(time_dim, rx_dim, tx_dim, subc_dim).transpose(0, 3, 2, 1)
        if time_dim in _WIFI_SUBC_CANDIDATES:
            return array.transpose(1, 0).reshape(int(array.shape[1]), int(array.shape[0]), 1, 1)
        raise ReaderError(f"Unsupported XRF55 2-D sample shape: {array.shape}")

    if array.ndim == 3:
        _, dim1, dim2 = array.shape
        if dim2 in _WIFI_SUBC_CANDIDATES:
            return array.transpose(0, 2, 1)[:, :, None, :]
        if dim1 in _WIFI_SUBC_CANDIDATES:
            return array[:, :, :, None]
        raise ReaderError(f"Unsupported XRF55 3-D sample shape: {array.shape}")

    if array.ndim == 4:
        _, dim1, dim2, dim3 = array.shape
        if dim1 in _WIFI_SUBC_CANDIDATES:
            return array
        if dim2 in _WIFI_SUBC_CANDIDATES:
            return array.transpose(0, 2, 1, 3)
        if dim3 in _WIFI_SUBC_CANDIDATES:
            return array.transpose(0, 3, 2, 1)
        raise ReaderError(f"Unsupported XRF55 4-D sample shape: {array.shape}")

    raise ReaderError(f"Unsupported XRF55 sample rank: {array.shape}")


class XRF55Reader(BaseWiFiReader):
    """Reader for XRF55 WiFi sample files."""

    modality = "wifi"
    device_family = "xrf55"
    device_name = "XRF55 WiFi"
    reader_version = "1.0"

    def __init__(self) -> None:
        super().__init__()
        config = self.reader_definition_bundle.config
        self._file_extensions = tuple(
            str(ext).lower()
            for ext in config.get(
                "file_extensions",
                (".npy", ".npz", ".mat", ".h5", ".hdf5", ".pkl", ".pickle"),
            )
        )

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"
        if path.suffix.lower() not in self._file_extensions:
            return False, (
                f"Invalid file extension: {path.suffix}. "
                f"XRF55Reader expects {', '.join(self._file_extensions)} files."
            )
        try:
            payload = _load_structured_payload(path)
            _canonicalize_sample_array(_coerce_payload_to_array(payload))
        except Exception as exc:
            return False, f"Invalid XRF55 WiFi payload: {exc}"
        return True, ""

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        return [self.read(file_path)]

    def read(self, file_path: str | Path) -> RadioTensor:
        path = Path(file_path)
        is_valid, msg = self.validate_format(path)
        if not is_valid:
            raise ReaderError(msg)

        payload = _load_structured_payload(path)
        raw_array = _coerce_payload_to_array(payload)
        canonical = _canonicalize_sample_array(raw_array)
        data = build_reader_axis_schema(self.reader_definition_bundle)
        tensor = np.asarray(canonical)
        signal = RadioTensor.from_reader(
            np_to_tensor(tensor),
            data,
            metadata=self._build_metadata(path, tensor),
        )
        return signal

    def _build_metadata(self, path: Path, tensor: np.ndarray) -> SignalMetadata:
        num_time, num_subc, num_tx, num_rx = (int(dim) for dim in tensor.shape)
        subcarrier_indices = build_centered_wifi_subcarrier_indices(num_subc)
        metadata = SignalMetadata(
            modality="wifi",
            subcarrier_spacing=resolve_wifi_subcarrier_spacing_hz(),
            subcarrier_indices=subcarrier_indices.tolist(),
            reader_id=self.reader_id,
            capture_device=self.device_name,
            extra={
                "sample_path": str(path),
                "source_format": path.suffix.lower(),
                "raw_shape": [int(dim) for dim in tensor.shape],
                "raw_dtype": str(tensor.dtype),
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
                "data_format": "complex64[time,subc,tx,rx]"
                if np.iscomplexobj(tensor)
                else "float32[time,subc,tx,rx]",
                "num_subc": num_subc,
                "num_tx": num_tx,
                "num_rx": num_rx,
            },
        )
        return metadata


def np_to_tensor(array: np.ndarray):
    import torch

    tensor = torch.from_numpy(array)
    if tensor.is_complex():
        return tensor.to(torch.complex64).contiguous()
    return tensor.float().contiguous()


__all__ = ["XRF55Reader"]
