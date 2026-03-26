"""Atheros WiFi CSI reader implementation.

Parses CSI data from Atheros WiFi chips (ath9k CSI tool format).
"""

import logging
import struct
from pathlib import Path

import numpy as np
import torch

from octosense.io.semantics.loader import load_reader_definition_bundle
from octosense.io.tensor import RadioTensor, SignalMetadata, build_wifi_csi_axis_schema
from octosense.io.readers.wifi.base import BaseWiFiReader, ReaderError
from octosense.io.profiles.wifi import (
    build_centered_wifi_subcarrier_indices,
    channel_to_center_freq,
    resolve_wifi_subcarrier_spacing_hz,
)

logger = logging.getLogger(__name__)

_READER_DEFINITION_BUNDLE = load_reader_definition_bundle("wifi", "atheros")
_CONFIG = _READER_DEFINITION_BUNDLE.config
DATA_COLUMNS_NAMES = tuple(_CONFIG["data_column_names"])
FILE_EXTENSIONS = tuple(_CONFIG["file_extensions"])
SIZE_STRUCT_BE = struct.Struct(_CONFIG["size_struct_be"])
SIZE_STRUCT_LE = struct.Struct(_CONFIG["size_struct_le"])
HEADER_STRUCT_BE = struct.Struct(_CONFIG["header_struct_be"])
HEADER_STRUCT_LE = struct.Struct(_CONFIG["header_struct_le"])
BITS_PER_SYMBOL = int(_CONFIG["bits_per_symbol"])
BANDWIDTH_MHZ_FOR_ZERO = int(_CONFIG["bandwidth_mhz_for_zero"])
BANDWIDTH_MHZ_FOR_NONZERO = int(_CONFIG["bandwidth_mhz_for_nonzero"])
_CANONICAL_DIMENSION_NAMES = tuple(build_wifi_csi_axis_schema().axes)
_CANONICAL_DATA_FORMAT = "complex64[time,subc,tx,rx]"
_PILOT_INDICES_BY_NUM_TONES = {
    int(num_tones): tuple(int(index) for index in indices)
    for num_tones, indices in _CONFIG.get("pilot_indices_by_num_tones", {}).items()
}


def _signbit_convert(data: int, maxbit: int) -> int:
    if data & (1 << (maxbit - 1)):
        data -= 1 << maxbit
    return data


def _get_next_bits(
    buf: bytes, current_data: int, idx: int, bits_left: int
) -> tuple[int, int, int]:
    """Read next 16-bit chunk from buffer.

    Raises:
        ReaderError: If buffer is too short (truncated CSI payload).
    """
    if idx + 1 >= len(buf):
        raise ReaderError(
            "Truncated Atheros CSI payload",
            offset=idx,
            context={"buffer_length": len(buf), "required_index": idx + 1},
        )
    h_data = buf[idx]
    h_data += buf[idx + 1] << 8
    current_data += h_data << bits_left
    idx += 2
    bits_left += 16
    return current_data, idx, bits_left


class AtherosReader(BaseWiFiReader):
    """Atheros WiFi CSI reader."""

    modality = "wifi"
    device_family = "atheros"
    device_name = str(_CONFIG["device_name"])
    reader_version = str(_CONFIG["reader_version"])

    def __init__(self) -> None:
        super().__init__()
        self._size_struct = SIZE_STRUCT_LE
        self._header_struct = HEADER_STRUCT_LE
        self._initial = 0

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        """Validate file format for Atheros reader.

        This method is side-effect-free: it does not mutate reader state.

        Returns:
            Tuple of (is_valid, error_message). error_message is empty when valid.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        if file_path.suffix.lower() not in FILE_EXTENSIONS:
            return False, (
                f"Invalid file extension: {file_path.suffix}. "
                f"AtherosReader expects {', '.join(FILE_EXTENSIONS)} files."
            )

        try:
            with open(file_path, "rb") as f:
                endian = f.read(1)
                if endian == b"\xFF":
                    size_struct = SIZE_STRUCT_BE
                    header_struct = HEADER_STRUCT_BE
                    initial = 1
                else:
                    size_struct = SIZE_STRUCT_LE
                    header_struct = HEADER_STRUCT_LE
                    initial = 0

                f.seek(initial)
                header = f.read(30)
                if len(header) < 30:
                    return False, "File too short to be a valid Atheros CSI file."

                header_block = header_struct.unpack(header[2:27])
                nr = header_block[8]
                nc = header_block[9]
                if not (1 <= nr <= 3 and 1 <= nc <= 3):
                    return False, (
                        f"Invalid antenna counts (nr={nr}, nc={nc}). "
                        "Not a valid Atheros CSI file."
                    )
                return True, ""
        except Exception as e:
            return False, f"Failed to validate file: {e}"

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        file_path = Path(file_path)
        is_valid, msg = self.validate_format(file_path)
        if not is_valid:
            raise ReaderError(msg)

        # Detect endianness (side effect confined to read_file)
        with open(file_path, "rb") as f:
            endian = f.read(1)
        if endian == b"\xFF":
            self._size_struct = SIZE_STRUCT_BE
            self._header_struct = HEADER_STRUCT_BE
            self._initial = 1
        else:
            self._size_struct = SIZE_STRUCT_LE
            self._header_struct = HEADER_STRUCT_LE
            self._initial = 0

        signals: list[RadioTensor] = []

        with open(file_path, "rb") as f:
            f.seek(self._initial)
            while True:
                size_bytes = f.read(2)
                if len(size_bytes) < 2:
                    break
                size = self._size_struct.unpack(size_bytes)[0]
                data_block = f.read(size)
                if len(data_block) != size:
                    break

                signal = self._parse_packet(data_block)
                if signal is not None:
                    signals.append(signal)

        if not signals:
            raise ReaderError(f"No valid CSI frames found in {file_path}.")
        self._assign_stream_sample_rate(signals)
        return signals

    def __getstate__(self):
        """Serialize state for pickling (struct.Struct -> format string)."""
        state = self.__dict__.copy()
        state["_size_struct_fmt"] = self._size_struct.format
        state["_header_struct_fmt"] = self._header_struct.format
        del state["_size_struct"]
        del state["_header_struct"]
        return state

    def __setstate__(self, state):
        """Restore state from pickle (format string -> struct.Struct)."""
        size_fmt = state.pop("_size_struct_fmt")
        header_fmt = state.pop("_header_struct_fmt")
        self.__dict__.update(state)
        self._size_struct = struct.Struct(size_fmt)
        self._header_struct = struct.Struct(header_fmt)

    def _parse_packet(self, data_block: bytes) -> RadioTensor | None:
        header = self._header_struct.unpack(data_block[:25])
        data_dict = {
            "dimension_names": _CANONICAL_DIMENSION_NAMES,
            "data_format": _CANONICAL_DATA_FORMAT,
            "timestamp": int(header[0]),
            "timestamp_raw": header[0],
            "csi_length": header[1],
            "tx_channel": header[2],
            "err_info": header[3],
            "noise_floor": header[4],
            "rate": header[5],
            "bandwidth": header[6],
            "num_tones": header[7],
            "nr": header[8],
            "nc": header[9],
            "rssi": header[10],
            "rssi_1": header[11],
            "rssi_2": header[12],
            "rssi_3": header[13],
            "payload_length": header[14],
            "csi_matrix": self._parse_csi_array(data_block[25 : 25 + header[1]]),
        }

        data_dict = self._cleanse_data(data_dict)
        if data_dict is None:
            return None

        csi = data_dict["csi_matrix"]  # (subc, tx, rx)
        data = torch.from_numpy(csi).unsqueeze(0)  # (1, subc, tx, rx)

        metadata = self._build_metadata(data_dict)
        return RadioTensor.from_reader(data, build_wifi_csi_axis_schema(), metadata=metadata)

    def _cleanse_data(self, data_dict: dict) -> dict | None:
        data_dict["bandwidth"] = (
            BANDWIDTH_MHZ_FOR_ZERO
            if data_dict["bandwidth"] == 0
            else BANDWIDTH_MHZ_FOR_NONZERO
        )
        expected = data_dict["nc"] * data_dict["nr"] * data_dict["num_tones"]
        if data_dict["csi_matrix"].size != expected:
            logger.warning("CSI matrix size mismatch, skipping frame.")
            return None
        data_dict["csi_matrix"] = data_dict["csi_matrix"].reshape(
            (data_dict["num_tones"], data_dict["nc"], data_dict["nr"])
        )
        return data_dict

    def _parse_csi_array(self, data_block: bytes) -> np.ndarray:
        num_complex = (len(data_block) * 8) // (2 * BITS_PER_SYMBOL)
        complex_vals = np.empty(num_complex, dtype=np.complex64)

        bitmask = (1 << BITS_PER_SYMBOL) - 1
        idx = 0
        bits_left = 16

        h_data = data_block[idx]
        idx += 1
        h_data += data_block[idx] << 8
        idx += 1
        current_data = h_data & ((1 << 16) - 1)

        for i in range(num_complex):
            if bits_left - BITS_PER_SYMBOL < 0:
                current_data, idx, bits_left = _get_next_bits(
                    data_block, current_data, idx, bits_left
                )

            imag = current_data & bitmask
            imag = _signbit_convert(imag, BITS_PER_SYMBOL) + 1

            bits_left -= BITS_PER_SYMBOL
            current_data >>= BITS_PER_SYMBOL

            if bits_left - BITS_PER_SYMBOL < 0:
                current_data, idx, bits_left = _get_next_bits(
                    data_block, current_data, idx, bits_left
                )

            real = current_data & bitmask
            real = _signbit_convert(real, BITS_PER_SYMBOL) + 1

            bits_left -= BITS_PER_SYMBOL
            current_data >>= BITS_PER_SYMBOL

            complex_vals[i] = real + imag * 1j

        return complex_vals

    def _build_metadata(self, data_dict: dict) -> SignalMetadata:
        binding_input = {
            key: value for key, value in data_dict.items() if key != "csi_matrix"
        }
        channel = int(data_dict["tx_channel"])
        try:
            center_freq = channel_to_center_freq(channel)
        except ValueError:
            if 2000 <= channel <= 7000:
                center_freq = float(channel) * 1e6
            else:
                raise
        bandwidth = float(data_dict["bandwidth"]) * 1e6
        num_tones = int(data_dict["num_tones"])
        timestamp_raw = int(data_dict["timestamp_raw"])

        metadata = SignalMetadata(
            modality="wifi",
            center_freq=center_freq,
            bandwidth=bandwidth,
            subcarrier_spacing=resolve_wifi_subcarrier_spacing_hz(),
            subcarrier_indices=build_centered_wifi_subcarrier_indices(num_tones).tolist(),
            reader_id=self.reader_id,
            capture_device=self.device_name,
            data_version=f"ts_{timestamp_raw}",
        )

        metadata.set_coord("time", np.array([0.0]), unit="s")
        metadata.set_coord("subc", build_centered_wifi_subcarrier_indices(num_tones), unit="index")

        self._finalize_runtime_contract(metadata, raw_payload=binding_input)
        pilot_indices = _PILOT_INDICES_BY_NUM_TONES.get(num_tones)
        if pilot_indices is not None:
            metadata.extra["pilot_indices"] = list(pilot_indices)
            metadata.extra["pilot_interpolated"] = False

        return metadata
