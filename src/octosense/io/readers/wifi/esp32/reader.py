"""ESP32 WiFi CSI reader implementation.

Reads CSI data from ESP32 WiFi chips (ESP32, ESP32-C3, ESP32-C5, ESP32-S2, ESP32-S3).

Reference:
- Official ESP32-CSI toolkit: https://github.com/espressif/esp-csi
- opensense-core ESP32 reader implementation

Format:
- CSV format with JSON arrays in 'data' column
- Columns: `type`, `id`, `mac_addr`, `rssi`, `rate`, `noise_floor`, `fft_gain`, `agc_gain`, `channel`, `local_timestamp`, `sig_len`, `rx_state`, `len`, `first_word`, `data`
- CSI data formats: 106, 114, 128, 234, 256, 384 subcarriers
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from octosense.io.readers.wifi.base import (
    BaseWiFiReader,
    ReaderError,
    _compact_metadata_kwargs,
)
from octosense.io.profiles.wifi import (
    bandwidth_from_subcarrier_count,
    build_centered_wifi_subcarrier_indices,
    channel_to_center_freq,
    resolve_wifi_subcarrier_spacing_hz,
)
from octosense.io.semantics.loader import load_reader_definition_bundle
from octosense.io.tensor import RadioTensor, SignalMetadata, build_wifi_csi_axis_schema

logger = logging.getLogger(__name__)

_READER_DEFINITION_BUNDLE = load_reader_definition_bundle("wifi", "esp32")
_CONFIG = _READER_DEFINITION_BUNDLE.config
VALID_CSI_FORMATS = {int(value) for value in _CONFIG["valid_csi_formats"]}
FORMAT_TO_SUBCARRIERS = {
    int(key): int(value) for key, value in _CONFIG["format_to_subcarriers"].items()
}
FILE_EXTENSIONS = tuple(_CONFIG["file_extensions"])
REQUIRED_FIELDS = set(_CONFIG["required_fields"])
DEFAULT_NOISE_FLOOR = int(_CONFIG["default_noise_floor_dbm"])
BANDWIDTH_BY_MAX_SUBC = {
    int(key): float(value) for key, value in _CONFIG["bandwidth_by_max_subc_hz"].items()
}
_CANONICAL_DIMENSION_NAMES = tuple(build_wifi_csi_axis_schema().axes)
_CANONICAL_DATA_FORMAT = "complex64[time,subc,tx,rx]"


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value, 0)


def _parse_optional_number(value: str | None) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        return int(value, 0)
    except ValueError:
        return float(value)


class ESP32Reader(BaseWiFiReader):
    """ESP32 WiFi CSI reader.

    Reads CSI data from ESP32 family chips and outputs RadioTensor objects
    with complete metadata.

    Example:
        >>> reader = ESP32Reader()
        >>> signals = reader.read_file('esp32_data.csv')
        >>> print(signals[0].describe())
    """

    modality = "wifi"
    device_family = "esp32"
    device_name = str(_CONFIG["device_name"])
    reader_version = str(_CONFIG["reader_version"])

    def __init__(self) -> None:
        """Initialize ESP32 reader."""
        super().__init__()
        self.current_format: str | None = None

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        """Validate CSV format with required columns.

        Args:
            file_path: Path to CSV file

        Returns:
            Tuple of (is_valid, error_message). error_message is empty when valid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False, f"File not found: {file_path}"

        if file_path.suffix.lower() not in FILE_EXTENSIONS:
            return False, (
                f"Invalid file extension: {file_path.suffix}. "
                f"ESP32Reader expects {', '.join(FILE_EXTENSIONS)} files."
            )

        # Check first line for required columns
        try:
            with open(file_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                missing = REQUIRED_FIELDS - set(fieldnames)

                if missing:
                    return False, (
                        f"Missing required columns: {missing}. "
                        f"Found columns: {fieldnames}."
                    )

                return True, ""

        except Exception as e:
            return False, f"Failed to read CSV: {e}"

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        """Read all CSI frames from ESP32 CSV file.

        Args:
            file_path: Path to ESP32 CSV file

        Returns:
            List of RadioTensor objects (one per valid CSI frame)

        Raises:
            ReaderError: If file format is invalid or parsing fails
        """
        file_path = Path(file_path)
        is_valid, msg = self.validate_format(file_path)
        if not is_valid:
            raise ReaderError(msg)

        signals = []

        with open(file_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader, start=1):
                try:
                    # Filter CSI_DATA frames
                    if row.get('type') != 'CSI_DATA':
                        continue

                    # Parse row
                    signal = self._parse_row(row, row_offset=i)
                    if signal is not None:
                        signals.append(signal)

                except ReaderError:
                    raise
                except Exception as e:
                    logger.warning(f"Failed to parse row {i}: {e}")
                    continue

        if not signals:
            raise ReaderError(
                f"No valid CSI frames found in {file_path}. "
                f"Fix: Check that file contains CSI_DATA frames with valid 'data' arrays."
            )

        self._assign_stream_sample_rate(signals)
        logger.info(f"Read {len(signals)} CSI frames from {file_path}")
        return signals

    def _parse_row(self, row: dict[str, str], row_offset: int = 0) -> RadioTensor | None:
        """Parse a single CSV row into RadioTensor.

        Args:
            row: CSV row as dictionary
            row_offset: Row number in original file for error reporting

        Returns:
            RadioTensor or None if parsing fails
        """
        # Extract basic metadata
        channel_raw = row.get("channel")
        channel = int(channel_raw) if channel_raw not in {None, ""} else None
        rssi = float(row.get('rssi', 0))
        noise_floor = int(row.get("noise_floor", DEFAULT_NOISE_FLOOR))
        local_timestamp = int(row.get('local_timestamp', 0))
        seq_num = int(row.get('id', 0))
        mac_addr = row.get('mac_addr') or row.get('mac') or 'unknown'
        raw_fields: dict[str, object] = {
            "type": row.get("type", ""),
            "id": seq_num,
            "mac_addr": mac_addr,
            "rssi": rssi,
            "channel": channel,
            "local_timestamp": local_timestamp,
            "noise_floor": noise_floor,
        }
        for key, value in (
            ("rate", _parse_optional_number(row.get("rate"))),
            ("fft_gain", _parse_optional_int(row.get("fft_gain"))),
            ("agc_gain", _parse_optional_int(row.get("agc_gain"))),
            ("sig_len", _parse_optional_int(row.get("sig_len"))),
            ("rx_state", _parse_optional_int(row.get("rx_state"))),
            ("len", _parse_optional_int(row.get("len"))),
            ("first_word", _parse_optional_int(row.get("first_word"))),
        ):
            if value is not None:
                raw_fields[key] = value

        # Parse CSI data array
        data_str = row.get('data', '[]')
        try:
            csi_array = self._parse_csi_array(data_str)
        except json.JSONDecodeError:
            raise ReaderError(
                "Invalid JSON in ESP32 CSI payload",
                offset=row_offset,
                context={"payload_preview": data_str[:80]},
            )
        except Exception as e:
            logger.warning(f"Failed to parse CSI array: {e}")
            return None

        if csi_array is None:
            return None

        # Compute metadata
        center_freq, bandwidth = self._compute_freq_params(channel, csi_array.shape[0])

        # Build SignalMetadata
        binding_input = self._build_binding_input(
            raw_fields=raw_fields,
            csi_array=csi_array,
            center_freq=center_freq,
            bandwidth=bandwidth,
            seq_num=seq_num,
        )
        metadata = self._build_metadata(
            center_freq=center_freq,
            bandwidth=bandwidth,
            seq_num=seq_num,
            num_subc=csi_array.shape[0],
            binding_input=binding_input,
        )
        return self._build_tensor(csi_array, metadata)

    def _parse_csi_array(self, data_str: str) -> np.ndarray | None:
        """Parse CSI data string into complex numpy array.

        Args:
            data_str: JSON array string (e.g., '[1,2,3,4,...]')

        Returns:
            Complex numpy array of shape (num_subc, 1, 1) or None if invalid

        Raises:
            json.JSONDecodeError: If JSON parsing fails (caught by caller).
        """
        # Parse JSON array - let JSONDecodeError propagate
        data_list = json.loads(data_str)

        if not isinstance(data_list, list):
            return None

        # Validate format
        data_len = len(data_list)
        if data_len not in VALID_CSI_FORMATS:
            logger.warning(
                f"Invalid CSI format: {data_len} values. "
                f"Valid formats: {VALID_CSI_FORMATS}"
            )
            return None
        self.current_format = str(data_len)

        # Convert to complex array (real/imag pairs)
        real_parts = np.array(data_list[0::2], dtype=np.int16)
        imag_parts = np.array(data_list[1::2], dtype=np.int16)

        csi_complex = real_parts.astype(np.float32) + 1j * imag_parts.astype(np.float32)

        # Determine number of subcarriers
        num_subc = FORMAT_TO_SUBCARRIERS.get(data_len, len(csi_complex))

        # Truncate to actual subcarriers (e.g., 128 values -> 64 subcarriers)
        csi_complex = csi_complex[:num_subc]

        # Reshape to (num_subc, 1, 1) for (subc, tx, rx)
        return csi_complex.reshape(num_subc, 1, 1)

    def _build_tensor(
        self,
        csi_array: np.ndarray,
        metadata: SignalMetadata,
    ) -> RadioTensor:
        """Build RadioTensor from CSI array.

        Args:
            csi_array: Complex CSI array of shape (num_subc, tx, rx)

        Returns:
            RadioTensor with AxisSchema
        """
        # Convert to PyTorch tensor (add time dimension)
        data = torch.from_numpy(csi_array).unsqueeze(0)  # (1, subc, tx, rx)

        return RadioTensor.from_reader(data, build_wifi_csi_axis_schema(), metadata=metadata)

    def _build_metadata(
        self,
        center_freq: float | None,
        bandwidth: float,
        seq_num: int,
        num_subc: int,
        binding_input: dict[str, object],
    ) -> SignalMetadata:
        """Build SignalMetadata for ESP32 CSI frame."""
        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality="wifi",
                center_freq=center_freq,
                bandwidth=bandwidth,
                subcarrier_spacing=resolve_wifi_subcarrier_spacing_hz(),
                subcarrier_indices=build_centered_wifi_subcarrier_indices(num_subc).tolist(),
                reader_id=self.reader_id,
                capture_device=self.device_name,
                data_version=f"seq_{seq_num}",
            )
        )

        # Explicit coordinate axes
        metadata.set_coord("time", np.array([0.0]), unit="s")
        metadata.set_coord("subc", build_centered_wifi_subcarrier_indices(num_subc), unit="index")

        self._finalize_runtime_contract(metadata, raw_payload=binding_input)

        return metadata

    def _build_binding_input(
        self,
        *,
        raw_fields: dict[str, object],
        csi_array: np.ndarray,
        center_freq: float | None,
        bandwidth: float,
        seq_num: int,
    ) -> dict[str, object]:
        """Build one binding payload that feeds the shared WiFi runtime helper."""
        binding_input: dict[str, object] = {
            key: raw_fields[key]
            for key in (
                "type",
                "id",
                "mac_addr",
                "rssi",
                "rate",
                "noise_floor",
                "fft_gain",
                "agc_gain",
                "channel",
                "local_timestamp",
                "rx_state",
                "first_word",
            )
            if key in raw_fields
        }
        if "len" in raw_fields:
            binding_input["payload_size"] = raw_fields["len"]
        elif "sig_len" in raw_fields:
            binding_input["payload_size"] = raw_fields["sig_len"]

        binding_input.update(
            {
                "device_type": self.device_name,
                "format": "esp32",
                "format_type": self.current_format or str(csi_array.shape[0] * 2),
                "timestamp": raw_fields["local_timestamp"],
                "dimension_names": _CANONICAL_DIMENSION_NAMES,
                "data_format": _CANONICAL_DATA_FORMAT,
                "tx_count": int(csi_array.shape[1]),
                "rx_count": int(csi_array.shape[2]),
                "subcarrier_count": int(csi_array.shape[0]),
                "sequence_num": seq_num,
                "bandwidth": bandwidth,
            }
        )
        if center_freq is not None:
            binding_input["center_freq"] = center_freq
        return binding_input

    @staticmethod
    def _compute_freq_params(channel: int | None, num_subc: int) -> tuple[float | None, float]:
        """Compute center frequency and bandwidth from channel number."""
        center_freq = None if channel is None else channel_to_center_freq(channel)
        bandwidth = bandwidth_from_subcarrier_count(num_subc, BANDWIDTH_BY_MAX_SUBC)
        return center_freq, bandwidth
