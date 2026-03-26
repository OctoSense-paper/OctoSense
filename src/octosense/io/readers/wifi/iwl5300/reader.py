"""Intel IWL5300 WiFi CSI reader implementation.

Reads Intel 5300 `.dat` captures through the external `csiread` parser.
This module keeps only the minimal header inspection needed for validation and
metadata recovery, and delegates CSI payload decoding plus antenna permutation
to `csiread` instead of maintaining a duplicate in-tree bit parser.

Reference:
- Linux 802.11n CSI Tool: https://dhalperi.github.io/linux-80211n-csitool/

Format:
- Binary format with 0xBB validation code
- Fixed 30 subcarriers
- Packed 8-bit signed real/imaginary values
- Antenna permutation metadata
"""

from copy import deepcopy
from collections.abc import Mapping
import logging
import math
import struct
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from octosense.io.semantics.loader import load_reader_definition_bundle
from octosense.io.tensor import RadioTensor, SignalMetadata, build_wifi_csi_axis_schema
from octosense.io.readers.wifi.base import (
    BaseWiFiReader,
    ReaderError,
    _compact_metadata_kwargs,
    _optional_float,
)
from octosense.io.profiles.wifi import (
    channel_to_center_freq,
    resolve_wifi_subcarrier_spacing_hz,
)

logger = logging.getLogger(__name__)

_READER_DEFINITION_BUNDLE = load_reader_definition_bundle("wifi", "iwl5300")
_CONFIG = _READER_DEFINITION_BUNDLE.config
VALID_CODE = int(_CONFIG["valid_code"])
HEADER_STRUCT = struct.Struct(_CONFIG["header_struct"])
SIZE_STRUCT = struct.Struct(_CONFIG["size_struct"])
CODE_STRUCT = struct.Struct(_CONFIG["code_struct"])
FILE_EXTENSIONS = tuple(_CONFIG["file_extensions"])
_RATE_MCS_HT_MSK = int(_CONFIG["rate_ht_msk"])
_RATE_MCS_HT40_MSK = int(_CONFIG["rate_ht40_msk"])
IWL5300_SUBCARRIER_SPACING_NARROW_HZ = float(
    _CONFIG["subcarrier_spacing_hz"]["narrow_78k"]
)
# IWL5300 exposes a sparse 30-tone subset, but the stored subcarrier indices
# are still canonical OFDM bin indices. Metadata therefore reports the physical
# WiFi tone spacing, not the larger gap between sampled bins.
IWL5300_SUBCARRIER_SPACING_20MHZ_HZ = resolve_wifi_subcarrier_spacing_hz(
    narrow_spacing=False
)
IWL5300_SUBCARRIER_SPACING_40MHZ_HZ = resolve_wifi_subcarrier_spacing_hz(
    narrow_spacing=False
)
_IWL5300_20MHZ_SUBCARRIER_INDICES = [
    int(value) for value in _CONFIG["subcarrier_indices_20mhz"]
]
_IWL5300_40MHZ_SUBCARRIER_INDICES = [
    int(value) for value in _CONFIG["subcarrier_indices_40mhz"]
]
_CANONICAL_DIMENSION_NAMES = tuple(build_wifi_csi_axis_schema().axes)
_CANONICAL_DATA_FORMAT = "complex64[time,subc,tx,rx]"
_IWL5300_CANONICAL_20MHZ_BANDWIDTH_HZ = 20.0e6


def get_subcarrier_spacing(
    narrow_spacing: bool = False,
    bandwidth: float | None = None,
    subcarrier_group_30: bool = True,
) -> float:
    """Get subcarrier spacing based on WiFi protocol configuration.

    Args:
        narrow_spacing: If True, use 802.11ax narrow spacing (78.125 kHz)
        bandwidth: Channel bandwidth in Hz. If omitted, use the reader definition default.
        subcarrier_group_30: If True, use 30-subcarrier grouping (IWL5300 default)

    Returns:
        Physical subcarrier spacing in Hz.

    Raises:
        ValueError: If configuration is not supported

    Example:
        >>> # IWL5300 default grouped CSI spacing
        >>> get_subcarrier_spacing(bandwidth=20e6, subcarrier_group_30=True)
        312500.0
    """
    if bandwidth is None:
        bandwidth = _IWL5300_CANONICAL_20MHZ_BANDWIDTH_HZ

    normalized_bandwidth = float(bandwidth)
    if subcarrier_group_30 and normalized_bandwidth not in {20e6, 40e6} and not narrow_spacing:
        raise ValueError(
            f"Unsupported configuration: narrow_spacing={narrow_spacing}, "
            f"bandwidth={normalized_bandwidth / 1e6:.0f} MHz, "
            f"subcarrier_group_30={subcarrier_group_30}"
        )
    if narrow_spacing:
        return IWL5300_SUBCARRIER_SPACING_NARROW_HZ
    if not subcarrier_group_30:
        return resolve_wifi_subcarrier_spacing_hz(narrow_spacing=False)
    if normalized_bandwidth == 20e6:
        return IWL5300_SUBCARRIER_SPACING_20MHZ_HZ
    return IWL5300_SUBCARRIER_SPACING_40MHZ_HZ


class IWL5300Reader(BaseWiFiReader):
    """Intel IWL5300 WiFi CSI reader.

    Reads binary CSI data from Intel 5300 NIC and outputs RadioTensor objects.

    Example:
        >>> reader = IWL5300Reader()
        >>> signals = reader.read_file('iwl5300_data.dat')
        >>> print(signals[0].describe())
    """

    modality = "wifi"
    device_family = "iwl5300"
    device_name = str(_CONFIG["device_name"])
    reader_version = str(_CONFIG["reader_version"])

    def __init__(
        self,
        channel: int | None = None,
        *,
        center_freq_hz: float | None = None,
        bandwidth_hz: float | None = None,
    ) -> None:
        """Initialize IWL5300 reader.

        Args:
            channel: WiFi channel number for center frequency calculation.
                Required unless ``center_freq_hz`` is provided explicitly.
            center_freq_hz: Explicit center frequency override in Hz.
            bandwidth_hz: Explicit bandwidth override in Hz. Required when the
                capture header does not expose a rate bitfield.
        """
        super().__init__()
        self.channel = channel
        self.center_freq_hz = center_freq_hz
        self.bandwidth_hz = bandwidth_hz

    def _coerce_input_path(self, file_path: str | Path) -> Path:
        """Validate the basic IWL5300 file path contract without parsing packets."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ReaderError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in FILE_EXTENSIONS:
            raise ReaderError(
                f"Invalid file extension: {file_path.suffix}. "
                f"IWL5300Reader expects {', '.join(FILE_EXTENSIONS)} files."
            )

        return file_path

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        """Validate binary format with 0xBB code.

        Args:
            file_path: Path to .dat binary file

        Returns:
            Tuple of (is_valid, error_message). error_message is empty when valid.
        """
        try:
            file_path = self._coerce_input_path(file_path)
            self._read_first_valid_header(file_path)
            return True, ""
        except ReaderError as exc:
            return False, str(exc)
        except struct.error as exc:
            return False, f"Binary format error: {exc}"
        except Exception as exc:
            return False, f"Failed to validate file: {exc}"

    def _read_csiread_capture(self, file_path: str | Path) -> dict[str, np.ndarray]:
        """Parse an Intel 5300 capture through csiread and normalize array layouts."""
        file_path = self._coerce_input_path(file_path)
        first_header = self._read_first_valid_header(file_path)
        csiread = self._load_csiread()

        try:
            nrxnum = max(int(first_header["n_rx"]), 1)
            ntxnum = max(int(first_header["n_tx"]), 1)
            parser = csiread.Intel(
                str(file_path),
                nrxnum=nrxnum,
                ntxnum=ntxnum,
                pl_size=0,
                if_report=False,
            )
            parser.read()
        except Exception as exc:
            raise ReaderError(f"Failed to read {file_path} with csiread: {exc}") from exc

        packet_count = int(parser.count)
        if packet_count <= 0:
            raise ReaderError(f"No valid CSI frames found in {file_path}.")

        csi_tx_first = np.transpose(
            parser.csi[:packet_count, :, :nrxnum, :ntxnum],
            (0, 1, 3, 2),
        ).astype(np.complex64, copy=False)
        timestamp_low = np.asarray(parser.timestamp_low[:packet_count], dtype=np.int64)
        return {
            "csi_tx_first": csi_tx_first,
            "timestamp_low": timestamp_low,
            "bfee_count": np.asarray(parser.bfee_count[:packet_count], dtype=np.int64),
            "n_rx": np.asarray(parser.Nrx[:packet_count], dtype=np.int64),
            "n_tx": np.asarray(parser.Ntx[:packet_count], dtype=np.int64),
            "rssi_a": np.asarray(parser.rssi_a[:packet_count], dtype=np.int64),
            "rssi_b": np.asarray(parser.rssi_b[:packet_count], dtype=np.int64),
            "rssi_c": np.asarray(parser.rssi_c[:packet_count], dtype=np.int64),
            "noise": np.asarray(parser.noise[:packet_count], dtype=np.int64),
            "agc": np.asarray(parser.agc[:packet_count], dtype=np.int64),
            "rate": np.asarray(parser.rate[:packet_count], dtype=np.int64),
        }

    def _build_header_from_capture(
        self,
        capture: dict[str, np.ndarray],
        index: int,
    ) -> dict[str, float | int]:
        return {
            "timestamp_low": int(capture["timestamp_low"][index]),
            "bfee_count": int(capture["bfee_count"][index]),
            "n_rx": int(capture["n_rx"][index]),
            "n_tx": int(capture["n_tx"][index]),
            "rssi_a": int(capture["rssi_a"][index]),
            "rssi_b": int(capture["rssi_b"][index]),
            "rssi_c": int(capture["rssi_c"][index]),
            "noise": int(capture["noise"][index]),
            "agc": int(capture["agc"][index]),
            "rate": int(capture["rate"][index]),
        }

    def _read_packet_tensors(self, file_path: str | Path) -> list[RadioTensor]:
        """Read all CSI packets from one IWL5300 capture as per-packet tensors."""
        capture = self._read_csiread_capture(file_path)
        signals: list[RadioTensor] = []
        for index, frame_tx_first in enumerate(capture["csi_tx_first"]):
            header = self._build_header_from_capture(capture, index)
            metadata = self._build_metadata(header)
            tensor = self._build_tensor(np.transpose(frame_tx_first, (0, 2, 1)), metadata)
            signals.append(tensor)

        self._assign_stream_sample_rate(signals)
        logger.info(f"Read {len(signals)} CSI frames from {file_path}")
        return signals

    def read_file(self, file_path: str | Path) -> RadioTensor:
        """Canonical file/sample reader path for Intel 5300 captures."""
        return self._read_sample_tensor(file_path)

    def read(self, file_path: str | Path) -> RadioTensor:
        """Canonical sample-level reader path for Intel 5300 captures."""
        return self._read_sample_tensor(file_path)

    def _read_sample_tensor(self, file_path: str | Path) -> RadioTensor:
        capture = self._read_csiread_capture(file_path)
        packet_count = int(capture["csi_tx_first"].shape[0])
        if packet_count < 2:
            raise ReaderError(
                f"Need at least 2 packets to compute sample_rate, got {packet_count}. "
                f"File: {file_path}"
            )

        merged_array = np.ascontiguousarray(capture["csi_tx_first"], dtype=np.complex64)
        merged_data = torch.from_numpy(merged_array)
        timestamps = self._capture_timestamps_seconds(capture)
        sample_rate = self._infer_sample_rate_from_timestamps(timestamps)
        first_header = self._build_header_from_capture(capture, 0)
        metadata = self._build_sample_metadata(
            first_header,
            timestamps=timestamps,
            sample_rate=sample_rate,
            num_tx=int(merged_data.shape[2]) if merged_data.ndim >= 3 else 0,
            num_rx=int(merged_data.shape[3]) if merged_data.ndim >= 4 else 0,
        )
        merged_tensor = RadioTensor.from_reader(
            merged_data,
            self._canonical_wifi_axis_schema(),
            metadata=metadata,
        )
        logger.info(
            "Merged %d packets into time series: shape=%s, sample_rate=%s Hz",
            packet_count,
            tuple(merged_tensor.shape),
            "unknown" if sample_rate is None else f"{sample_rate:.2f}",
        )
        return merged_tensor

    def _load_csiread(self) -> Any:
        try:
            return import_module("csiread")
        except ImportError as exc:
            raise ReaderError(
                "IWL5300 fast path requires the 'csiread' package. "
                "Install project dependencies before reading Intel 5300 CSI."
            ) from exc

    def _read_first_valid_header(self, file_path: Path) -> dict[str, float | int]:
        try:
            with open(file_path, "rb") as f:
                header_prefix = f.read(3)
                if len(header_prefix) < 3:
                    raise ReaderError(f"File too short: {len(header_prefix)} bytes.")

                size = SIZE_STRUCT.unpack(header_prefix[:2])[0]
                code = CODE_STRUCT.unpack(header_prefix[2:3])[0]
                if code != VALID_CODE:
                    raise ReaderError(
                        f"Invalid validation code: 0x{code:02X} "
                        f"(expected 0x{VALID_CODE:02X})."
                    )

                data_block = f.read(size - 1)
                if len(data_block) != size - 1:
                    raise ReaderError(
                        f"Incomplete first packet (expected {size - 1}, got {len(data_block)})"
                    )

            header = self._parse_header(data_block[:20])
        except ReaderError:
            raise
        except Exception as exc:
            raise ReaderError(f"Failed to inspect {file_path}: {exc}") from exc

        if header is None:
            raise ReaderError(f"Failed to parse first CSI packet header in {file_path}.")
        return header

    def _parse_header(self, header_bytes: bytes) -> dict | None:
        """Parse 20-byte packet header."""
        try:
            fields = HEADER_STRUCT.unpack(header_bytes)

            timestamp_low = fields[0]
            bfee_count = fields[1]
            n_rx = fields[3]
            n_tx = fields[4]
            rssi_a = fields[5]
            rssi_b = fields[6]
            rssi_c = fields[7]
            noise = fields[8]
            agc = fields[9]
            antenna_sel = fields[10]
            payload_len = fields[11]
            rate = fields[12]

            return {
                "timestamp_low": timestamp_low,
                "bfee_count": bfee_count,
                "n_rx": n_rx,
                "n_tx": n_tx,
                "rssi_a": rssi_a,
                "rssi_b": rssi_b,
                "rssi_c": rssi_c,
                "noise": noise,
                "agc": agc,
                "antenna_sel": antenna_sel,
                "length": payload_len,
                "rate": rate,
            }

        except struct.error as e:
            logger.warning(f"Header parse error: {e}")
            return None

    def _build_tensor(self, csi_array: np.ndarray, metadata: SignalMetadata) -> RadioTensor:
        """Build RadioTensor from CSI array."""
        # Transpose to (num_subc, n_tx, n_rx) for consistency
        csi_array = np.transpose(csi_array, (0, 2, 1))

        # Convert to PyTorch tensor (add time dimension)
        data = torch.from_numpy(csi_array).unsqueeze(0)  # (1, subc, tx, rx)

        # Define AxisSchema
        return RadioTensor.from_reader(data, build_wifi_csi_axis_schema(), metadata=metadata)

    def _build_metadata(self, header: dict, sample_rate: float | None = None) -> SignalMetadata:
        """Build SignalMetadata from packet header."""
        binding_input = self._build_binding_input(header)
        center_freq = self._resolve_center_freq_hz()
        bandwidth = self._resolve_bandwidth_hz(header)
        subcarrier_indices = self._resolve_subcarrier_indices(bandwidth)

        subcarrier_spacing = get_subcarrier_spacing(
            narrow_spacing=False,
            bandwidth=bandwidth,
            subcarrier_group_30=True,
        )

        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality="wifi",
                center_freq=center_freq,
                bandwidth=bandwidth,
                sample_rate=_optional_float(sample_rate),
                subcarrier_spacing=subcarrier_spacing,
                subcarrier_indices=subcarrier_indices,
                reader_id=self.reader_id,
                capture_device=self.device_name,
                data_version=f"bfee_{int(header['bfee_count'])}",
            )
        )

        metadata.set_coord("time", np.array([0.0]), unit="s")
        metadata.set_coord(
            "subc",
            np.asarray(subcarrier_indices, dtype=np.float64),
            unit="index",
        )

        self._finalize_runtime_contract(metadata, raw_payload=binding_input)
        metadata.modality = "wifi"

        return metadata

    def _capture_timestamps_seconds(self, capture: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(capture["timestamp_low"], dtype=np.float64) * float(
            _CONFIG["timestamp_unit_s"]
        )

    def _infer_sample_rate_from_timestamps(self, timestamps: np.ndarray) -> float | None:
        if timestamps.size < 2:
            return None
        diffs = np.diff(timestamps)
        valid_diffs = diffs[diffs > 0]
        if valid_diffs.size == 0:
            return None
        return float(1.0 / np.median(valid_diffs))

    def _build_sample_metadata(
        self,
        header: Mapping[str, object],
        *,
        timestamps: np.ndarray,
        sample_rate: float | None,
        num_tx: int,
        num_rx: int,
    ) -> SignalMetadata:
        binding_input = self._build_binding_input(header)
        canonical_payload = self._canonicalize_payload(binding_input, keep_unmapped=False)
        center_freq = self._resolve_center_freq_hz()
        bandwidth = self._resolve_bandwidth_hz(header)
        subcarrier_indices = self._resolve_subcarrier_indices(bandwidth)
        subcarrier_spacing = get_subcarrier_spacing(
            narrow_spacing=False,
            bandwidth=bandwidth,
            subcarrier_group_30=True,
        )
        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality="wifi",
                center_freq=center_freq,
                bandwidth=bandwidth,
                sample_rate=_optional_float(sample_rate),
                subcarrier_spacing=subcarrier_spacing,
                timestamp_start=float(timestamps[0]),
                subcarrier_indices=subcarrier_indices,
                reader_id=self.reader_id,
                capture_device=self.device_name,
                data_version=f"bfee_{int(header['bfee_count'])}",
            )
        )
        metadata.set_coord("time", timestamps - float(timestamps[0]), unit="s")
        metadata.set_coord(
            "subc",
            np.asarray(subcarrier_indices, dtype=np.float64),
            unit="index",
        )
        if num_tx > 0:
            metadata.set_coord("tx", np.arange(num_tx, dtype=np.float64), unit="index")
        if num_rx > 0:
            metadata.set_coord("rx", np.arange(num_rx, dtype=np.float64), unit="index")
        for field_name in self._structured_merged_extra_fields():
            if field_name in canonical_payload:
                metadata.extra[field_name] = deepcopy(canonical_payload[field_name])
        self._apply_runtime_bridge(metadata)
        metadata.modality = "wifi"
        return metadata

    def _resolve_center_freq_hz(self) -> float:
        if self.center_freq_hz is not None:
            return float(self.center_freq_hz)
        if self.channel is not None:
            try:
                return channel_to_center_freq(int(self.channel))
            except ValueError as exc:
                raise ReaderError(f"Unknown WiFi channel number: {self.channel}") from exc

        raise ReaderError(
            "IWL5300Reader requires explicit channel or center_freq_hz; "
            "the capture does not expose channel metadata."
        )

    def _resolve_bandwidth_hz(self, header: Mapping[str, object]) -> float:
        if self.bandwidth_hz is not None:
            return self._normalize_bandwidth_hz(float(self.bandwidth_hz))

        raw_rate = header.get("rate")
        if raw_rate is not None:
            return self._decode_bandwidth_hz_from_rate(int(raw_rate))

        raise ReaderError(
            "IWL5300Reader requires explicit bandwidth_hz when the header "
            "does not expose a rate bitfield."
        )

    def _decode_bandwidth_hz_from_rate(self, rate_flags: int) -> float:
        if rate_flags & _RATE_MCS_HT_MSK and rate_flags & _RATE_MCS_HT40_MSK:
            return 40.0e6
        return 20.0e6

    def _resolve_subcarrier_indices(self, bandwidth_hz: float) -> list[int]:
        normalized_bandwidth = self._normalize_bandwidth_hz(bandwidth_hz)
        if math.isclose(normalized_bandwidth, 20.0e6, rel_tol=0.0, abs_tol=1.0):
            return list(_IWL5300_20MHZ_SUBCARRIER_INDICES)
        if math.isclose(normalized_bandwidth, 40.0e6, rel_tol=0.0, abs_tol=1.0):
            return list(_IWL5300_40MHZ_SUBCARRIER_INDICES)
        raise ReaderError(
            "IWL5300Reader supports only 20 MHz or 40 MHz bandwidth for grouped CSI."
        )

    def _normalize_bandwidth_hz(self, bandwidth_hz: float) -> float:
        if math.isclose(bandwidth_hz, 20.0e6, rel_tol=0.0, abs_tol=1.0):
            return 20.0e6
        if math.isclose(bandwidth_hz, 40.0e6, rel_tol=0.0, abs_tol=1.0):
            return 40.0e6
        raise ReaderError(
            f"Unsupported IWL5300 bandwidth {bandwidth_hz / 1.0e6:.3f} MHz. "
            "Expected 20 MHz or 40 MHz."
        )

    def _build_binding_input(self, header: Mapping[str, object]) -> dict[str, object]:
        raw_fields = dict(header)
        timestamp_low = raw_fields.get("timestamp_low")
        if timestamp_low is not None and "timestamp" not in raw_fields:
            raw_fields["timestamp"] = timestamp_low
        raw_fields.setdefault("dimension_names", _CANONICAL_DIMENSION_NAMES)
        raw_fields.setdefault("data_format", _CANONICAL_DATA_FORMAT)
        return raw_fields
