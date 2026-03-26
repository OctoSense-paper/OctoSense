"""IWLMVM WiFi CSI reader implementation.

Parses CSI data from Intel iwlmvm-based devices (e.g., AX200/AX210).
Runtime semantics expose canonical `timestamp` plus the explicit special
timestamp surface `timestamp_nano`.
"""

import logging
import struct
import warnings
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
)
from octosense.io.profiles.wifi import (
    build_centered_wifi_subcarrier_indices,
    channel_to_center_freq,
    resolve_wifi_subcarrier_spacing_hz,
)

logger = logging.getLogger(__name__)

_READER_DEFINITION_BUNDLE = load_reader_definition_bundle("wifi", "iwlmvm")
_CONFIG = _READER_DEFINITION_BUNDLE.config
LENGTH_STRUCT = struct.Struct(_CONFIG["length_struct"])
HEADER_STRUCT = struct.Struct(_CONFIG["header_struct"])
FILE_EXTENSIONS = tuple(_CONFIG["file_extensions"])
BIT_FIELDS = {
    str(key): int(value) for key, value in _CONFIG["bit_fields"].items()
}
RATE_HT_MCS_CODE_MASK = BIT_FIELDS["ht_mcs_code_mask"]
RATE_MCS_MOD_TYPE_POS = BIT_FIELDS["mod_type_pos"]
RATE_MCS_MOD_TYPE_MSK = BIT_FIELDS["mod_type_mask"]
RATE_MCS_CHAN_WIDTH_POS = BIT_FIELDS["chan_width_pos"]
RATE_MCS_CHAN_WIDTH_MSK = BIT_FIELDS["chan_width_mask"]
RATE_MCS_ANT_A_MSK = BIT_FIELDS["ant_a_mask"]
RATE_MCS_ANT_B_MSK = BIT_FIELDS["ant_b_mask"]
RATE_MCS_LDPC_MSK = BIT_FIELDS["ldpc_mask"]
RATE_MCS_SS_MSK = BIT_FIELDS["ss_mask"]
RATE_MCS_BEAMFORMING_MSK = BIT_FIELDS["beamforming_mask"]
CHANNEL_WIDTH = {
    int(key): int(value) for key, value in _CONFIG["channel_width_mhz_by_flag"].items()
}
RATE_FORMAT = {
    int(key): str(value) for key, value in _CONFIG["rate_format_by_mod_type"].items()
}
_CANONICAL_DIMENSION_NAMES = tuple(build_wifi_csi_axis_schema().axes)
_CANONICAL_DATA_FORMAT = "complex64[time,subc,tx,rx]"


class IWLMVMReader(BaseWiFiReader):
    """Intel iwlmvm CSI reader."""

    modality = "wifi"
    device_family = "iwlmvm"
    device_name = str(_CONFIG["device_name"])
    reader_version = str(_CONFIG["reader_version"])

    def __init__(
        self,
        channel: int | None = None,
        *,
        merge_policy: str = "strict",
    ) -> None:
        """Initialize IWLMVM reader.

        Args:
            channel: WiFi channel number for center frequency calculation.
                     If None, center frequency stays unset unless another source supplies it.
            merge_policy: `strict` rejects mixed payload shapes, while `dominant`
                keeps the most frequent payload shape before merging.
        """
        super().__init__()
        self.channel = channel
        if merge_policy not in {"strict", "dominant"}:
            raise ValueError("merge_policy must be 'strict' or 'dominant'")
        self.merge_policy = merge_policy

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        """Validate file format for IWLMVM reader.

        Returns:
            Tuple of (is_valid, error_message). error_message is empty when valid.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        if file_path.suffix.lower() not in FILE_EXTENSIONS:
            return False, (
                f"Invalid file extension: {file_path.suffix}. "
                f"IWLMVMReader expects {', '.join(FILE_EXTENSIONS)} files."
            )

        try:
            with open(file_path, "rb") as f:
                header = f.read(LENGTH_STRUCT.size)
                if len(header) != LENGTH_STRUCT.size:
                    return False, "File too short to be a valid IWLMVM CSI file."
                _, hdr_len, _, data_len = LENGTH_STRUCT.unpack(header)
                if hdr_len <= 0 or data_len <= 0:
                    return False, "Invalid header/data lengths in IWLMVM file."
            return True, ""
        except Exception as e:
            return False, f"Failed to validate file: {e}"

    def read_file(
        self,
        file_path: str | Path,
        *,
        src_mac: str | None = None,
        n_subc: int | None = None,
        n_tx: int | None = None,
        n_rx: int | None = None,
        bw: int | None = None,
        bandwidth_mhz: int | None = None,
    ) -> list[RadioTensor]:
        file_path = Path(file_path)
        is_valid, msg = self.validate_format(file_path)
        if not is_valid:
            raise ReaderError(msg)

        signals: list[RadioTensor] = []
        with open(file_path, "rb") as f:
            packet_index = 0
            while True:
                offset = f.tell()
                lengths = f.read(LENGTH_STRUCT.size)
                if len(lengths) != LENGTH_STRUCT.size:
                    break

                timestamp_raw, hdr_len, _, data_len = LENGTH_STRUCT.unpack(lengths)
                header_bytes = f.read(hdr_len)
                if len(header_bytes) != hdr_len:
                    break
                data = f.read(data_len)
                if len(data) != data_len:
                    raise ReaderError(
                        "Truncated IWLMVM packet payload",
                        offset=offset,
                        context={
                            "packet_index": packet_index,
                            "expected": data_len,
                            "actual": len(data),
                        },
                    )

                signal = self._parse_packet(timestamp_raw, header_bytes, data)
                if signal is not None:
                    signals.append(signal)
                packet_index += 1

        if not signals:
            raise ReaderError(f"No valid CSI frames found in {file_path}.")
        self._warn_if_mixed_payload_shapes(signals, file_path=file_path)
        selected_signals, _selection_note = self._select_first_matching_class(
            signals,
            file_path=file_path,
            src_mac=src_mac,
            n_subc=n_subc,
            n_tx=n_tx,
            n_rx=n_rx,
            bw=bw,
            bandwidth_mhz=bandwidth_mhz,
        )
        self._assign_stream_sample_rate(selected_signals)
        return selected_signals

    def read(
        self,
        file_path: str | Path,
        *,
        src_mac: str | None = None,
        n_subc: int | None = None,
        n_tx: int | None = None,
        n_rx: int | None = None,
        bw: int | None = None,
        bandwidth_mhz: int | None = None,
    ) -> RadioTensor:
        """Merge packet tensors only when the frame payload contract is stable."""
        file_path = Path(file_path)
        signals = self.read_file(
            file_path,
            src_mac=src_mac,
            n_subc=n_subc,
            n_tx=n_tx,
            n_rx=n_rx,
            bw=bw,
            bandwidth_mhz=bandwidth_mhz,
        )
        selection_note: dict[str, object] | None = None
        if any(value is not None for value in (src_mac, n_subc, n_tx, n_rx, bw, bandwidth_mhz)):
            selected_src_mac, selected_n_subc, selected_n_tx, selected_n_rx, selected_bw = (
                self._signal_class_signature(signals[0])
            )
            selection_note = {
                "selection_src_mac": selected_src_mac,
                "selection_n_subc": selected_n_subc,
                "selection_n_tx": selected_n_tx,
                "selection_n_rx": selected_n_rx,
                "selection_bandwidth_mhz": selected_bw,
                "selection_packet_count": len(signals),
            }
        if self.merge_policy == "dominant":
            signals, dominant_note = self._select_dominant_payload_shape(
                signals,
                file_path=file_path,
            )
            if dominant_note:
                if selection_note is None:
                    selection_note = dict(dominant_note)
                else:
                    selection_note.update(dominant_note)
        else:
            self._validate_mergeable_packets(signals, file_path=file_path)
        merged = self._merge_signals_to_tensor(signals, file_path=file_path)
        if selection_note:
            merged.metadata.extra.update(selection_note)
        return merged

    def _merge_timestamps_seconds(self, signals: list[RadioTensor]) -> torch.Tensor:
        local_timestamps = [
            float(signal.metadata.extra.get("iwlmvm_timestamp_nano", 0)) * 1.0e-9
            for signal in signals
        ]
        if any(timestamp > 0.0 for timestamp in local_timestamps):
            return torch.tensor(local_timestamps, dtype=torch.float64)
        return super()._merge_timestamps_seconds(signals)

    def _warn_if_mixed_payload_shapes(
        self,
        signals: list[RadioTensor],
        *,
        file_path: Path,
    ) -> None:
        shape_counts: dict[tuple[int, ...], int] = {}
        for signal in signals:
            shape = tuple(int(dim) for dim in signal.shape[1:])
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        if len(shape_counts) <= 1:
            return
        warnings.warn(
            (
                "IWLMVM capture contains multiple CSI payload shapes; "
                f"shape_counts={dict(sorted(shape_counts.items()))} for {file_path}"
            ),
            UserWarning,
            stacklevel=2,
        )

    def _resolve_bandwidth_filter(
        self,
        *,
        bw: int | None,
        bandwidth_mhz: int | None,
    ) -> int | None:
        if bw is not None and bandwidth_mhz is not None and int(bw) != int(bandwidth_mhz):
            raise ValueError("bw and bandwidth_mhz must match when both are provided")
        if bw is not None:
            return int(bw)
        if bandwidth_mhz is not None:
            return int(bandwidth_mhz)
        return None

    def _signal_class_signature(
        self,
        signal: RadioTensor,
    ) -> tuple[str, int, int, int, int]:
        extra = signal.metadata.extra
        return (
            str(extra.get("mac_src", "")).lower(),
            int(extra.get("n_subc", signal.shape[1])),
            int(extra.get("n_tx", signal.shape[2])),
            int(extra.get("n_rx", signal.shape[3])),
            int(extra.get("bandwidth", 0)),
        )

    def _signal_matches_filters(
        self,
        signal: RadioTensor,
        *,
        src_mac: str | None,
        n_subc: int | None,
        n_tx: int | None,
        n_rx: int | None,
        bandwidth_mhz: int | None,
    ) -> bool:
        signal_src_mac, signal_subc, signal_n_tx, signal_n_rx, signal_bw = (
            self._signal_class_signature(signal)
        )
        if src_mac is not None and signal_src_mac != str(src_mac).strip().lower():
            return False
        if n_subc is not None and signal_subc != int(n_subc):
            return False
        if n_tx is not None and signal_n_tx != int(n_tx):
            return False
        if n_rx is not None and signal_n_rx != int(n_rx):
            return False
        if bandwidth_mhz is not None and signal_bw != int(bandwidth_mhz):
            return False
        return True

    def _select_first_matching_class(
        self,
        signals: list[RadioTensor],
        *,
        file_path: Path,
        src_mac: str | None,
        n_subc: int | None,
        n_tx: int | None,
        n_rx: int | None,
        bw: int | None,
        bandwidth_mhz: int | None,
    ) -> tuple[list[RadioTensor], dict[str, object] | None]:
        resolved_bw = self._resolve_bandwidth_filter(bw=bw, bandwidth_mhz=bandwidth_mhz)
        if all(
            value is None
            for value in (src_mac, n_subc, n_tx, n_rx, resolved_bw)
        ):
            return signals, None

        selected_signature: tuple[str, int, int, int, int] | None = None
        for signal in signals:
            if self._signal_matches_filters(
                signal,
                src_mac=src_mac,
                n_subc=n_subc,
                n_tx=n_tx,
                n_rx=n_rx,
                bandwidth_mhz=resolved_bw,
            ):
                selected_signature = self._signal_class_signature(signal)
                break

        if selected_signature is None:
            raise ReaderError(
                "No IWLMVM CSI class matched the requested filters.",
                context={
                    "file_path": str(file_path),
                    "src_mac": src_mac,
                    "n_subc": n_subc,
                    "n_tx": n_tx,
                    "n_rx": n_rx,
                    "bandwidth_mhz": resolved_bw,
                },
            )

        selected_signals = [
            signal
            for signal in signals
            if self._signal_class_signature(signal) == selected_signature
        ]
        selected_src_mac, selected_n_subc, selected_n_tx, selected_n_rx, selected_bw = (
            selected_signature
        )
        return selected_signals, {
            "selection_src_mac": selected_src_mac,
            "selection_n_subc": selected_n_subc,
            "selection_n_tx": selected_n_tx,
            "selection_n_rx": selected_n_rx,
            "selection_bandwidth_mhz": selected_bw,
            "selection_packet_count": len(selected_signals),
        }

    def _parse_packet(
        self, timestamp_raw: bytes, header_bytes: bytes, data_bytes: bytes
    ) -> RadioTensor | None:
        try:
            headers = HEADER_STRUCT.unpack(header_bytes[: HEADER_STRUCT.size])
        except struct.error:
            return None

        csi_size = headers[0]
        ftm_clock = headers[1]
        n_rx = headers[2]
        n_tx = headers[3]
        n_link = headers[4]
        n_subc = headers[5]
        rssi_a = headers[6]
        rssi_b = headers[7]
        mac_src = headers[8]
        seq = headers[9]
        header_timestamp = headers[10]
        rate_flags = headers[11]

        # Keep the outer ASCII capture counter as the explicit special
        # timestamp source.
        try:
            local_timestamp_nano = int(timestamp_raw.decode("ascii").strip("\x00"))
        except Exception:
            local_timestamp_nano = 0

        # Parse CSI data
        complex_vals = np.frombuffer(data_bytes, dtype="<i2")
        complex_vals = complex_vals[::2] + 1j * complex_vals[1::2]
        expected = n_tx * n_rx * n_subc
        if complex_vals.size < expected:
            return None
        complex_vals = complex_vals[:expected]
        csi_matrix = complex_vals.reshape(n_tx, n_rx, n_subc).transpose(2, 0, 1)

        data = torch.from_numpy(csi_matrix.astype(np.complex64)).unsqueeze(0)

        metadata = self._build_metadata(
            local_timestamp_nano=local_timestamp_nano,
            header_timestamp=header_timestamp,
            n_rx=n_rx,
            n_tx=n_tx,
            n_link=n_link,
            n_subc=n_subc,
            rate_flags=rate_flags,
            rssi_a=rssi_a,
            rssi_b=rssi_b,
            mac_src=mac_src,
            seq=seq,
            csi_size=csi_size,
            ftm_clock=ftm_clock,
        )
        return RadioTensor.from_reader(data, build_wifi_csi_axis_schema(), metadata=metadata)

    def _validate_mergeable_packets(
        self,
        signals: list[RadioTensor],
        *,
        file_path: Path,
    ) -> None:
        payload_shapes = {tuple(int(dim) for dim in signal.shape[1:]) for signal in signals}
        if len(payload_shapes) <= 1:
            return

        subcarrier_counts = sorted({int(signal.shape[1]) for signal in signals})
        tx_counts = sorted({int(signal.shape[2]) for signal in signals})
        rx_counts = sorted({int(signal.shape[3]) for signal in signals})
        raise ReaderError(
            "IWLMVM read() cannot merge packets with inconsistent payload shapes.",
            context={
                "file_path": str(file_path),
                "payload_shapes": sorted(payload_shapes),
                "subcarrier_counts": subcarrier_counts,
                "tx_counts": tx_counts,
                "rx_counts": rx_counts,
            },
        )

    def _select_dominant_payload_shape(
        self,
        signals: list[RadioTensor],
        *,
        file_path: Path,
    ) -> tuple[list[RadioTensor], dict[str, object] | None]:
        payload_shapes: dict[tuple[int, ...], list[RadioTensor]] = {}
        for signal in signals:
            shape = tuple(int(dim) for dim in signal.shape[1:])
            payload_shapes.setdefault(shape, []).append(signal)
        if len(payload_shapes) <= 1:
            return signals, None

        dominant_shape, dominant_signals = max(
            payload_shapes.items(),
            key=lambda item: (len(item[1]), *item[0]),
        )
        dropped_packets = len(signals) - len(dominant_signals)
        logger.warning(
            "IWLMVM dominant merge_policy kept payload shape %s and dropped %s mixed-shape packets for %s",
            dominant_shape,
            dropped_packets,
            file_path,
        )
        return dominant_signals, {
            "merge_policy": "dominant",
            "selected_payload_shape": list(dominant_shape),
            "dropped_packets": int(dropped_packets),
        }

    def _normalize_mac_src(self, mac_src: bytes | bytearray | memoryview) -> str:
        raw_mac = bytes(mac_src)
        return ":".join(f"{byte:02X}" for byte in raw_mac)

    def _build_metadata(
        self,
        local_timestamp_nano: int,
        header_timestamp: int,
        n_rx: int,
        n_tx: int,
        n_link: int,
        n_subc: int,
        rate_flags: int,
        rssi_a: int,
        rssi_b: int,
        mac_src: bytes,
        seq: int,
        csi_size: int,
        ftm_clock: int,
    ) -> SignalMetadata:
        chan_width_val = (rate_flags & RATE_MCS_CHAN_WIDTH_MSK) >> RATE_MCS_CHAN_WIDTH_POS
        bandwidth_mhz = CHANNEL_WIDTH.get(chan_width_val, 20)
        bandwidth = float(bandwidth_mhz) * 1e6
        rate_format = RATE_FORMAT.get(
            (rate_flags & RATE_MCS_MOD_TYPE_MSK) >> RATE_MCS_MOD_TYPE_POS,
            "UNKNOWN",
        )
        mcs = int(rate_flags & RATE_HT_MCS_CODE_MASK)
        antenna_a = bool(rate_flags & RATE_MCS_ANT_A_MSK)
        antenna_b = bool(rate_flags & RATE_MCS_ANT_B_MSK)
        ldpc = bool(rate_flags & RATE_MCS_LDPC_MSK)
        ss = int(bool(rate_flags & RATE_MCS_SS_MSK)) + 1
        beamforming = bool(rate_flags & RATE_MCS_BEAMFORMING_MSK)

        # Use channel to compute center frequency
        center_freq: float | None = None
        if self.channel is not None:
            try:
                center_freq = channel_to_center_freq(self.channel)
            except ValueError:
                center_freq = None

        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality="wifi",
                center_freq=center_freq,
                bandwidth=bandwidth,
                subcarrier_spacing=resolve_wifi_subcarrier_spacing_hz(rate_format=rate_format),
                subcarrier_indices=build_centered_wifi_subcarrier_indices(n_subc).tolist(),
                reader_id=self.reader_id,
                capture_device=self.device_name,
                data_version=f"seq_{seq}",
            )
        )

        metadata.set_coord("time", np.array([0.0]), unit="s")
        metadata.set_coord("subc", build_centered_wifi_subcarrier_indices(n_subc), unit="index")

        self._finalize_runtime_contract(
            metadata,
            raw_payload={
                "timestamp_nano": local_timestamp_nano,
                "timestamp": header_timestamp,
                "csi_size": csi_size,
                "ftm_clock": ftm_clock,
                "n_rx": n_rx,
                "n_tx": n_tx,
                "n_link": n_link,
                "n_subc": n_subc,
                "rssi_a": rssi_a,
                "rssi_b": rssi_b,
                "mac_src": self._normalize_mac_src(mac_src),
                "seq": seq,
                "rate_flags": rate_flags,
                "rate_format": rate_format,
                "bandwidth": bandwidth_mhz,
                "mcs": mcs,
                "antenna_a": antenna_a,
                "antenna_b": antenna_b,
                "ldpc": ldpc,
                "ss": ss,
                "beamforming": beamforming,
                "dimension_names": _CANONICAL_DIMENSION_NAMES,
                "data_format": _CANONICAL_DATA_FORMAT,
            },
        )

        return metadata
