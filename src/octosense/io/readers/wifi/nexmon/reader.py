"""Nexmon WiFi CSI reader implementation.

Parses CSI data from Nexmon-modified Broadcom devices (.pcap files).
"""

import logging
import struct
from pathlib import Path

from octosense.io.codecs.pcap import Pcap, PcapFrame
from octosense.io.profiles.wifi import (
    build_centered_wifi_subcarrier_indices,
    channel_to_center_freq,
    resolve_wifi_subcarrier_spacing_hz,
)
from octosense.io.readers.wifi.base import (
    BaseWiFiReader,
    ReaderError,
    _compact_metadata_kwargs,
)
from octosense.io.semantics.loader import load_reader_definition_bundle
from octosense.io.tensor import RadioTensor, SignalMetadata, build_wifi_csi_axis_schema
import numpy as np
import torch

logger = logging.getLogger(__name__)

_READER_DEFINITION_BUNDLE = load_reader_definition_bundle("wifi", "nexmon")
_CONFIG = _READER_DEFINITION_BUNDLE.config
FILE_EXTENSIONS = tuple(_CONFIG["file_extensions"])
_PCAP_CONFIG = _CONFIG["pcap"]
_PCAP_CHIPS = {str(key): str(value) for key, value in _PCAP_CONFIG["chips"].items()}
_PCAP_BW_SIZES = {
    int(key): int(value) for key, value in _PCAP_CONFIG["bw_sizes"].items()
}
_PCAP_HOFFSET_WORDS = int(_PCAP_CONFIG["hoffset_words"])
_PCAP_BW_SUBCARRIERS = {
    int(key): int(value) for key, value in _PCAP_CONFIG["bw_subcarriers"].items()
}
_PCAP_PAYLOAD_OFFSETS = {
    str(key): int(value) for key, value in _PCAP_CONFIG["payload_offsets"].items()
}
_PCAP_FLOAT_UNPACK = {
    str(chip): {str(key): int(value) for key, value in payload.items()}
    for chip, payload in _PCAP_CONFIG["float_unpack"].items()
}
_CANONICAL_DIMENSION_NAMES = tuple(build_wifi_csi_axis_schema().axes)
_CANONICAL_DATA_FORMAT = "complex64[time,subc,tx,rx]"


def _hex_to_mac_string(hex_str: str) -> str:
    return ":".join(hex_str[i : i + 2] for i in range(0, 12, 2)).upper()


def _channel_spec_to_channel(channel_spec: str) -> int | None:
    """Decode Broadcom chanspec hex into the canonical channel number."""
    if len(channel_spec) != 4:
        return None
    try:
        chanspec = int.from_bytes(bytes.fromhex(channel_spec), byteorder="little")
    except ValueError:
        return None
    return chanspec & 0xFF


def _parse_nexmon_payload_header(frame: PcapFrame) -> dict[str, object] | None:
    raw = frame.payload_bytes
    if len(raw) < 64:
        return None
    payload = raw[42:64]
    payload_header: dict[str, object] = {"magic_bytes": payload[:2]}
    if payload[:4] == b"\x11\x11\x11\x11":
        payload_header["rssi"] = -1
        payload_header["frame_control"] = -1
    else:
        payload_header["rssi"] = struct.unpack("b", payload[2:3])[0]
        payload_header["frame_control"] = struct.unpack("B", payload[3:4])[0]

    payload_header["source_mac"] = _hex_to_mac_string(payload[4:10].hex())
    payload_header["sequence_no"] = int.from_bytes(payload[10:12], byteorder="little")

    core_spatial = int.from_bytes(payload[12:14], byteorder="little")
    if core_spatial > 63:
        core_spatial = int.from_bytes(payload[12:14], byteorder="big")
    core_bits = bin(core_spatial)[2:].zfill(6)
    payload_header["core"] = int(core_bits[3:6], 2)
    payload_header["spatial_stream"] = int(core_bits[:3], 2)

    payload_header["channel_spec"] = payload[14:16].hex()
    chip_identifier = payload[16:18].hex()
    payload_header["chip"] = _PCAP_CHIPS.get(chip_identifier, "UNKNOWN")
    return payload_header


def _resolve_frame_bandwidth(frame: PcapFrame) -> int | None:
    if frame.header is None or frame.payload is None:
        return None
    given_size = int(frame.header["orig_len"][0]) - (_PCAP_HOFFSET_WORDS - 1) * 4
    return _PCAP_BW_SIZES.get(given_size)


def _unpack_float_acphy(
    nbits: int,
    autoscale: int,
    shft: int,
    fmt: int,
    nman: int,
    nexp: int,
    nfft: int,
    H: np.ndarray,
) -> np.ndarray:
    k_tof_unpack_sgn_mask = 1 << 31

    He = [0] * nfft
    Hout = [0] * nfft * 2

    iq_mask = (1 << (nman - 1)) - 1
    e_mask = (1 << nexp) - 1
    e_p = 1 << (nexp - 1)
    sgnr_mask = 1 << (nexp + 2 * nman - 1)
    sgni_mask = sgnr_mask >> nman
    e_zero = -nman

    out = np.zeros((nfft * 2), dtype=np.int64)
    n_out = nfft << 1
    e_shift = 1
    maxbit = -e_p

    for i in range(len(H)):
        hi = int(H[i])
        vi = (hi >> (nexp + nman)) & iq_mask
        vq = (hi >> nexp) & iq_mask
        e = hi & e_mask

        if e >= e_p:
            e -= e_p << 1

        He[i] = e
        x = vi | vq

        if autoscale and x:
            m = 0xFFFF0000
            b = 0xFFFF
            s = 16
            while s > 0:
                if x & m:
                    e += s
                    x >>= s
                s >>= 1
                m = (m >> s) & b
                b >>= s
            if e > maxbit:
                maxbit = e

        if hi & sgnr_mask:
            vi |= k_tof_unpack_sgn_mask
        if hi & sgni_mask:
            vq |= k_tof_unpack_sgn_mask

        Hout[i << 1] = vi
        Hout[(i << 1) + 1] = vq

    shft = nbits - maxbit
    for i in range(n_out):
        e = He[i >> e_shift] + shft
        vi = int(Hout[i])
        sgn = 1
        if vi & k_tof_unpack_sgn_mask:
            sgn = -1
            vi &= ~k_tof_unpack_sgn_mask
        if e < e_zero:
            vi = 0
        elif e < 0:
            vi >>= -e
        else:
            vi <<= e
        out[i] = sgn * vi
    return out


class NexmonReader(BaseWiFiReader):
    """Nexmon CSI reader."""

    modality = "wifi"
    device_family = "nexmon"
    device_name = str(_CONFIG["device_name"])
    reader_version = str(_CONFIG["reader_version"])
    BW_SUBS = _PCAP_BW_SUBCARRIERS

    def __init__(self) -> None:
        super().__init__()

    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        """Validate file format for Nexmon reader.

        Returns:
            Tuple of (is_valid, error_message). error_message is empty when valid.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        if file_path.suffix.lower() not in FILE_EXTENSIONS:
            return False, (
                f"Invalid file extension: {file_path.suffix}. "
                f"NexmonReader expects {', '.join(FILE_EXTENSIONS)} files."
            )
        return True, ""

    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        file_path = Path(file_path)
        is_valid, msg = self.validate_format(file_path)
        if not is_valid:
            raise ReaderError(msg)

        pcap = Pcap(str(file_path))
        pcap.read()

        signals: list[RadioTensor] = []
        resolved_bandwidth: int | None = None
        skipped_frames = 0
        for frame_index, frame in enumerate(pcap.frames):
            try:
                payload_header = _parse_nexmon_payload_header(frame)
                if payload_header is None:
                    skipped_frames += 1
                    continue
                frame_bandwidth = _resolve_frame_bandwidth(frame)
                if frame_bandwidth is None:
                    skipped_frames += 1
                    continue
                if resolved_bandwidth is None:
                    resolved_bandwidth = frame_bandwidth
                if frame_bandwidth != resolved_bandwidth:
                    skipped_frames += 1
                    continue
                signal = self._read_bfee(frame, payload_header, resolved_bandwidth)
            except Exception as exc:
                raise ReaderError(
                    "Malformed Nexmon PCAP frame",
                    offset=frame_index,
                    context={"frame_index": frame_index, "detail": str(exc)},
                ) from exc
            if signal is not None:
                signals.append(signal)

        pcap.close()

        if not signals:
            # If we had frames that all failed parsing or all were skipped, report malformed
            if pcap.frames or skipped_frames > 0:
                raise ReaderError(
                    "Malformed Nexmon PCAP frame",
                    offset=0,
                    context={"frame_index": 0, "detail": "All frames failed parsing or were malformed"},
                )
            raise ReaderError(f"No valid CSI frames found in {file_path}.")
        self._assign_stream_sample_rate(signals)
        return signals

    def _read_bfee(
        self,
        pcap_frame: PcapFrame,
        payload_header: dict[str, object],
        bandwidth: int,
    ) -> RadioTensor | None:
        if pcap_frame is None or pcap_frame.header is None or pcap_frame.payload is None:
            return None

        usecs = pcap_frame.header["ts_usec"][0] / 1e6
        timestamp = float(pcap_frame.header["ts_sec"][0]) + usecs

        data = pcap_frame.payload
        chip_type = payload_header.get("chip", "UNKNOWN")

        if chip_type in ["4339", "43455c0"]:
            data = data.astype(np.int16)
            data = data[_PCAP_PAYLOAD_OFFSETS["int16_start"] :]
        elif chip_type in _PCAP_FLOAT_UNPACK:
            nfft = int(bandwidth * 3.2)
            unpack_cfg = _PCAP_FLOAT_UNPACK[chip_type]
            float_start = _PCAP_PAYLOAD_OFFSETS["float_start"]
            data = data[float_start : float_start + nfft]
            data = _unpack_float_acphy(
                unpack_cfg["nbits"],
                unpack_cfg["autoscale"],
                unpack_cfg["shft"],
                unpack_cfg["fmt"],
                unpack_cfg["nman"],
                unpack_cfg["nexp"],
                nfft,
                data,
            )
        else:
            return None

        if len(data) % 2 != 0:
            data = data[:-1]
        if len(data) < 2:
            return None

        csi_data = data.reshape(-1, 2)
        csi = csi_data.astype(np.float32).view(np.complex64).flatten()

        subc_expected = self.BW_SUBS.get(bandwidth, csi.size)
        subc = subc_expected if csi.size >= subc_expected else csi.size
        if subc <= 0:
            return None
        csi = csi[:subc]

        csi_matrix = csi.reshape(subc, 1, 1)
        data_tensor = torch.from_numpy(csi_matrix).unsqueeze(0)

        channel = _channel_spec_to_channel(str(payload_header.get("channel_spec", "")))
        center_freq = None
        if channel is not None:
            try:
                center_freq = channel_to_center_freq(channel)
            except ValueError:
                center_freq = None

        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality="wifi",
                center_freq=center_freq,
                bandwidth=float(bandwidth) * 1e6,
                subcarrier_spacing=resolve_wifi_subcarrier_spacing_hz(),
                timestamp_start=timestamp,
                subcarrier_indices=build_centered_wifi_subcarrier_indices(subc).tolist(),
                reader_id=self.reader_id,
                capture_device=self.device_name,
                data_version=f"seq_{payload_header.get('sequence_no', 0)}",
            )
        )

        metadata.set_coord("time", np.array([0.0]), unit="s")
        metadata.set_coord("subc", build_centered_wifi_subcarrier_indices(subc), unit="index")

        binding_input: dict[str, object] = {
            "dimension_names": _CANONICAL_DIMENSION_NAMES,
            "data_format": _CANONICAL_DATA_FORMAT,
            "rssi": payload_header.get("rssi", 0),
            "mac_addr": payload_header.get("source_mac", ""),
            "sequence_no": payload_header.get("sequence_no", 0),
            "frame_control": payload_header.get("frame_control", 0),
            "core": payload_header.get("core", 0),
            "spatial_stream": payload_header.get("spatial_stream", 0),
            "chip": chip_type,
        }
        if channel is not None:
            binding_input["channel"] = channel

        self._finalize_runtime_contract(metadata, raw_payload=binding_input)

        return RadioTensor.from_reader(
            data_tensor,
            build_wifi_csi_axis_schema(),
            metadata=metadata,
        )
