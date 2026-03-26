"""Generic PCAP container helpers."""

from __future__ import annotations

import io
import numpy as np
import struct
from pathlib import Path


class PcapFrame:
    """One decoded PCAP frame with raw header and payload bytes."""

    FRAME_HEADER_DTYPE = np.dtype(
        [
            ("ts_sec", np.uint32),
            ("ts_usec", np.uint32),
            ("incl_len", np.uint32),
            ("orig_len", np.uint32),
        ]
    )

    def __init__(self, data: io.BytesIO) -> None:
        self.data = data
        self.length = 0
        self.header = self.read_header()
        self.payload_bytes = b""
        self.payload = self.read_payload()

    def read_header(self) -> np.ndarray:
        header_bytes = self.data.read(self.FRAME_HEADER_DTYPE.itemsize)
        if len(header_bytes) != self.FRAME_HEADER_DTYPE.itemsize:
            raise BufferError("Unable to read data for header")
        header = np.frombuffer(header_bytes, dtype=self.FRAME_HEADER_DTYPE)
        self.length += self.FRAME_HEADER_DTYPE.itemsize
        return header

    def read_payload(self) -> np.ndarray | None:
        if self.header is None or len(self.header["incl_len"]) == 0:
            return None
        incl_len = int(self.header["incl_len"][0])
        if incl_len <= 0:
            return None
        payload_bytes = self.data.read(incl_len)
        if payload_bytes is None:
            raise BufferError("Could not read payload")
        actual_len = len(payload_bytes)
        if actual_len == 0:
            raise BufferError("Could not read payload: empty data")
        if actual_len < incl_len:
            payload_bytes = payload_bytes + b"\x00" * (incl_len - actual_len)
        self.payload_bytes = payload_bytes
        if (incl_len % 4) == 0:
            ints_size = int(incl_len / 4)
            payload = np.array(struct.unpack(ints_size * "I", payload_bytes), dtype=np.uint32)
        else:
            ints_size = incl_len
            payload = np.array(struct.unpack(ints_size * "B", payload_bytes), dtype=np.uint8)
        self.length += incl_len
        return payload


class Pcap:
    """Minimal PCAP container reader."""

    PCAP_HEADER_DTYPE = np.dtype(
        [
            ("magic_number", np.uint32),
            ("version_major", np.uint16),
            ("version_minor", np.uint16),
            ("thiszone", np.int32),
            ("sigfigs", np.uint32),
            ("snaplen", np.uint32),
            ("network", np.uint32),
        ]
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            file_data = handle.read()
        self.data = io.BytesIO(file_data)
        self.header = self.data.read(self.PCAP_HEADER_DTYPE.itemsize)
        self.frames: list[PcapFrame] = []

    def read(self) -> None:
        while True:
            try:
                next_frame = PcapFrame(self.data)
                self.frames.append(next_frame)
            except (BufferError, struct.error):
                break

    def close(self) -> None:
        self.data.close()
        self.frames = []


__all__ = ["Pcap", "PcapFrame"]
