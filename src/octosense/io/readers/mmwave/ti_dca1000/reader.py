"""TI DCA1000 EVM reader and radar configuration.

Reads raw int16 interleaved IQ samples from DCA1000 capture card.
Supports channel-interleaved format (standard DCA1000 output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from octosense.io.codecs.binary import read_int16_array
from octosense.io.profiles.mmwave import RadarConfig
from octosense.io.semantics.loader import load_reader_definition_bundle
from octosense.io.tensor import (
    RadioTensor,
    SignalMetadata,
    apply_reader_runtime_contract,
    build_reader_axis_schema,
)
from octosense.io.readers.mmwave.base import BaseRadarReader, ReaderError

_READER_DEFINITION_BUNDLE = load_reader_definition_bundle("mmwave", "ti_dca1000")
_CONFIG = _READER_DEFINITION_BUNDLE.config
_MMAP_THRESHOLD = int(_CONFIG["validation"]["mmap_threshold_bytes"])
# ---------------------------------------------------------------------------
# TI_DCA1000Reader
# ---------------------------------------------------------------------------


class TI_DCA1000Reader(BaseRadarReader):
    """Reader for TI DCA1000 EVM raw ADC binary format.

    Binary layout (``chInterleave=1``, ``iqSwapSel=0``):
        Per ADC sample: ``[I_rx0, I_rx1, ..., I_rxN, Q_rx0, Q_rx1, ..., Q_rxN]``
        Each value: int16 (2 bytes, little-endian).

    With TDM-MIMO the chirps in the binary file are ordered as:
        ``[chirp0_tx0, chirp0_tx1, ..., chirp0_txM, chirp1_tx0, ...]``
    The reader reorganises them into a virtual antenna dimension.

    Output:
        RadioTensor shape ``(frame, chirp, adc, ant)`` where ``ant = num_tx * num_rx``.

    Example::

        >>> reader = TI_DCA1000Reader()
        >>> config = RadarConfig.IWR1443()
        >>> signal = reader.read_file("adc_data.bin", config)
        >>> print(signal.axis_schema.axes)
        ('frame', 'chirp', 'adc', 'ant')
    """

    modality = "mmwave"
    device_family = "ti_dca1000"
    device_name: str = str(_CONFIG["device_name"])
    reader_version: str = str(_CONFIG["reader_version"])

    # ---- public API ----

    def read_file(
        self,
        file_path: str | Path,
        config: RadarConfig,
    ) -> RadioTensor:
        """Read entire binary file into RadioTensor."""
        path = Path(file_path)
        if not path.exists():
            raise ReaderError(f"File not found: {path}")

        file_size = path.stat().st_size

        # Infer num_frames if needed
        num_frames = config.num_frames
        if num_frames <= 0:
            num_frames = self.infer_num_frames(path, config)

        # Validate file size
        is_valid, msg = self.validate_format(path, config)
        if not is_valid:
            raise ReaderError(msg)

        # Read raw int16 data
        raw = np.asarray(read_int16_array(path, mmap_threshold_bytes=_MMAP_THRESHOLD))

        # Parse interleaved IQ → complex
        complex_data = self._parse_raw(raw, config, num_frames)

        # Build RadioTensor
        tensor_data = torch.from_numpy(complex_data)  # complex64
        metadata = self._build_metadata(config, num_frames)
        return RadioTensor.from_reader(
            tensor_data,
            build_reader_axis_schema(_READER_DEFINITION_BUNDLE),
            metadata,
        )

    def validate_format(
        self,
        file_path: str | Path,
        config: RadarConfig,
    ) -> tuple[bool, str]:
        """Validate file size against config."""
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {path}"

        file_size = path.stat().st_size
        samples_per_frame = (
            config.num_chirps_per_frame * config.num_tx *
            config.num_rx * config.num_adc_samples
        )
        frame_bytes = samples_per_frame * 4  # 2 int16 per IQ pair = 4 bytes

        if frame_bytes == 0:
            return False, "Invalid config: frame size is 0 bytes"

        if file_size % frame_bytes != 0:
            actual_frames_f = file_size / frame_bytes
            closest = int(actual_frames_f)
            return False, (
                f"File size mismatch for '{path.name}':\n"
                f"  Actual: {file_size:,} bytes\n"
                f"  Frame size: {frame_bytes:,} bytes "
                f"({config.num_chirps_per_frame} chirps * {config.num_tx} tx * "
                f"{config.num_rx} rx * {config.num_adc_samples} adc * 4 bytes)\n"
                f"  Detected: {actual_frames_f:.2f} frames (not integer)\n"
                f"  Suggestion: Try num_frames={closest} or adjust "
                f"num_chirps_per_frame / num_tx / num_adc_samples in RadarConfig"
            )

        num_frames_detected = file_size // frame_bytes
        if config.num_frames > 0 and num_frames_detected != config.num_frames:
            return False, (
                f"File size mismatch for '{path.name}':\n"
                f"  Actual: {file_size:,} bytes ({num_frames_detected} frames detected)\n"
                f"  Expected: {config.expected_file_size():,} bytes "
                f"({config.num_frames} frames per config)\n"
                f"  Suggestion: Set num_frames={num_frames_detected} in RadarConfig"
            )

        return True, ""

    def infer_num_frames(
        self,
        file_path: str | Path,
        config: RadarConfig,
    ) -> int:
        """Infer number of frames from file size and config."""
        path = Path(file_path)
        file_size = path.stat().st_size
        samples_per_frame = (
            config.num_chirps_per_frame * config.num_tx *
            config.num_rx * config.num_adc_samples
        )
        frame_bytes = samples_per_frame * 4

        if frame_bytes == 0:
            raise ReaderError("Cannot infer frames: frame size is 0 bytes")

        if file_size % frame_bytes != 0:
            raise ReaderError(
                f"File size {file_size:,} is not divisible by frame size {frame_bytes:,}. "
                f"Check RadarConfig parameters."
            )

        return file_size // frame_bytes

    # ---- internal helpers ----

    def _parse_raw(
        self, raw: np.ndarray, config: RadarConfig, num_frames: int
    ) -> np.ndarray:
        """Parse raw int16 array into complex64 shaped (frame, chirp, adc, ant).

        DCA1000 channel-interleaved format:
            Per ADC sample: [I_rx0, I_rx1, ..., I_rxN, Q_rx0, Q_rx1, ..., Q_rxN]
        """
        num_rx = config.num_rx
        num_tx = config.num_tx
        num_adc = config.num_adc_samples
        num_chirps = config.num_chirps_per_frame

        # Total raw chirps in file = num_frames * num_chirps * num_tx (TDM interleaved)
        total_raw_chirps = num_frames * num_chirps * num_tx

        # Each raw chirp has num_adc * num_rx * 2 (I+Q) int16 values
        expected_len = total_raw_chirps * num_adc * num_rx * 2
        raw_arr = np.array(raw[:expected_len], dtype=np.float32)

        # DCA1000 channel-interleaved layout per ADC sample:
        # [I_rx0, I_rx1, ..., I_rxN, Q_rx0, Q_rx1, ..., Q_rxN]
        # Shape to (..., iq, rx) then form complex I + jQ.
        raw_arr = raw_arr.reshape(total_raw_chirps, num_adc, 2, num_rx)
        complex_data = raw_arr[:, :, 0, :] + 1j * \
            raw_arr[:, :, 1, :]  # (raw_chirps, adc, rx)

        # Reshape with TDM-MIMO: group by frame, then by chirp loops
        # Raw order: frame0[chirp0_tx0, chirp0_tx1, ..., chirp0_txM, chirp1_tx0, ...]
        complex_data = complex_data.reshape(
            num_frames, num_chirps, num_tx, num_adc, num_rx)

        # Merge TX and RX into virtual antenna dimension: (frame, chirp, adc, num_tx*num_rx)
        complex_data = complex_data.transpose(
            0, 1, 3, 2, 4)  # (frame, chirp, adc, tx, rx)
        complex_data = complex_data.reshape(
            num_frames, num_chirps, num_adc, num_tx * num_rx)

        return complex_data.astype(np.complex64)

    def _build_metadata(self, config: RadarConfig, num_frames: int) -> SignalMetadata:
        """Build SignalMetadata from RadarConfig."""
        metadata = SignalMetadata(
            modality="mmwave",
            center_freq=config.center_freq,
            bandwidth=config.bandwidth,
            sample_rate=config.sample_rate,
            chirp_period=config.chirp_period,
            reader_id=self.reader_id,
            capture_device=self.device_name,
        )
        metadata.set_coord("frame", np.arange(num_frames, dtype=np.int64), unit="")
        metadata.set_coord(
            "chirp",
            np.arange(config.num_chirps_per_frame, dtype=np.float64) * config.chirp_period,
            unit="s",
        )
        metadata.set_coord(
            "adc",
            np.arange(config.num_adc_samples, dtype=np.float64) / config.sample_rate,
            unit="s",
        )
        metadata.set_coord("ant", np.arange(config.num_virtual_antennas, dtype=np.int64), unit="")
        apply_reader_runtime_contract(
            metadata,
            _READER_DEFINITION_BUNDLE,
            raw_payload={
                "num_tx": config.num_tx,
                "num_rx": config.num_rx,
                "num_adc_samples": config.num_adc_samples,
                "num_chirps_per_frame": config.num_chirps_per_frame,
                "num_frames": num_frames,
                "sample_rate": config.sample_rate,
                "center_freq": config.center_freq,
                "bandwidth": config.bandwidth,
                "chirp_period": config.chirp_period,
                "antenna_positions": list(config.antenna_positions)
                if config.antenna_positions is not None
                else None,
            },
        )
        return metadata
