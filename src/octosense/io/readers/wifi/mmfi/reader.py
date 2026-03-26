"""MMFi WiFi reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as scio
import torch

from octosense.io.readers._base import CanonicalReader, ReaderError
from octosense.io.semantics.metadata import SignalMetadata
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor

_SAMPLE_RATE_HZ = 320.0
_CENTER_FREQ_HZ = 5_000_000_000.0
_BANDWIDTH_HZ = 40_000_000.0
_TIME_UNIT = "s"
_AXIS_SCHEMA = AxisSchema(("time", "subc", "tx", "rx"))


def _canonicalize_mmfi_wifi_frame(frame: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(frame, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if array.ndim != 3:
        raise ReaderError(f"MM-Fi WiFi frame must be 3D, got {array.shape}")

    shape = tuple(int(dim) for dim in array.shape)
    if shape[0] > 8 and shape[1] <= 8 and shape[2] <= 8:
        return array[None, :, :, :]
    if shape[0] <= 8 and shape[1] > 8 and shape[2] > 1:
        return np.transpose(array, (2, 1, 0))[:, :, None, :]
    if shape[0] > 1 and shape[1] <= 8 and shape[2] > 8:
        return np.transpose(array, (0, 2, 1))[:, :, None, :]
    if shape[0] > 8 and shape[1] > 1 and shape[2] <= 8:
        return np.transpose(array, (1, 0, 2))[:, :, None, :]
    raise ReaderError(f"Unsupported MM-Fi WiFi CSIamp frame shape: {array.shape}")


def _sample_context(sample_root: Path) -> dict[str, str]:
    action_dir = sample_root.parent
    subject_dir = action_dir.parent
    scene_dir = subject_dir.parent
    return {
        "scene": scene_dir.name,
        "subject": subject_dir.name,
        "action": action_dir.name,
        "sample_path": str(sample_root),
    }


class MmfiReader(CanonicalReader):
    """Reader for MM-Fi WiFi sample directories."""

    modality = "wifi"
    device_family = "mmfi"
    device_name = "Atheros CSI Tool"
    reader_version = "1.0"

    def validate_format(self, sample_root: str | Path) -> tuple[bool, str]:
        path = Path(sample_root)
        if not path.exists():
            return False, f"Sample directory not found: {path}"
        if not path.is_dir():
            return False, f"MM-Fi WiFi reader expects a sample directory, got: {path}"
        if not list(path.glob("frame*.mat")):
            return False, f"No frame*.mat files found under {path}"
        return True, ""

    def read(self, sample_root: str | Path) -> RadioTensor:
        path = Path(sample_root)
        is_valid, message = self.validate_format(path)
        if not is_valid:
            raise ReaderError(message)

        frames: list[np.ndarray] = []
        for mat_path in sorted(path.glob("frame*.mat")):
            payload = scio.loadmat(mat_path)
            if "CSIamp" not in payload:
                raise ReaderError(f"Expected CSIamp in {mat_path}")
            frames.append(_canonicalize_mmfi_wifi_frame(payload["CSIamp"]))
        if not frames:
            raise ReaderError(f"No MM-Fi WiFi frames found under {path}")

        merged = torch.from_numpy(np.concatenate(frames, axis=0)).float().contiguous()
        context = _sample_context(path)
        metadata = SignalMetadata(
            modality=self.modality,
            center_freq=_CENTER_FREQ_HZ,
            bandwidth=_BANDWIDTH_HZ,
            sample_rate=_SAMPLE_RATE_HZ,
            subcarrier_indices=list(range(int(merged.shape[1]))),
            reader_id=self.reader_id,
            capture_device=self.device_name,
            extra={
                "dataset": "MMFi",
                **context,
            },
        )
        metadata.set_coord(
            "time",
            np.arange(int(merged.shape[0]), dtype=np.float64) / _SAMPLE_RATE_HZ,
            unit=_TIME_UNIT,
        )
        metadata.set_coord("subc", np.arange(int(merged.shape[1]), dtype=np.int64), unit="index")
        metadata.set_coord("tx", np.arange(int(merged.shape[2]), dtype=np.int64), unit="index")
        metadata.set_coord("rx", np.arange(int(merged.shape[3]), dtype=np.int64), unit="index")
        return RadioTensor(merged, _AXIS_SCHEMA, metadata)
