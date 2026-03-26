"""MMFi mmWave reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from octosense.io.readers._base import CanonicalReader, ReaderError
from octosense.io.semantics.metadata import SignalMetadata
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor

_CENTER_FREQ_HZ = 60_000_000_000.0
_BANDWIDTH_HZ = 0.0
_AXIS_SCHEMA = AxisSchema(("time", "point", "feature"))


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
    """Reader for MM-Fi mmWave point-cloud sample directories."""

    modality = "mmwave"
    device_family = "mmfi"
    device_name = "MM-Fi mmWave point cloud"
    reader_version = "1.0"

    def __init__(self, max_points: int | None = None) -> None:
        super().__init__()
        self.max_points = None if max_points is None else int(max_points)

    def validate_format(self, sample_root: str | Path) -> tuple[bool, str]:
        path = Path(sample_root)
        if not path.exists():
            return False, f"Sample directory not found: {path}"
        if not path.is_dir():
            return False, f"MM-Fi mmWave reader expects a sample directory, got: {path}"
        if not list(path.glob("frame*.bin")):
            return False, f"No frame*.bin files found under {path}"
        return True, ""

    def read(self, sample_root: str | Path) -> RadioTensor:
        path = Path(sample_root)
        is_valid, message = self.validate_format(path)
        if not is_valid:
            raise ReaderError(message)

        frame_arrays: list[np.ndarray] = []
        point_counts: list[int] = []
        for bin_path in sorted(path.glob("frame*.bin")):
            raw = np.fromfile(bin_path, dtype=np.float64)
            if raw.size == 0:
                points = np.zeros((0, 5), dtype=np.float32)
            else:
                points = raw.reshape(-1, 5).astype(np.float32, copy=False)
            if self.max_points is not None and points.shape[0] > self.max_points:
                points = points[: self.max_points]
            point_counts.append(int(points.shape[0]))
            frame_arrays.append(points)
        if not frame_arrays:
            raise ReaderError(f"No MM-Fi mmWave frames found under {path}")

        padded_points = max(point_counts, default=0)
        merged = np.zeros((len(frame_arrays), padded_points, 5), dtype=np.float32)
        for index, frame in enumerate(frame_arrays):
            if frame.size > 0:
                merged[index, : frame.shape[0], :] = frame
        tensor = torch.from_numpy(merged).float().contiguous()

        context = _sample_context(path)
        metadata = SignalMetadata(
            modality=self.modality,
            center_freq=_CENTER_FREQ_HZ,
            bandwidth=_BANDWIDTH_HZ,
            reader_id=self.reader_id,
            capture_device=self.device_name,
            extra={
                "dataset": "MMFi",
                "point_counts": point_counts,
                **context,
            },
        )
        metadata.set_coord("time", np.arange(int(tensor.shape[0]), dtype=np.int64), unit="frame")
        metadata.set_coord("point", np.arange(int(tensor.shape[1]), dtype=np.int64), unit="index")
        metadata.set_coord(
            "feature",
            np.arange(int(tensor.shape[2]), dtype=np.int64),
            unit="index",
        )
        return RadioTensor(tensor, _AXIS_SCHEMA, metadata)
