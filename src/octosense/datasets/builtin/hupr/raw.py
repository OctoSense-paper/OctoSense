"""HuPR raw binary decoding and canonical case2 preparation surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from octosense.io.semantics.metadata import SignalMetadata
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor

DEFAULT_NUM_RX = 4
DEFAULT_NUM_LANES = 2
DEFAULT_NUM_ADC_SAMPLES = 256
DEFAULT_NUM_CHIRPS_PER_FRAME = 64 * 3
DEFAULT_PROC_CHIRPS = 64
DEFAULT_GROUP_CHIRPS = 4
DEFAULT_ADC_RATIO = 4
DEFAULT_NUM_ANGLE_BINS = DEFAULT_NUM_ADC_SAMPLES // DEFAULT_ADC_RATIO
DEFAULT_NUM_ELEVATION_BINS = 8
DEFAULT_OUTPUT_CHIRPS = DEFAULT_PROC_CHIRPS // DEFAULT_GROUP_CHIRPS
DEFAULT_MODEL_FRAMES = 8
DEFAULT_DURATION_SECONDS = 60
DEFAULT_FPS = 10
DEFAULT_NUM_SCENE_FRAMES = DEFAULT_DURATION_SECONDS * DEFAULT_FPS
DEFAULT_GROUP_FRAMES = 8

VIEW_NAMES = ("hori", "vert")
VIEW_TO_INDEX = {name: index for index, name in enumerate(VIEW_NAMES)}


def resolve_hupr_raw_root(dataset_path: str | Path) -> Path:
    root = Path(dataset_path)
    candidates = [
        root,
        root / "radar_unzipped",
        root / "HuPR",
        root / "raw_data" / "iwr1843" / "HuPR",
        root / "preprocessing" / "raw_data" / "iwr1843" / "HuPR",
    ]
    for candidate in candidates:
        if (candidate / "single_1" / "hori" / "adc_data.bin").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve HuPR raw root under {dataset_path}; expected single_x/{{hori,vert}}/adc_data.bin"
    )


def resolve_hupr_radar_map_root(dataset_path: str | Path) -> Path | None:
    root = Path(dataset_path)
    candidates = [
        root / "radar_maps",
        root.parent / "radar_maps",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class HuPRRawClipDataset(Dataset[RadioTensor]):
    """Canonical owner-native raw HuPR clip dataset for case2 demos."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        num_group_frames: int = DEFAULT_GROUP_FRAMES,
        max_samples: int | None = None,
        scene_ids: Sequence[int] | None = None,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.raw_root = resolve_hupr_raw_root(dataset_path)
        self.radar_map_root = resolve_hupr_radar_map_root(dataset_path)
        self.num_group_frames = int(num_group_frames)
        self.scenes = collect_hupr_raw_scenes(dataset_path, scene_ids=scene_ids)
        self.records = self._collect_records(max_samples=max_samples)

    def _collect_records(self, *, max_samples: int | None) -> list[HuPRRawRecord]:
        records: list[HuPRRawRecord] = []
        for scene in self.scenes:
            for target_frame in range(scene.num_frames):
                records.append(
                    HuPRRawRecord(
                        scene=scene,
                        target_frame=target_frame,
                        group_frame_indexes=compute_group_frame_indexes(
                            target_frame,
                            total_frames=scene.num_frames,
                            num_group_frames=self.num_group_frames,
                        ),
                    )
                )
                if max_samples is not None and len(records) >= int(max_samples):
                    return records
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> RadioTensor:
        record = self.records[idx]
        clip = load_hupr_raw_clip(record.scene, group_frame_indexes=record.group_frame_indexes)
        metadata = SignalMetadata(
            modality="mmwave",
            reader_id="hupr/raw_clip",
            capture_device="IWR1843Boost",
            extra={
                "sample_id": record.sample_id,
                "scene": record.scene.scene_name,
                "scene_id": record.scene.scene_id,
                "target_frame": record.target_frame,
                "group_frame_indexes": list(record.group_frame_indexes),
                "center_group_index": self.num_group_frames // 2,
                "num_group_frames": self.num_group_frames,
                "radar_map_root": str(self.radar_map_root) if self.radar_map_root is not None else "",
            },
        )
        metadata.set_coord("group", np.asarray(record.group_frame_indexes, dtype=np.int32), unit="frame")
        metadata.set_coord("rx", np.arange(DEFAULT_NUM_RX, dtype=np.int32), unit="index")
        metadata.set_coord("chirp", np.arange(DEFAULT_NUM_CHIRPS_PER_FRAME, dtype=np.int32), unit="index")
        metadata.set_coord("adc", np.arange(DEFAULT_NUM_ADC_SAMPLES, dtype=np.int32), unit="sample")
        return RadioTensor(
            data=torch.from_numpy(clip),
            axis_schema=AxisSchema(("view", "group", "rx", "chirp", "adc")),
            metadata=metadata,
        )

    def describe(self) -> str:
        sample = self[0]
        return (
            "HuPRRawClipDataset\n"
            f"  root: {self.dataset_path}\n"
            f"  raw_root: {self.raw_root}\n"
            f"  samples: {len(self)}\n"
            f"  sample_axes: {sample.axis_schema.axes}\n"
            f"  sample_shape: {tuple(sample.shape)}"
        )


def prepare_hupr_case2_inputs(
    dataset_path: str | Path,
    *,
    transform: Callable[[RadioTensor], RadioTensor | torch.Tensor],
    max_samples: int | None = None,
    num_group_frames: int = DEFAULT_GROUP_FRAMES,
    scene_ids: Sequence[int] | None = None,
) -> tuple[HuPRRawClipDataset, list[RadioTensor | torch.Tensor], tuple[int, ...]]:
    """Materialize case2-ready model inputs using a caller-owned demo transform."""

    dataset = HuPRRawClipDataset(
        dataset_path,
        num_group_frames=num_group_frames,
        max_samples=max_samples,
        scene_ids=scene_ids,
    )
    prepared_inputs = [transform(dataset[index]) for index in range(len(dataset))]
    if not prepared_inputs:
        raise ValueError("prepare_hupr_case2_inputs(...) requires at least one prepared sample.")

    first = prepared_inputs[0]
    first_tensor = first.as_tensor() if isinstance(first, RadioTensor) else first
    batch_shape = (len(prepared_inputs), *tuple(first_tensor.shape))
    return dataset, prepared_inputs, batch_shape


@dataclass(frozen=True)
class HuPRSceneRecord:
    scene_dir: Path
    scene_id: int
    hori_bin: Path
    vert_bin: Path
    num_frames: int

    @property
    def scene_name(self) -> str:
        return self.scene_dir.name


@dataclass(frozen=True)
class HuPRRawRecord:
    scene: HuPRSceneRecord
    target_frame: int
    group_frame_indexes: tuple[int, ...]

    @property
    def sample_id(self) -> str:
        return f"{self.scene.scene_name}:{self.target_frame:04d}"


def _parse_scene_id(scene_dir: Path) -> int:
    return int(scene_dir.name.split("_")[-1])


def collect_hupr_raw_scenes(
    dataset_path: str | Path,
    *,
    scene_ids: Sequence[int] | None = None,
) -> list[HuPRSceneRecord]:
    dataset_root = resolve_hupr_raw_root(dataset_path)
    selected = set(int(scene_id) for scene_id in scene_ids) if scene_ids is not None else None
    records: list[HuPRSceneRecord] = []
    for scene_dir in sorted(dataset_root.glob("single_*")):
        scene_id = _parse_scene_id(scene_dir)
        if selected is not None and scene_id not in selected:
            continue
        hori_bin = scene_dir / "hori" / "adc_data.bin"
        vert_bin = scene_dir / "vert" / "adc_data.bin"
        if not hori_bin.exists() or not vert_bin.exists():
            continue
        num_frames = min(
            infer_num_frames(hori_bin),
            infer_num_frames(vert_bin),
        )
        records.append(
            HuPRSceneRecord(
                scene_dir=scene_dir,
                scene_id=scene_id,
                hori_bin=hori_bin,
                vert_bin=vert_bin,
                num_frames=num_frames,
            )
        )
    if not records:
        raise ValueError(f"No HuPR raw scenes found under {dataset_root}")
    return records


def compute_group_frame_indexes(
    target_frame: int,
    *,
    total_frames: int = DEFAULT_NUM_SCENE_FRAMES,
    num_group_frames: int = DEFAULT_GROUP_FRAMES,
) -> tuple[int, ...]:
    pad_size = int(target_frame % total_frames)
    cursor = int(target_frame - num_group_frames // 2 - 1)
    indexes: list[int] = []
    for group_index in range(num_group_frames):
        if (group_index + pad_size) <= num_group_frames // 2:
            cursor = target_frame - pad_size
        elif group_index > (total_frames - 1 - pad_size) + num_group_frames // 2:
            cursor = target_frame + (total_frames - 1 - pad_size)
        else:
            cursor += 1
        indexes.append(int(cursor))
    return tuple(indexes)


def infer_num_frames(
    bin_path: str | Path,
    *,
    num_rx: int = DEFAULT_NUM_RX,
    num_adc_samples: int = DEFAULT_NUM_ADC_SAMPLES,
    num_chirps_per_frame: int = DEFAULT_NUM_CHIRPS_PER_FRAME,
) -> int:
    ints_per_chirp = num_adc_samples * num_rx * 2
    total_int16 = Path(bin_path).stat().st_size // np.dtype(np.int16).itemsize
    if total_int16 % ints_per_chirp != 0:
        raise ValueError(f"{bin_path} is not aligned to full chirps")
    total_chirps = total_int16 // ints_per_chirp
    if total_chirps % num_chirps_per_frame != 0:
        raise ValueError(f"{bin_path} is not aligned to full HuPR frames")
    return total_chirps // num_chirps_per_frame


def decode_iwr1843_adc(
    bin_path: str | Path,
    *,
    num_rx: int = DEFAULT_NUM_RX,
    num_lanes: int = DEFAULT_NUM_LANES,
    num_adc_samples: int = DEFAULT_NUM_ADC_SAMPLES,
) -> np.ndarray:
    ints_per_chirp = num_adc_samples * num_rx * 2
    total_int16 = Path(bin_path).stat().st_size // np.dtype(np.int16).itemsize
    if total_int16 % ints_per_chirp != 0:
        raise ValueError(f"{bin_path} is not aligned to full chirps")
    chirps_per_scene = total_int16 // ints_per_chirp
    return decode_iwr1843_adc_segment(
        bin_path,
        chirp_start=0,
        chirp_count=chirps_per_scene,
        num_rx=num_rx,
        num_lanes=num_lanes,
        num_adc_samples=num_adc_samples,
    )


def decode_iwr1843_adc_segment(
    bin_path: str | Path,
    *,
    chirp_start: int,
    chirp_count: int,
    num_rx: int = DEFAULT_NUM_RX,
    num_lanes: int = DEFAULT_NUM_LANES,
    num_adc_samples: int = DEFAULT_NUM_ADC_SAMPLES,
) -> np.ndarray:
    ints_per_chirp = num_adc_samples * num_rx * 2
    byte_offset = int(chirp_start * ints_per_chirp * np.dtype(np.int16).itemsize)
    adc_data = np.fromfile(
        bin_path,
        dtype=np.int16,
        count=int(chirp_count * ints_per_chirp),
        offset=byte_offset,
    )
    if adc_data.size != chirp_count * ints_per_chirp:
        raise ValueError(
            f"Expected {chirp_count * ints_per_chirp} int16 values from {bin_path}, got {adc_data.size}"
        )
    adc_data = adc_data.reshape(-1, num_lanes * 2).transpose()

    file_size = int(adc_data.shape[1] * num_lanes * 2 // 2)
    lvds = np.zeros((2, file_size), dtype=np.float32)

    temp = np.empty(adc_data[0].size + adc_data[1].size, dtype=np.float32)
    temp[0::2] = adc_data[0]
    temp[1::2] = adc_data[1]
    lvds[0] = temp

    temp = np.empty(adc_data[2].size + adc_data[3].size, dtype=np.float32)
    temp[0::2] = adc_data[2]
    temp[1::2] = adc_data[3]
    lvds[1] = temp

    complex_adc = np.zeros((num_rx, file_size // num_rx), dtype=np.complex64)
    cursor = 0
    step = num_adc_samples * 4
    for idx in range(0, file_size, step):
        complex_adc[0, cursor : cursor + num_adc_samples] = lvds[0, idx : idx + num_adc_samples] + 1j * lvds[1, idx : idx + num_adc_samples]
        complex_adc[1, cursor : cursor + num_adc_samples] = lvds[0, idx + num_adc_samples : idx + num_adc_samples * 2] + 1j * lvds[1, idx + num_adc_samples : idx + num_adc_samples * 2]
        complex_adc[2, cursor : cursor + num_adc_samples] = lvds[0, idx + num_adc_samples * 2 : idx + num_adc_samples * 3] + 1j * lvds[1, idx + num_adc_samples * 2 : idx + num_adc_samples * 3]
        complex_adc[3, cursor : cursor + num_adc_samples] = lvds[0, idx + num_adc_samples * 3 : idx + num_adc_samples * 4] + 1j * lvds[1, idx + num_adc_samples * 3 : idx + num_adc_samples * 4]
        cursor += num_adc_samples

    return complex_adc.reshape(num_rx, chirp_count, num_adc_samples)


def load_hupr_raw_clip(
    scene: HuPRSceneRecord,
    *,
    group_frame_indexes: Sequence[int],
) -> np.ndarray:
    frame_span_start = int(min(group_frame_indexes))
    frame_span_stop = int(max(group_frame_indexes)) + 1
    chirp_start = frame_span_start * DEFAULT_NUM_CHIRPS_PER_FRAME
    chirp_count = (frame_span_stop - frame_span_start) * DEFAULT_NUM_CHIRPS_PER_FRAME

    hori_segment = decode_iwr1843_adc_segment(
        scene.hori_bin,
        chirp_start=chirp_start,
        chirp_count=chirp_count,
    )
    vert_segment = decode_iwr1843_adc_segment(
        scene.vert_bin,
        chirp_start=chirp_start,
        chirp_count=chirp_count,
    )

    def _slice_group(segment: np.ndarray) -> np.ndarray:
        frames: list[np.ndarray] = []
        for frame_index in group_frame_indexes:
            relative = int(frame_index - frame_span_start)
            start = relative * DEFAULT_NUM_CHIRPS_PER_FRAME
            stop = start + DEFAULT_NUM_CHIRPS_PER_FRAME
            frames.append(np.asarray(segment[:, start:stop, :], dtype=np.complex64))
        return np.stack(frames, axis=0)

    return np.stack((_slice_group(hori_segment), _slice_group(vert_segment)), axis=0)


__all__ = [
    "DEFAULT_ADC_RATIO",
    "DEFAULT_GROUP_CHIRPS",
    "DEFAULT_GROUP_FRAMES",
    "DEFAULT_MODEL_FRAMES",
    "DEFAULT_NUM_ADC_SAMPLES",
    "DEFAULT_NUM_ANGLE_BINS",
    "DEFAULT_NUM_CHIRPS_PER_FRAME",
    "DEFAULT_NUM_ELEVATION_BINS",
    "DEFAULT_NUM_LANES",
    "DEFAULT_NUM_RX",
    "DEFAULT_OUTPUT_CHIRPS",
    "DEFAULT_PROC_CHIRPS",
    "HuPRRawClipDataset",
    "HuPRRawRecord",
    "HuPRSceneRecord",
    "VIEW_NAMES",
    "VIEW_TO_INDEX",
    "collect_hupr_raw_scenes",
    "compute_group_frame_indexes",
    "decode_iwr1843_adc",
    "decode_iwr1843_adc_segment",
    "infer_num_frames",
    "load_hupr_raw_clip",
    "prepare_hupr_case2_inputs",
    "resolve_hupr_radar_map_root",
    "resolve_hupr_raw_root",
]
