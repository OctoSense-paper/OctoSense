"""Raw ingest helpers and dataset materialization for the Widar3 builtin dataset."""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from octosense.core import DescribeNode
from octosense.datasets.base import resolve_declared_variant
from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor


DATASET_ID = "widar3"
logger = logging.getLogger(__name__)


class Widar3SampleRecord(Protocol):
    file_path: str
    user_id: int
    gesture_id: int
    gesture_name: str
    orientation: int
    trial: int
    rx_id: int
    room: int
    date: str

    def sample_id(self) -> str: ...

    def group_id(self) -> str: ...


def parse_widar3_capture_filename(
    filename: str | Path,
    *,
    gesture_map: dict[int, str],
) -> dict[str, object]:
    """Parse Widar3-style raw capture filenames into machine-readable fields."""

    stem = Path(filename).stem
    tokens = stem.split("-")
    if len(tokens) not in {5, 6}:
        raise ValueError(f"Unexpected Widar3 filename format: {Path(filename).name}")

    try:
        user_id = int(tokens[0].replace("user", ""))
        gesture_id = int(tokens[1])
        rx_id = int(tokens[-1].replace("r", ""))
    except ValueError as exc:
        raise ValueError(f"Cannot parse Widar3 filename: {Path(filename).name}") from exc

    parsed: dict[str, object] = {
        "sample_id": stem,
        "user_id": user_id,
        "gesture_id": gesture_id,
        "gesture_name": gesture_map.get(gesture_id, f"Unknown-{gesture_id}"),
        "rx_id": rx_id,
    }
    if len(tokens) == 5:
        parsed["filename_variant"] = "canonical"
        parsed["orientation"] = int(tokens[2])
        parsed["trial"] = int(tokens[3])
        return parsed

    parsed["filename_variant"] = "capture"
    parsed["location_id"] = int(tokens[2])
    parsed["orientation_id"] = int(tokens[3])
    parsed["repeat_id"] = int(tokens[4])
    return parsed


def coerce_int_list(payload: Any) -> list[int] | None:
    if payload is None or payload == "":
        return None
    if not isinstance(payload, list):
        raise TypeError(f"Expected integer list payload, got {type(payload)!r}")
    return [int(value) for value in payload]


def coerce_gesture_list(payload: Any) -> list[str] | None:
    if payload is None or payload == "":
        return None
    if not isinstance(payload, list):
        raise TypeError(f"Expected gesture list payload, got {type(payload)!r}")
    return [str(value) for value in payload]


def _widar3_card() -> DatasetCard:
    return get_dataset_card(DATASET_ID)


def _widar3_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields(DATASET_ID))


def _widar3_rf_profile() -> dict[str, object]:
    profile = _widar3_schema_fields().get("rf_profile")
    if not isinstance(profile, dict):
        raise ValueError("Widar3 schema.yaml must define mapping field 'rf_profile'")
    return dict(profile)


def _widar3_signal_profile() -> dict[str, object]:
    payload = _widar3_schema_fields().get("signal_profile")
    if not isinstance(payload, dict):
        raise ValueError("Widar3 schema.yaml must define mapping field 'signal_profile'")
    return dict(payload)


def _require_string(payload: dict[str, object], key: str, *, owner: str) -> str:
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"{owner} must define non-empty field '{key}'")
    return str(value)


def _require_float(payload: dict[str, object], key: str, *, owner: str) -> float:
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"{owner} must define numeric field '{key}'")
    return float(value)


def _widar3_resolved_variant(variant: str | None) -> str:
    return resolve_declared_variant(
        variant,
        supported_variants=list(_widar3_card().supported_variants or tuple(_widar3_card().variants)),
        owner="Widar3",
    )


def _widar3_variant_payload(variant: str) -> dict[str, object]:
    payload = _widar3_card().variants.get(variant)
    if isinstance(payload, dict):
        return payload
    available = ", ".join(sorted(_widar3_card().variants))
    raise ValueError(f"Unsupported Widar3 variant '{variant}'. Available variants: {available}")


def _widar3_gesture_map() -> dict[int, str]:
    raw_gesture_map = _widar3_schema_fields().get("gesture_map")
    if not isinstance(raw_gesture_map, dict):
        raise ValueError("Widar3 schema.yaml must define mapping field 'gesture_map'")
    return {int(key): str(value) for key, value in raw_gesture_map.items()}


def _widar3_reader_payload(variant: str) -> dict[str, object]:
    payload = _widar3_variant_payload(variant).get("reader", {})
    if not isinstance(payload, dict):
        return {}
    return payload


def _widar3_paths_payload(variant: str) -> dict[str, object]:
    payload = _widar3_variant_payload(variant).get("paths", {})
    if not isinstance(payload, dict):
        return {}
    return payload


def _widar3_channel(variant: str) -> int:
    payload = _widar3_reader_payload(variant)
    value = payload.get("channel")
    if value in {None, ""}:
        raise ValueError(f"Widar3 variant '{variant}' is missing reader.channel")
    return int(value)


def _widar3_metadata_csv(variant: str) -> str:
    payload = _widar3_paths_payload(variant)
    value = payload.get("metadata_csv")
    if value in {None, ""}:
        raise ValueError(f"Widar3 variant '{variant}' is missing paths.metadata_csv")
    return str(value)


def _widar3_raw_dir(variant: str) -> str:
    payload = _widar3_paths_payload(variant)
    value = payload.get("raw_dir")
    if value in {None, ""}:
        raise ValueError(f"Widar3 variant '{variant}' is missing paths.raw_dir")
    return str(value)


def _widar3_signal_modality() -> str:
    return _require_string(_widar3_signal_profile(), "signal_modality", owner="Widar3 schema.yaml signal_profile")


def _widar3_device_type() -> str:
    return _require_string(_widar3_signal_profile(), "device_type", owner="Widar3 schema.yaml signal_profile")


def _widar3_capture_device() -> str:
    return _require_string(_widar3_signal_profile(), "capture_device", owner="Widar3 schema.yaml signal_profile")


def _widar3_reader_id() -> str:
    return _require_string(_widar3_signal_profile(), "reader_id", owner="Widar3 schema.yaml signal_profile")


def _widar3_axis_schema() -> tuple[str, ...]:
    payload = _widar3_signal_profile().get("axis_schema")
    if isinstance(payload, list):
        return tuple(str(axis) for axis in payload)
    if isinstance(payload, tuple):
        return tuple(str(axis) for axis in payload)
    raise ValueError("Widar3 schema.yaml signal_profile.axis_schema must be a string list")


def _widar3_fixed_dims() -> dict[str, int]:
    payload = _widar3_signal_profile().get("fixed_dims")
    if not isinstance(payload, dict):
        raise ValueError("Widar3 schema.yaml signal_profile.fixed_dims must be a mapping")
    return {
        str(axis): int(size)
        for axis, size in payload.items()
        if size not in {None, ""}
    }


def _widar3_coord_units() -> dict[str, str]:
    payload = _widar3_signal_profile().get("coord_units")
    if not isinstance(payload, dict):
        raise ValueError("Widar3 schema.yaml signal_profile.coord_units must be a mapping")
    return {str(axis): str(unit) for axis, unit in payload.items()}


WIDAR3_CENTER_FREQ = _require_float(_widar3_rf_profile(), "center_freq_hz", owner="Widar3 schema.yaml rf_profile")
WIDAR3_BANDWIDTH = _require_float(_widar3_rf_profile(), "bandwidth_hz", owner="Widar3 schema.yaml rf_profile")
WIDAR3_NOMINAL_SAMPLE_RATE = _require_float(
    _widar3_rf_profile(),
    "nominal_sample_rate_hz",
    owner="Widar3 schema.yaml rf_profile",
)
GESTURE_MAP = _widar3_gesture_map()


class Widar3Meta:
    """Metadata entry from metas.csv."""

    def __init__(
        self,
        *,
        date: str,
        room: int,
        gesture_map: dict[int, str],
        users: list[int],
        sample_num: int,
    ) -> None:
        self.date = date
        self.room = room
        self.gesture_map = gesture_map
        self.users = users
        self.sample_num = sample_num

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "Widar3Meta":
        date = row["file"].strip()
        room = int(row["room"].strip())
        sample_num = int(row["sample_num"].strip())

        gesture_str = row["gesture_list"].strip()
        gesture_map = {}
        for entry in gesture_str.split(";"):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            idx_str, name = entry.split(":", 1)
            gesture_map[int(idx_str.strip())] = name.strip()

        user_str = row["user"].strip()
        if user_str.startswith("User"):
            user_str = user_str[4:]
        users = [int(u.strip()) for u in user_str.split(",")]

        return cls(
            date=date,
            room=room,
            gesture_map=gesture_map,
            users=users,
            sample_num=sample_num,
        )


class Widar3MetadataExtractor:
    """Extract and filter Widar3 dataset metadata."""

    _user_file_cache: dict[tuple[str, str, str, int], tuple[str, ...]] = {}

    def __init__(
        self,
        dataset_path: str,
        *,
        metadata_csv: str,
        raw_dir: str,
        skip_list: list[str] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.metadata_csv = metadata_csv
        self.raw_dir = raw_dir.strip("/") or raw_dir
        self.skip_list = tuple(skip_list or [])
        self.metas: list[Widar3Meta] = []
        self._load_metas()

    def _load_metas(self) -> None:
        metas_file = self.dataset_path / self.metadata_csv
        if not metas_file.exists():
            raise FileNotFoundError(f"metas.csv not found at {metas_file}")

        with open(metas_file, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            for row in reader:
                normalized_row = {key.strip(): value.strip() for key, value in row.items()}
                self.metas.append(Widar3Meta.from_csv_row(normalized_row))

    @classmethod
    def _list_user_files_cached(
        cls,
        dataset_path: Path,
        raw_dir: str,
        date: str,
        user_id: int,
    ) -> tuple[str, ...]:
        cache_key = (str(dataset_path.resolve()), str(raw_dir), str(date), int(user_id))
        cached = cls._user_file_cache.get(cache_key)
        if cached is None:
            user_dir = dataset_path / raw_dir / str(date) / f"user{int(user_id)}"
            if not user_dir.exists():
                cached = ()
            else:
                cached = tuple(
                    sorted(
                        entry.path
                        for entry in os.scandir(user_dir)
                        if entry.is_file() and entry.name.endswith(".dat")
                    )
                )
            cls._user_file_cache[cache_key] = cached
        return cached

    def filter_samples(
        self,
        users: list[int] | None = None,
        gestures: list[str] | None = None,
        rooms: list[int] | None = None,
        rx_ids: list[int] | None = None,
        dates: list[str] | None = None,
    ) -> list[Widar3SampleRecord]:
        from .manifest import load_widar3_sample

        samples: list[Widar3SampleRecord] = []
        selected_dates = None if dates is None else {str(date) for date in dates}

        for meta in self.metas:
            if selected_dates is not None and str(meta.date) not in selected_dates:
                continue
            if rooms is not None and meta.room not in rooms:
                continue

            selected_users = set(meta.users)
            if users is not None:
                selected_users &= set(users)
            if not selected_users:
                continue

            selected_gestures: list[str] = []
            if gestures is None:
                selected_gestures = list(meta.gesture_map.values())
            else:
                for gesture_name in gestures:
                    if gesture_name in meta.gesture_map.values():
                        selected_gestures.append(gesture_name)
            if not selected_gestures:
                continue

            for user_id in selected_users:
                matching_files = self._list_user_files_cached(
                    self.dataset_path,
                    self.raw_dir,
                    meta.date,
                    user_id,
                )
                for file_path in matching_files:
                    rel_path = os.path.relpath(file_path, self.dataset_path)
                    if any(skip in rel_path for skip in self.skip_list):
                        continue
                    try:
                        sample = load_widar3_sample(
                            file_path=rel_path,
                            room=meta.room,
                            gesture_map=meta.gesture_map,
                        )
                    except ValueError:
                        continue
                    if sample.gesture_name not in selected_gestures:
                        continue
                    if rx_ids is not None and sample.rx_id not in rx_ids:
                        continue
                    samples.append(sample)

        return samples


class Widar3Dataset(Dataset[tuple[RadioTensor | torch.Tensor, int]]):
    """PyTorch dataset for Widar3 gesture recognition."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        variant: str | None = None,
        users: list[int] | None = None,
        gestures: list[str] | None = None,
        rooms: list[int] | None = None,
        rx_ids: list[int] | None = None,
        transform: Callable[[RadioTensor], RadioTensor] | None = None,
        preload: bool = False,
        channel: int | None = None,
        return_radiotensor: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.variant = _widar3_resolved_variant(variant)
        self.transform = transform
        variant_payload = _widar3_variant_payload(self.variant)
        self.preload = preload
        self.channel = _widar3_channel(self.variant) if channel is None else int(channel)
        self.return_radiotensor = return_radiotensor
        resolved_gestures = gestures or [
            str(gesture_name) for gesture_name in variant_payload.get("gestures", [])
        ]
        resolved_dates = coerce_gesture_list(variant_payload.get("dates"))

        extractor = Widar3MetadataExtractor(
            str(self.dataset_path),
            metadata_csv=_widar3_metadata_csv(self.variant),
            raw_dir=_widar3_raw_dir(self.variant),
            skip_list=[str(item) for item in variant_payload.get("skip_list", [])],
        )
        self.samples = extractor.filter_samples(
            users=users,
            gestures=resolved_gestures or None,
            rooms=rooms,
            rx_ids=rx_ids,
            dates=resolved_dates,
        )
        if not self.samples:
            raise ValueError("No samples found matching the criteria")

        unique_gestures = sorted({sample.gesture_name for sample in self.samples})
        self.label_mapping = {name: idx for idx, name in enumerate(unique_gestures)}
        self.reader = io_readers.load("wifi/iwl5300", channel=self.channel)
        self._dataset_metadata = DatasetMetadata(
            name="Widar3",
            sample_count=len(self.samples),
            users=sorted({sample.user_id for sample in self.samples}),
            gestures=list(self.label_mapping.keys()),
            rooms=sorted({sample.room for sample in self.samples}),
            collection_dates=sorted({sample.date for sample in self.samples}),
            device_type=_widar3_device_type(),
            center_freq=WIDAR3_CENTER_FREQ,
            bandwidth=WIDAR3_BANDWIDTH,
            nominal_sample_rate=WIDAR3_NOMINAL_SAMPLE_RATE,
            extra={
                "variant": self.variant,
                "variant_split": variant_payload.get("split"),
                "variant_dates": list(resolved_dates or []),
                "skip_list_count": len(variant_payload.get("skip_list", [])),
                "reader_channel": self.channel,
                "declared_center_freq": WIDAR3_CENTER_FREQ,
                "declared_bandwidth": WIDAR3_BANDWIDTH,
                "declared_nominal_sample_rate": WIDAR3_NOMINAL_SAMPLE_RATE,
            },
        )
        self.data_cache: list[RadioTensor | None] | None = None
        if self.preload:
            self._preload_data()

    def _preload_data(self) -> None:
        self.data_cache = []
        for sample in self.samples:
            full_path = self.dataset_path / sample.file_path
            try:
                loaded = self.reader.read(str(full_path))
                self._record_observed_physical_metadata(loaded)
                self.data_cache.append(loaded)
            except Exception as exc:
                logger.warning("Failed to preload %s: %s", sample.file_path, exc)
                self.data_cache.append(None)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_radiotensor(self, idx: int) -> RadioTensor:
        sample = self.samples[idx]
        if self.preload and self.data_cache is not None:
            rt = self.data_cache[idx]
            if rt is None:
                raise RuntimeError(f"Failed to load cached data for {sample.file_path}")
        else:
            full_path = self.dataset_path / sample.file_path
            rt = self.reader.read(str(full_path))
        self._record_observed_physical_metadata(rt)
        if self.transform is not None:
            rt = self.transform(rt)
        return rt

    def _record_observed_physical_metadata(self, sample: RadioTensor) -> None:
        center_freq = sample.metadata.center_freq
        if center_freq is not None:
            center_freq = float(center_freq)
            if np.isfinite(center_freq) and center_freq > 0:
                self._dataset_metadata.center_freq = center_freq

        bandwidth = sample.metadata.bandwidth
        if bandwidth is not None:
            bandwidth = float(bandwidth)
            if np.isfinite(bandwidth) and bandwidth > 0:
                self._dataset_metadata.bandwidth = bandwidth

        sample_rate = sample.metadata.sample_rate
        if sample_rate is not None:
            sample_rate = float(sample_rate)
            if np.isfinite(sample_rate) and sample_rate > 0:
                self._dataset_metadata.estimated_sample_rate = sample_rate

    def __getitem__(self, idx: int) -> tuple[RadioTensor | torch.Tensor, int]:
        rt = self._load_radiotensor(idx)
        label = self.label_mapping[self.samples[idx].gesture_name]
        if self.return_radiotensor:
            return rt, label
        return rt.as_tensor(), label

    def get_label_name(self, label: int) -> str:
        for name, index in self.label_mapping.items():
            if index == label:
                return name
        raise ValueError(f"Invalid label: {label}")

    def get_label_mapping(self) -> dict[str, int]:
        return self.label_mapping.copy()

    def get_labels(self) -> list[int]:
        return [self.label_mapping[sample.gesture_name] for sample in self.samples]

    def get_sample(self, idx: int) -> Widar3SampleRecord:
        return self.samples[idx]

    def get_sample_id(self, idx: int) -> str:
        return self.get_sample(idx).sample_id()

    def get_group_id(self, idx: int) -> str:
        return self.get_sample(idx).group_id()

    def metadata_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        labels = self.get_labels()
        for index, (sample, label) in enumerate(zip(self.samples, labels, strict=True)):
            rows.append(
                {
                    "sample_index": int(index),
                    "sample_id": sample.sample_id(),
                    "sample_group_id": sample.group_id(),
                    "user_id": int(sample.user_id),
                    "subject": int(sample.user_id),
                    "gesture_id": int(sample.gesture_id),
                    "gesture_name": str(sample.gesture_name),
                    "room": int(sample.room),
                    "environment": int(sample.room),
                    "rx_id": int(sample.rx_id),
                    "orientation": int(sample.orientation),
                    "trial": int(sample.trial),
                    "date": str(sample.date),
                    "session_id": f"{sample.date}|room={sample.room}",
                    "file_path": str(sample.file_path),
                    "label": int(label),
                }
            )
        return rows

    def sample_describe_tree(self) -> DescribeNode:
        if self.transform is None:
            fixed_dims = _widar3_fixed_dims()
            coord_units = _widar3_coord_units()
            return DescribeNode(
                kind="radiotensor",
                name="sample",
                fields={
                    "shape": [
                        "variable",
                        int(fixed_dims.get("subc", 30)),
                        int(fixed_dims.get("tx", 1)),
                        int(fixed_dims.get("rx", 3)),
                    ],
                    "dtype": str(torch.complex64),
                    "device": "cpu",
                    "modality": _widar3_signal_modality(),
                },
                children=(
                    DescribeNode(
                        kind="axis_schema",
                        name="axis_schema",
                        fields={"axes": list(_widar3_axis_schema())},
                    ),
                    DescribeNode(
                        kind="signal_metadata",
                        name="metadata",
                        fields={
                            "reader_id": self.reader.reader_id,
                            "capture_device": _widar3_capture_device(),
                            "sample_rate": (
                                self._dataset_metadata.estimated_sample_rate
                                or self._dataset_metadata.nominal_sample_rate
                            ),
                            "center_freq": self._dataset_metadata.center_freq,
                            "bandwidth": self._dataset_metadata.bandwidth,
                            "modality": _widar3_signal_modality(),
                        },
                    ),
                    DescribeNode(
                        kind="coords",
                        name="coords",
                        children=(
                            DescribeNode(
                                kind="coord_axis",
                                name="time",
                                fields={
                                    "available": True,
                                    "length": "variable",
                                    "unit": coord_units.get("time", "s"),
                                },
                            ),
                            DescribeNode(
                                kind="coord_axis",
                                name="subc",
                                fields={
                                    "available": True,
                                    "length": int(fixed_dims.get("subc", 30)),
                                    "unit": coord_units.get("subc", "Hz"),
                                },
                            ),
                        ),
                    ),
                ),
            )

        sample, _ = self[0]
        return sample.describe_tree().with_name("sample")

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["reader"] = None
        state["data_cache"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.reader = io_readers.load("wifi/iwl5300", channel=self.channel)

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        return self._dataset_metadata

    @property
    def dataset_card(self) -> DatasetCard:
        return get_dataset_card(DATASET_ID)

    def estimate_sample_rate(
        self,
        sample_indices: list[int] | None = None,
        update_metadata: bool = False,
    ) -> float | None:
        if sample_indices is None:
            if self._dataset_metadata.estimated_sample_rate is not None:
                return self._dataset_metadata.estimated_sample_rate
            sample_indices = list(range(min(10, len(self))))
        if not sample_indices:
            return self._dataset_metadata.estimated_sample_rate

        sample_rates: list[float] = []
        for idx in sample_indices:
            sample = self.samples[idx]
            if self.preload and self.data_cache is not None:
                rt = self.data_cache[idx]
            else:
                full_path = self.dataset_path / sample.file_path
                rt = self.reader.read(str(full_path))
            if rt is not None and rt.metadata.sample_rate is not None:
                self._record_observed_physical_metadata(rt)
                sample_rates.append(rt.metadata.sample_rate)

        if not sample_rates:
            logger.warning("No sample_rate found in metadata for Widar3 sample probe")
            return self._dataset_metadata.estimated_sample_rate

        estimated_rate = float(torch.tensor(sample_rates).mean().item())
        if update_metadata:
            self._dataset_metadata.estimated_sample_rate = estimated_rate
        return estimated_rate


__all__ = [
    "DATASET_ID",
    "GESTURE_MAP",
    "WIDAR3_BANDWIDTH",
    "WIDAR3_CENTER_FREQ",
    "WIDAR3_NOMINAL_SAMPLE_RATE",
    "Widar3Dataset",
    "Widar3Meta",
    "Widar3MetadataExtractor",
    "coerce_gesture_list",
    "coerce_int_list",
    "parse_widar3_capture_filename",
]
