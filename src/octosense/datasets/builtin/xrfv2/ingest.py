"""Ingest owner for the XRFV2 builtin dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
from torch.utils.data import Dataset

from octosense.datasets.base import resolve_dataset_root
from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor


@lru_cache(maxsize=1)
def _xrfv2_card() -> DatasetCard:
    return get_dataset_card("xrfv2")


def _xrfv2_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields("xrfv2"))


def _xrfv2_rf_profile() -> dict[str, object]:
    profile = _xrfv2_schema_fields().get("rf_profile")
    if not isinstance(profile, dict):
        raise ValueError("XRFV2 schema.yaml must define mapping field 'rf_profile'")
    return dict(profile)


def _xrfv2_signal_profile() -> dict[str, object]:
    profile = _xrfv2_schema_fields().get("signal_profile")
    if not isinstance(profile, dict):
        raise ValueError("XRFV2 schema.yaml must define mapping field 'signal_profile'")
    return dict(profile)


def _xrfv2_label_mapping_payload() -> dict[str, object]:
    label_mapping = _xrfv2_schema_fields().get("label_mapping")
    if not isinstance(label_mapping, dict):
        raise ValueError("XRFV2 schema.yaml must define mapping field 'label_mapping'")
    return dict(label_mapping)


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


def _xrfv2_variant_payload(variant_key: str) -> dict[str, object]:
    resolved_variant_key = str(variant_key).strip()
    if not resolved_variant_key:
        raise ValueError("XRFV2 variant must be a non-empty canonical binding id.")

    payload = _xrfv2_card().variants.get(resolved_variant_key)
    if not isinstance(payload, dict):
        available = ", ".join(sorted(_xrfv2_card().variants))
        raise ValueError(
            f"Unsupported XRFV2 variant '{resolved_variant_key}'. Available variants: {available}"
        )
    return payload


XRFV2_DATASET_ID = _xrfv2_card().dataset_id
_RF_PROFILE = _xrfv2_rf_profile()
_SIGNAL_PROFILE = _xrfv2_signal_profile()
_LABEL_MAPPING = _xrfv2_label_mapping_payload()
XRFV2_WIFI_CENTER_FREQ = _require_float(
    _RF_PROFILE,
    "center_freq_hz",
    owner="XRFV2 schema.yaml rf_profile",
)
XRFV2_WIFI_BANDWIDTH = _require_float(
    _RF_PROFILE,
    "bandwidth_hz",
    owner="XRFV2 schema.yaml rf_profile",
)
XRFV2_DEFAULT_NEW_MAPPING: dict[int, int] = {
    int(key): int(value) for key, value in dict(_LABEL_MAPPING.get("new_mapping", {})).items()
}
XRFV2_DEFAULT_ID_TO_ACTION: dict[int, str] = {
    int(key): str(value) for key, value in dict(_LABEL_MAPPING.get("id_to_action", {})).items()
}


@dataclass(frozen=True)
class XRFV2ClipSample:
    split: str
    segment_index: int
    start: int
    end: int
    action_id: int
    mapped_action_id: int
    base_index: int | None = None
    source_file: str | None = None
    volunteer_id: int | None = None
    scene_id: int | None = None

    def sample_id(self) -> str:
        parts = [f"split={self.split}"]
        if self.base_index is not None:
            parts.append(f"sample={self.base_index}")
        if self.source_file is not None:
            parts.append(f"file={self.source_file}")
        if self.volunteer_id is not None:
            parts.append(f"volunteer={self.volunteer_id}")
        if self.scene_id is not None:
            parts.append(f"scene={self.scene_id}")
        parts.extend(
            [
                f"segment={self.segment_index}",
                f"start={self.start}",
                f"end={self.end}",
                f"label={self.mapped_action_id}",
            ]
        )
        return "|".join(parts)

    def group_id(self) -> str:
        if self.base_index is not None:
            return f"split={self.split}|sample={self.base_index}"
        if self.source_file is not None:
            return f"split={self.split}|file={self.source_file}"
        return f"split={self.split}|segment={self.segment_index}"


@dataclass(frozen=True)
class XRFV2RawRecord:
    volunteer_id: int
    scene_id: int
    action_group_id: int
    wifi_relative_path: str
    file_name: str


def _parse_segment(item: Any) -> tuple[int, int, int]:
    if isinstance(item, dict):
        if "segment" in item and "label" in item:
            start, end = item["segment"]
            return int(start), int(end), int(item["label"])
        if {"start", "end", "label"}.issubset(item):
            return int(item["start"]), int(item["end"]), int(item["label"])
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return int(item[0]), int(item[1]), int(item[-1])
    raise ValueError(f"Unsupported XRF V2 segment annotation format: {item!r}")


def _parse_raw_segment_row(row: Any) -> tuple[int, int, int, int]:
    values = list(int(value) for value in row)
    if len(values) < 4:
        raise ValueError(f"Unsupported XRF V2 raw label row: {row!r}")
    return values[0], values[1], values[2], values[3]


def _load_info_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_xrfv2_label_mapping(
    info: dict[str, Any],
) -> tuple[dict[int, int] | None, dict[int, str]]:
    raw_mapping = info.get("segment_info", {}).get("new_mapping")
    new_mapping = (
        {int(key): int(value) for key, value in raw_mapping.items()}
        if isinstance(raw_mapping, dict)
        else dict(XRFV2_DEFAULT_NEW_MAPPING)
    )
    id_to_action_raw = info.get("segment_info", {}).get("id2action")
    if isinstance(id_to_action_raw, dict):
        id_to_action = {int(key): str(value) for key, value in id_to_action_raw.items()}
    else:
        id_to_action = dict(XRFV2_DEFAULT_ID_TO_ACTION)
    return new_mapping, id_to_action


def _resolve_xrfv2_raw_root(dataset_path: Path) -> Path | None:
    candidates = [dataset_path / "WWADL_open", dataset_path]
    for candidate in candidates:
        if (candidate / "file_records.csv").exists() and (candidate / "wifi").is_dir():
            return candidate
    return None


def _load_raw_split_file_names(
    raw_root: Path,
    split: str,
) -> tuple[Path | None, set[str]]:
    candidate = raw_root / f"{split}.csv"
    if not candidate.exists():
        return None, set()
    with candidate.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names: set[str] = set()
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            if file_name:
                names.add(file_name)
    return candidate, names


def _load_raw_records(file_records_path: Path) -> list[XRFV2RawRecord]:
    with file_records_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records: list[XRFV2RawRecord] = []
        for row in reader:
            wifi_relative_path = (row.get("wifi_path") or "").strip()
            file_name = (row.get("file_name") or "").strip()
            if not wifi_relative_path or not file_name:
                continue
            records.append(
                XRFV2RawRecord(
                    volunteer_id=int(row.get("volunteer_id", 0)),
                    scene_id=int(row.get("scene_id", 0)),
                    action_group_id=int(row.get("action_group_id", 0)),
                    wifi_relative_path=wifi_relative_path,
                    file_name=file_name,
                )
            )
    return records


def _select_raw_records_for_split(
    records: list[XRFV2RawRecord],
    *,
    raw_root: Path,
    split: str,
) -> list[XRFV2RawRecord]:
    split_file, named_files = _load_raw_split_file_names(raw_root, split)
    if split_file is None:
        raise ValueError(
            "XRFV2 raw layout requires an explicit split CSV owned by the raw layout. "
            f"Missing: {raw_root / f'{split}.csv'}"
        )
    if not named_files:
        raise ValueError(
            f"XRFV2 raw split file {split_file} does not contain any file_name entries."
        )
    selected = [record for record in records if record.file_name in named_files]
    if not selected:
        raise ValueError(
            "XRFV2 raw split file did not match any file_records.csv entries. "
            f"split={split!r}, split_file={split_file}"
        )
    return selected


def _processed_layout_available(dataset_path: Path, split: str) -> bool:
    return (
        (dataset_path / f"{split}_data.h5").exists()
        and (dataset_path / f"{split}_label.json").exists()
        and (dataset_path / "info.json").exists()
    )


def _resolve_variant_config(variant_key: str) -> dict[str, Any]:
    return dict(_xrfv2_variant_payload(variant_key))


def _augment_xrfv2_sample_metadata(
    signal: RadioTensor,
    sample: XRFV2ClipSample,
    *,
    variant_name: str,
    variant_key: str,
) -> RadioTensor:
    metadata = signal.metadata.copy()
    metadata.extra.update(
        {
            "dataset": XRFV2_DATASET_ID,
            "sample_id": sample.sample_id(),
            "split": sample.split,
            "base_index": sample.base_index,
            "segment_index": sample.segment_index,
            "segment_start": sample.start,
            "segment_end": sample.end,
            "action_id": sample.action_id,
            "mapped_action_id": sample.mapped_action_id,
            "source_file": sample.source_file,
            "volunteer_id": sample.volunteer_id,
            "scene_id": sample.scene_id,
            "variant": variant_name,
            "variant_key": variant_key,
        }
    )
    return signal.with_metadata(metadata)


class _XRFV2WiFiDataset(Dataset[tuple[RadioTensor, int]]):
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        variant: str,
        split: str = "train",
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.dataset_path = resolve_dataset_root(XRFV2_DATASET_ID, override=dataset_path)
        self.split = split
        self.layout = (
            "processed" if _processed_layout_available(self.dataset_path, split) else "raw"
        )
        resolved_variant = str(variant).strip()
        if not resolved_variant:
            raise ValueError("XRFV2 WiFi dataset requires an explicit canonical variant binding id.")
        variant_payload = _resolve_variant_config(resolved_variant)
        self.variant = resolved_variant
        self.variant_name = str(variant_payload.get("variant", resolved_variant))
        self.variant_key = str(variant_payload.get("variant_key", resolved_variant))
        self.reader = io_readers.load(
            _require_string(
                _SIGNAL_PROFILE,
                "reader_id",
                owner="XRFV2 schema.yaml signal_profile",
            )
        )
        self.data_path: Path | None = None
        self.label_path: Path | None = None
        self.info_path: Path | None = None
        self.raw_root: Path | None = None
        self.info: dict[str, Any] = {}
        self.samples: list[XRFV2ClipSample] = []
        if self.layout == "processed":
            self._init_processed_layout()
        else:
            self._init_raw_layout()
        if not self.samples:
            raise ValueError("No XRF V2 WiFi segments matched the requested split")

        unique_action_ids = sorted({sample.mapped_action_id for sample in self.samples})
        self.class_id_to_contiguous = {
            action_id: idx for idx, action_id in enumerate(unique_action_ids)
        }
        self.label_mapping = {
            self.id_to_action.get(action_id, f"action_{action_id}"): contiguous
            for action_id, contiguous in self.class_id_to_contiguous.items()
        }
        self._dataset_metadata = DatasetMetadata(
            name=_xrfv2_card().display_name,
            sample_count=len(self.samples),
            gestures=sorted(self.label_mapping),
            device_type=_require_string(
                _SIGNAL_PROFILE,
                "device_type",
                owner="XRFV2 schema.yaml signal_profile",
            ),
            center_freq=XRFV2_WIFI_CENTER_FREQ,
            bandwidth=XRFV2_WIFI_BANDWIDTH,
            extra={
                "split": split,
                "layout": self.layout,
                "modality": "wifi",
                "variant": self.variant_name,
                "variant_key": self.variant_key,
            },
        )

    def _init_processed_layout(self) -> None:
        self.data_path = self.dataset_path / f"{self.split}_data.h5"
        self.label_path = self.dataset_path / f"{self.split}_label.json"
        self.info_path = self.dataset_path / "info.json"

        self.info = _load_info_file(self.info_path)
        label_payload = json.loads(self.label_path.read_text(encoding="utf-8"))
        wifi_labels = label_payload.get("wifi")
        if not isinstance(wifi_labels, dict):
            raise ValueError(f"Expected 'wifi' labels in {self.label_path}")

        self.new_mapping, self.id_to_action = _resolve_xrfv2_label_mapping(self.info)
        for raw_index, segments in sorted(wifi_labels.items(), key=lambda item: int(item[0])):
            if not isinstance(segments, list):
                continue
            for segment_index, segment in enumerate(segments):
                start, end, action_id = _parse_segment(segment)
                if end <= start:
                    continue
                mapped_action_id = self.new_mapping.get(action_id, action_id)
                self.samples.append(
                    XRFV2ClipSample(
                        split=self.split,
                        base_index=int(raw_index),
                        segment_index=segment_index,
                        start=start,
                        end=end,
                        action_id=action_id,
                        mapped_action_id=mapped_action_id,
                    )
                )

    def _init_raw_layout(self) -> None:
        raw_root = _resolve_xrfv2_raw_root(self.dataset_path)
        if raw_root is None:
            raise FileNotFoundError(
                "XRF V2 data not found. Expected either processed "
                f"{self.dataset_path / f'{self.split}_data.h5'} or raw "
                f"{self.dataset_path / 'WWADL_open' / 'file_records.csv'}."
            )

        self.raw_root = raw_root
        info_candidates = [self.dataset_path / "info.json", raw_root / "info.json"]
        self.info_path = next((path for path in info_candidates if path.exists()), None)
        self.info = _load_info_file(self.info_path)
        self.new_mapping, self.id_to_action = _resolve_xrfv2_label_mapping(self.info)

        records = _load_raw_records(raw_root / "file_records.csv")
        records = _select_raw_records_for_split(
            records,
            raw_root=raw_root,
            split=self.split,
        )

        for record in records:
            wifi_path = raw_root / record.wifi_relative_path
            with h5py.File(wifi_path, "r") as handle:
                if "label" not in handle:
                    raise ValueError(f"Expected 'label' dataset in {wifi_path}")
                labels = handle["label"][:]
            for row in labels:
                segment_index, action_id, start, end = _parse_raw_segment_row(row)
                if end <= start:
                    continue
                mapped_action_id = self.new_mapping.get(action_id, action_id)
                self.samples.append(
                    XRFV2ClipSample(
                        split=self.split,
                        segment_index=segment_index,
                        start=start,
                        end=end,
                        action_id=action_id,
                        mapped_action_id=mapped_action_id,
                        source_file=record.wifi_relative_path,
                        volunteer_id=record.volunteer_id,
                        scene_id=record.scene_id,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def materialize_sample(self, sample: XRFV2ClipSample) -> tuple[RadioTensor, int]:
        if sample.base_index is not None:
            if self.data_path is None:
                raise ValueError("Processed XRF V2 layout is not initialized")
            signal = self.reader.read(
                self.data_path,
                dataset_key="wifi",
                base_index=sample.base_index,
                start=sample.start,
                end=sample.end,
            )
        else:
            if self.raw_root is None or sample.source_file is None:
                raise ValueError("Raw XRF V2 layout is not initialized")
            signal = self.reader.read(
                self.raw_root / sample.source_file,
                dataset_key="amp",
                start=sample.start,
                end=sample.end,
            )
        signal = _augment_xrfv2_sample_metadata(
            signal,
            sample,
            variant_name=self.variant_name,
            variant_key=self.variant_key,
        )
        label = self.class_id_to_contiguous[sample.mapped_action_id]
        return signal, label

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        sample = self.samples[idx]
        return self.materialize_sample(sample)

    def get_labels(self) -> list[int]:
        return [self.class_id_to_contiguous[sample.mapped_action_id] for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        return self.label_mapping.copy()

    def get_sample(self, idx: int) -> XRFV2ClipSample:
        return self.samples[idx]

    def get_sample_id(self, idx: int) -> str:
        return self.samples[idx].sample_id()

    def get_group_id(self, idx: int) -> str:
        return self.samples[idx].group_id()

    def metadata_rows(self) -> list[dict[str, object]]:
        return [
            {
                "sample_index": int(index),
                "sample_id": sample.sample_id(),
                "sample_group_id": sample.group_id(),
                "split": str(sample.split),
                "segment_index": int(sample.segment_index),
                "start": int(sample.start),
                "end": int(sample.end),
                "action_id": int(sample.action_id),
                "mapped_action_id": int(sample.mapped_action_id),
                "label": int(self.class_id_to_contiguous[sample.mapped_action_id]),
                "volunteer_id": sample.volunteer_id,
                "scene_id": sample.scene_id,
                "source_file": sample.source_file,
            }
            for index, sample in enumerate(self.samples)
        ]

    def sample_describe_tree(self):
        sample, _ = self[0]
        return sample.describe_tree().with_name("sample")

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        return self._dataset_metadata

    @property
    def dataset_card(self) -> DatasetCard:
        return _xrfv2_card()


class XRFV2Dataset(Dataset[tuple[RadioTensor, int]]):
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        modality: str | None = None,
        variant: str,
        split: str = "train",
    ) -> None:
        requested_modality = None if modality in {None, ""} else str(modality).strip().lower()
        resolved_variant = str(variant).strip()
        if not resolved_variant:
            raise ValueError("XRFV2Dataset requires an explicit canonical variant binding id.")
        variant_config = _resolve_variant_config(resolved_variant)
        resolved_modality = str(variant_config.get("modality", "")).strip().lower()
        if resolved_modality != "wifi":
            raise ValueError("XRFV2Dataset currently supports only modality='wifi'")
        if requested_modality not in {None, "wifi"}:
            raise ValueError("XRFV2Dataset currently supports only modality='wifi'")
        self.modality = resolved_modality
        self.variant = resolved_variant
        self.variant_key = str(variant_config.get("variant_key", resolved_variant))
        self._dataset = _XRFV2WiFiDataset(
            dataset_path,
            variant=resolved_variant,
            split=split,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        return self._dataset[idx]

    def __getattr__(self, name: str) -> object:
        return getattr(self._dataset, name)


def open_xrfv2_dataset(
    dataset_path: str | Path | None = None,
    *,
    modality: str | None = None,
    variant: str,
    split: str = "train",
) -> XRFV2Dataset:
    return XRFV2Dataset(
        dataset_path,
        modality=modality,
        variant=variant,
        split=split,
    )


__all__ = [
    "XRFV2Dataset",
    "XRFV2_WIFI_BANDWIDTH",
    "XRFV2_WIFI_CENTER_FREQ",
    "open_xrfv2_dataset",
]
