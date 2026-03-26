"""WiFi ingest helpers and dataset materialization for the builtin OctoNet dataset."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor

DATASET_ID = "octonet"


def _octonet_card() -> DatasetCard:
    return get_dataset_card(DATASET_ID)


def _octonet_schema_fields() -> dict[str, Any]:
    return dict(get_dataset_schema_fields(DATASET_ID))


def _octonet_mapping_field(field_name: str) -> dict[str, Any]:
    raw = _octonet_schema_fields().get(field_name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"OctoNet schema.yaml field '{field_name}' must be a mapping")
    return dict(raw)


def _octonet_rf_profile(modality: str) -> dict[str, Any]:
    profiles = _octonet_mapping_field("rf_profiles")
    payload = profiles.get(modality)
    if not isinstance(payload, dict):
        raise ValueError(f"OctoNet schema.yaml rf_profiles.{modality!r} must be a mapping")
    return dict(payload)


_OCTONET_WIFI_PROFILE = _octonet_rf_profile("wifi")
OCTONET_WIFI_CENTER_FREQ = float(
    _OCTONET_WIFI_PROFILE.get("center_freq_hz", 0.0)
)
OCTONET_WIFI_BANDWIDTH = float(
    _OCTONET_WIFI_PROFILE.get("bandwidth_hz", 0.0)
)
OCTONET_MODALITY_READER_IDS = {
    str(key): str(value)
    for key, value in _octonet_mapping_field("reader_hints").items()
}


def safe_strip(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_optional_int(raw_value: object) -> int | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _octonet_started_at_token(
    row: dict[str, object],
    *,
    relative_path: str,
) -> str:
    for field_name in ("started_at", "start_time", "recording_time"):
        value = safe_strip(row.get(field_name))
        if value:
            return value
    fallback = Path(relative_path).stem.strip()
    if fallback:
        return f"path:{fallback}"
    return "path:unknown"


@dataclass(frozen=True)
class OctonetWiFiSample:
    relative_path: str
    user_id: int
    activity: str
    node_id: int
    started_at: str
    subject_id: int | None = None
    exp_id: int | None = None
    scene_id: int | None = None
    scene_name: str | None = None
    sample_profile: str | None = None

    def sample_id(self) -> str:
        return f"{self.started_at}|user={self.user_id}|activity={self.activity}|node={self.node_id}"

    def group_id(self) -> str:
        return f"{self.started_at}|user={self.user_id}|activity={self.activity}"


def load_octonet_wifi_file_as_radiotensor(
    path: Path,
    *,
    sample_id: str,
    dataset_name: str,
    node_id: int | None,
    activity: str,
    subject_id: int | None,
    exp_id: int | None,
    scene_id: int | None,
    sample_profile: str | None,
    record_id: str | None = None,
    session_id: str | None = None,
    user_id: int | None = None,
    reader: Any | None = None,
) -> RadioTensor:
    active_reader = reader or io_readers.load(OCTONET_MODALITY_READER_IDS["wifi"])
    signal = active_reader.read(path)
    signal.metadata.extra.update(
        {
            "dataset": dataset_name,
            "sample_id": sample_id,
            "record_id": record_id,
            "session_id": session_id,
            "node_id": node_id,
            "activity": activity,
            "subject_id": subject_id,
            "user_id": user_id,
            "exp_id": exp_id,
            "scene_id": scene_id,
            "sample_profile": sample_profile,
        }
    )
    return signal


class OctonetWiFiDataset(Dataset[tuple[RadioTensor, int]]):
    """WiFi-only OctoNet dataset materialized from the builtin dataset root."""

    def __init__(
        self,
        dataset_path: str,
        *,
        node_id: int = 1,
        users: list[int] | None = None,
        activities: list[str] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.node_id = node_id
        self.users = users
        self.activities = activities
        self.reader = io_readers.load(OCTONET_MODALITY_READER_IDS["wifi"])
        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError("No Octonet WiFi samples matched the requested filters")

        unique_activities = sorted({sample.activity for sample in self.samples})
        self.label_mapping = {activity: idx for idx, activity in enumerate(unique_activities)}
        self._dataset_metadata = DatasetMetadata(
            name="OctonetWiFi",
            sample_count=len(self.samples),
            users=sorted({sample.user_id for sample in self.samples}),
            gestures=unique_activities,
            device_type=str(_OCTONET_WIFI_PROFILE.get("capture_device", "AX200")),
            center_freq=OCTONET_WIFI_CENTER_FREQ,
            bandwidth=OCTONET_WIFI_BANDWIDTH,
            extra={"node_id": node_id},
        )

    def _load_samples(self) -> list[OctonetWiFiSample]:
        meta_path = self.dataset_path / "cut_manual.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"cut_manual.csv not found at {meta_path}")
        wifi_column = f"node_{self.node_id}_wifi_data_path"
        samples: list[OctonetWiFiSample] = []
        with open(meta_path, newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if wifi_column not in row or not row[wifi_column]:
                    continue
                user_id = int(row["user_id"])
                subject_id = parse_optional_int(row.get("subject_id"))
                exp_id = parse_optional_int(row.get("exp_id"))
                scene_id = parse_optional_int(row.get("scene_id"))
                activity = row["activity"].strip()
                scene_name = safe_strip(row.get("scene_name")) or None
                if self.users is not None and user_id not in self.users:
                    continue
                if self.activities is not None and activity not in self.activities:
                    continue
                samples.append(
                    OctonetWiFiSample(
                        relative_path=row[wifi_column].strip(),
                        user_id=user_id,
                        activity=activity,
                        node_id=self.node_id,
                        started_at=_octonet_started_at_token(
                            row,
                            relative_path=row[wifi_column].strip(),
                        ),
                        subject_id=subject_id,
                        exp_id=exp_id,
                        scene_id=scene_id,
                        scene_name=scene_name,
                        sample_profile=safe_strip(row.get("sample_profile")) or None,
                    )
                )
        return samples

    def _load_radiotensor(self, sample: OctonetWiFiSample) -> RadioTensor:
        return load_octonet_wifi_file_as_radiotensor(
            self.dataset_path / sample.relative_path,
            sample_id=sample.sample_id(),
            dataset_name="octonet",
            node_id=sample.node_id,
            activity=sample.activity,
            subject_id=sample.subject_id,
            exp_id=sample.exp_id,
            scene_id=sample.scene_id,
            sample_profile=sample.sample_profile,
            user_id=sample.user_id,
            reader=self.reader,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        sample = self.samples[idx]
        return self._load_radiotensor(sample), self.label_mapping[sample.activity]

    def get_labels(self) -> list[int]:
        return [self.label_mapping[sample.activity] for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        return self.label_mapping.copy()

    def get_sample(self, idx: int) -> OctonetWiFiSample:
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
                "node_id": int(sample.node_id),
                "activity": str(sample.activity),
                "label": int(self.label_mapping[sample.activity]),
                "subject_id": int(sample.subject_id) if sample.subject_id is not None else None,
                "exp_id": int(sample.exp_id) if sample.exp_id is not None else None,
                "scene_id": int(sample.scene_id) if sample.scene_id is not None else None,
                "sample_profile": str(sample.sample_profile)
                if sample.sample_profile is not None
                else None,
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
        return _octonet_card()


def detect_octonet_wifi_node_id(dataset_root: Path) -> int:
    meta_path = dataset_root / "cut_manual.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"cut_manual.csv not found at {meta_path}")
    with open(meta_path, newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        node_ids = sorted(
            {
                int(match.group(1))
                for field_name in fieldnames
                for match in [re.fullmatch(r"node_(\d+)_wifi_data_path", field_name or "")]
                if match is not None
            }
        )
        if not node_ids:
            raise ValueError(f"No OctoNet WiFi columns found in {meta_path}")
        availability = {node_id: False for node_id in node_ids}
        for row in reader:
            for node_id in node_ids:
                if safe_strip(row.get(f"node_{node_id}_wifi_data_path")):
                    availability[node_id] = True
        available_nodes = [node_id for node_id in node_ids if availability[node_id]]
        if not available_nodes:
            raise ValueError(f"No OctoNet WiFi samples found in {meta_path}")
        return available_nodes[0]


__all__ = [
    "DATASET_ID",
    "OCTONET_WIFI_BANDWIDTH",
    "OCTONET_WIFI_CENTER_FREQ",
    "OctonetWiFiDataset",
    "OctonetWiFiSample",
    "OCTONET_MODALITY_READER_IDS",
    "detect_octonet_wifi_node_id",
    "load_octonet_wifi_file_as_radiotensor",
    "parse_optional_int",
]
