"""Ingest owner for the FallDar builtin dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from octosense.datasets.base import resolve_dataset_root, resolve_declared_variant
from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor


@lru_cache(maxsize=1)
def _falldar_card() -> DatasetCard:
    return get_dataset_card("falldar")


def _falldar_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields("falldar"))


def _falldar_rf_profile() -> dict[str, object]:
    profile = _falldar_schema_fields().get("rf_profile")
    if not isinstance(profile, dict):
        raise ValueError("FallDar schema.yaml must define mapping field 'rf_profile'")
    return dict(profile)


def _falldar_signal_profile() -> dict[str, object]:
    payload = _falldar_schema_fields().get("signal_profile")
    if not isinstance(payload, dict):
        raise ValueError("FallDar schema.yaml must define mapping field 'signal_profile'")
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


def _falldar_variant_payload(variant: str | None = None) -> dict[str, object]:
    variant_key = resolve_declared_variant(
        variant,
        supported_variants=list(_falldar_card().supported_variants or tuple(_falldar_card().variants)),
        owner="FallDar",
    )
    payload = _falldar_card().variants.get(variant_key)
    if not isinstance(payload, dict):
        available = ", ".join(sorted(_falldar_card().variants))
        raise ValueError(f"Unsupported FallDar variant '{variant}'. Available: {available}")
    return payload


def _falldar_labels() -> dict[str, int]:
    payload = _falldar_schema_fields().get("label_semantics")
    if not isinstance(payload, dict):
        raise ValueError("FallDar schema.yaml must define mapping field 'label_semantics'")
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("FallDar schema.yaml must define label_semantics.labels")
    return {str(name): int(index) for name, index in labels.items()}


def _falldar_csv_columns(variant: str | None = None) -> dict[str, str]:
    payload = _falldar_variant_payload(variant).get("csv_columns", {})
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if value not in {None, ""}
    }


def _falldar_required_path(variant: str, key: str) -> str:
    payload = _falldar_variant_payload(variant).get("paths", {})
    if not isinstance(payload, dict):
        raise ValueError(f"FallDar variant '{variant}' is missing paths.{key}")
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"FallDar variant '{variant}' is missing paths.{key}")
    return str(value)


def _falldar_channel(variant: str) -> int:
    payload = _falldar_variant_payload(variant).get("reader", {})
    if not isinstance(payload, dict):
        raise ValueError(f"FallDar variant '{variant}' is missing reader.channel")
    value = payload.get("channel")
    if value in {None, ""}:
        raise ValueError(f"FallDar variant '{variant}' is missing reader.channel")
    return int(value)


FALLDAR_CENTER_FREQ = _require_float(_falldar_rf_profile(), "center_freq_hz", owner="FallDar schema.yaml rf_profile")
FALLDAR_BANDWIDTH = _require_float(_falldar_rf_profile(), "bandwidth_hz", owner="FallDar schema.yaml rf_profile")
FALLDAR_NOMINAL_SAMPLE_RATE = _require_float(
    _falldar_rf_profile(),
    "nominal_sample_rate_hz",
    owner="FallDar schema.yaml rf_profile",
)


def _parse_index_spec(text: str) -> set[int]:
    value = str(text).strip()
    if not value or value.lower() == "non":
        return set()
    values: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", 1)
            start = int(start_text)
            end = int(end_text)
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return values


@dataclass(frozen=True)
class FallDarSample:
    date: str
    user_name: str
    clip_index: int
    receiver_id: int
    relative_path: str
    label_name: str

    def sample_id(self) -> str:
        return (
            f"{self.date}|user={self.user_name}|clip={self.clip_index}"
            f"|rx={self.receiver_id}|label={self.label_name}"
        )

    def group_id(self) -> str:
        return f"{self.date}|user={self.user_name}|clip={self.clip_index}|label={self.label_name}"


class FallDarDataset(Dataset[tuple[RadioTensor, int]]):
    """FallDar raw WiFi CSI adapter using ``metas.csv`` for fall/nonfall labels."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        variant: str | None = None,
        reader: Any | None = None,
    ) -> None:
        self.root = resolve_dataset_root("falldar", override=dataset_path)
        self.variant = resolve_declared_variant(
            variant,
            supported_variants=list(
                _falldar_card().supported_variants or tuple(_falldar_card().variants)
            ),
            owner="FallDar",
        )
        variant_payload = _falldar_variant_payload(self.variant)
        raw_dir = _falldar_required_path(self.variant, "raw_dir")
        metadata_csv = _falldar_required_path(self.variant, "metadata_csv")
        self.reader = reader or io_readers.load("wifi/iwl5300", channel=_falldar_channel(self.variant))
        self.raw_root = self.root / raw_dir
        self.meta_path = self.root / metadata_csv
        if not self.raw_root.exists():
            raise FileNotFoundError(f"FallDar raw_data directory not found: {self.raw_root}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"FallDar metadata CSV not found: {self.meta_path}")

        label_mapping_by_date: dict[str, dict[int, str]] = {}
        csv_columns = _falldar_csv_columns(self.variant)
        folder_name_column = csv_columns.get("folder_name", "Folder_name")
        fall_index_column = csv_columns.get("fall_index", "Fall_index")
        nonfall_index_column = csv_columns.get("nonfall_index", "Nonfall_index")
        with self.meta_path.open("r", encoding="utf-8-sig") as handle:
            reader_csv = csv.DictReader(handle)
            for raw_row in reader_csv:
                row = {str(key).strip(): value for key, value in raw_row.items()}
                date = str(row[folder_name_column]).strip()
                fall_indices = _parse_index_spec(str(row[fall_index_column]))
                nonfall_indices = _parse_index_spec(str(row[nonfall_index_column]))
                label_by_clip = label_mapping_by_date.setdefault(date, {})
                for index in nonfall_indices:
                    label_by_clip[index] = "nonfall"
                for index in fall_indices:
                    label_by_clip[index] = "fall"

        samples: list[FallDarSample] = []
        for date_dir in sorted(self.raw_root.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            clip_labels = label_mapping_by_date.get(date, {})
            for file_path in sorted(date_dir.glob("*.dat")):
                stem_tokens = file_path.stem.split("-")
                if len(stem_tokens) < 4:
                    continue
                user_name = stem_tokens[0]
                clip_index = int(stem_tokens[1])
                receiver_id = int(stem_tokens[2])
                label_name = clip_labels.get(clip_index)
                if label_name is None:
                    continue
                samples.append(
                    FallDarSample(
                        date=date,
                        user_name=user_name,
                        clip_index=clip_index,
                        receiver_id=receiver_id,
                        relative_path=str(file_path.relative_to(self.root)),
                        label_name=label_name,
                    )
                )
        if not samples:
            raise ValueError(f"No labeled FallDar raw CSI samples found under {self.raw_root}")
        self.samples = samples
        self._label_mapping = _falldar_labels()
        self.dataset_metadata = DatasetMetadata(
            name="FallDar",
            sample_count=len(self.samples),
            gestures=sorted(self._label_mapping, key=self._label_mapping.get),
            collection_dates=sorted({sample.date for sample in self.samples}),
            device_type=_require_string(
                _falldar_signal_profile(),
                "device_type",
                owner="FallDar schema.yaml signal_profile",
            ),
            center_freq=FALLDAR_CENTER_FREQ,
            bandwidth=FALLDAR_BANDWIDTH,
            nominal_sample_rate=FALLDAR_NOMINAL_SAMPLE_RATE,
            extra={
                "variant": self.variant,
                "source_view": raw_dir,
                "meta_csv": str(self.meta_path),
                "variant_description": variant_payload.get("description"),
            },
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        sample = self.samples[idx]
        tensor = self.reader.read(self.root / sample.relative_path)
        tensor.metadata.extra.update(
            {
                "dataset": "falldar",
                "date": sample.date,
                "user_name": sample.user_name,
                "clip_index": sample.clip_index,
                "receiver_id": sample.receiver_id,
                "sample_id": sample.sample_id(),
                "label_name": sample.label_name,
            }
        )
        return tensor, self._label_mapping[sample.label_name]

    def get_labels(self) -> list[int]:
        return [self._label_mapping[sample.label_name] for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        return dict(self._label_mapping)

    def get_sample_id(self, idx: int) -> str:
        return self.samples[idx].sample_id()

    def metadata_rows(self) -> list[dict[str, object]]:
        return [
            {
                "date": sample.date,
                "user_name": sample.user_name,
                "clip_index": sample.clip_index,
                "receiver_id": sample.receiver_id,
                "label": self._label_mapping[sample.label_name],
                "label_name": sample.label_name,
            }
            for sample in self.samples
        ]

    def sample_describe_tree(self):
        first_sample, _ = self[0]
        return first_sample.describe_tree().with_name("sample")


def falldar_dataset_card() -> DatasetCard:
    return get_dataset_card("falldar")


def load_falldar_dataset(
    dataset_path: str | Path | None = None,
    *,
    variant: str | None = None,
    reader: Any | None = None,
) -> FallDarDataset:
    return FallDarDataset(
        dataset_path,
        variant=variant,
        reader=reader,
    )


__all__ = [
    "FALLDAR_BANDWIDTH",
    "FALLDAR_CENTER_FREQ",
    "FALLDAR_NOMINAL_SAMPLE_RATE",
    "FallDarDataset",
    "FallDarSample",
    "load_falldar_dataset",
    "falldar_dataset_card",
]
