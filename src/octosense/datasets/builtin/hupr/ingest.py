"""Ingest owner for the HuPR builtin dataset."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from octosense.datasets.base import resolve_dataset_root, resolve_declared_variant
from octosense.datasets.catalog import (
    DatasetCard,
    get_dataset_card,
    get_dataset_schema_fields,
)
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io.readers import load as load_reader
from octosense.io.tensor import RadioTensor


def _hupr_card() -> DatasetCard:
    return get_dataset_card("hupr")


def _hupr_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields("hupr"))


def _hupr_rf_profile() -> dict[str, object]:
    payload = _hupr_schema_fields().get("rf_profile")
    if not isinstance(payload, dict):
        raise ValueError("HuPR schema.yaml must define mapping field 'rf_profile'")
    return dict(payload)


def _hupr_signal_profile() -> dict[str, object]:
    payload = _hupr_schema_fields().get("signal_profile")
    if not isinstance(payload, dict):
        raise ValueError("HuPR schema.yaml must define mapping field 'signal_profile'")
    return dict(payload)


def _hupr_label_semantics() -> dict[str, object]:
    payload = _hupr_schema_fields().get("label_semantics")
    if not isinstance(payload, dict):
        raise ValueError("HuPR schema.yaml must define mapping field 'label_semantics'")
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


def _hupr_declared_concrete_target_layout() -> dict[str, object]:
    payload = _hupr_label_semantics().get("target_schema")
    if not isinstance(payload, dict):
        raise ValueError("HuPR schema.yaml label_semantics.target_schema must be a mapping")
    return dict(payload)


def _hupr_resolved_variant(variant: str | None) -> str:
    return resolve_declared_variant(
        variant,
        supported_variants=list(_hupr_card().supported_variants or tuple(_hupr_card().variants)),
        owner="HuPR",
    )


def _hupr_variant_payload(variant: str) -> dict[str, object]:
    payload = _hupr_card().variants.get(variant)
    if not isinstance(payload, dict):
        available = ", ".join(sorted(_hupr_card().variants))
        raise ValueError(f"Unsupported HuPR variant '{variant}'. Available variants: {available}")
    return payload


def _hupr_coord_units() -> dict[str, str]:
    payload = _hupr_signal_profile().get("coord_units")
    if not isinstance(payload, dict):
        raise ValueError("HuPR schema.yaml signal_profile.coord_units must be a mapping")
    return {str(axis): str(unit) for axis, unit in payload.items()}


def _hupr_required_path(variant: str, key: str) -> str:
    payload = _hupr_variant_payload(variant).get("paths", {})
    if not isinstance(payload, dict):
        raise ValueError(f"HuPR variant '{variant}' is missing paths.{key}")
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"HuPR variant '{variant}' is missing paths.{key}")
    return str(value)


def _hupr_required_payload_key(variant: str, key: str) -> str:
    payload = _hupr_variant_payload(variant).get("payload_keys", {})
    if not isinstance(payload, dict):
        raise ValueError(f"HuPR variant '{variant}' is missing payload_keys.{key}")
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"HuPR variant '{variant}' is missing payload_keys.{key}")
    return str(value)


def _hupr_required_target_key(variant: str, key: str) -> str:
    payload = _hupr_variant_payload(variant).get("target_keys", {})
    if not isinstance(payload, dict):
        raise ValueError(f"HuPR variant '{variant}' is missing target_keys.{key}")
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"HuPR variant '{variant}' is missing target_keys.{key}")
    return str(value)


HUPR_CENTER_FREQ = _require_float(_hupr_rf_profile(), "center_freq_hz", owner="HuPR schema.yaml rf_profile")
HUPR_BANDWIDTH = _require_float(_hupr_rf_profile(), "bandwidth_hz", owner="HuPR schema.yaml rf_profile")


def _hupr_require_series(
    payload: dict[str, Any],
    *,
    key: str,
    file_path: Path,
) -> list[Any]:
    series = payload.get(key)
    if not isinstance(series, list):
        raise ValueError(
            f"HuPR payload field {key!r} in {file_path} must be a list, got {type(series)!r}"
        )
    return series


def _hupr_indexable_sample_count(
    payload: dict[str, Any],
    *,
    file_path: Path,
    variant: str,
) -> int:
    horizontal_series = _hupr_require_series(
        payload,
        key=_hupr_required_payload_key(variant, "horizontal"),
        file_path=file_path,
    )
    vertical_series = _hupr_require_series(
        payload,
        key=_hupr_required_payload_key(variant, "vertical"),
        file_path=file_path,
    )
    labels = _hupr_require_series(
        payload,
        key=_hupr_required_payload_key(variant, "labels"),
        file_path=file_path,
    )
    if len(horizontal_series) != len(vertical_series):
        raise ValueError(
            "HuPR horizontal/vertical radar series length mismatch during indexing: "
            f"{file_path} has horizontal_count={len(horizontal_series)} "
            f"and vertical_count={len(vertical_series)}"
        )
    if len(labels) < len(horizontal_series):
        raise ValueError(
            "HuPR labels do not cover all indexed radar samples: "
            f"{file_path} has sample_count={len(horizontal_series)} and labels_count={len(labels)}"
        )
    return len(horizontal_series)


@dataclass(frozen=True)
class HuPRSample:
    file_path: str
    index: int

    def sample_id(self) -> str:
        return f"{Path(self.file_path).name}#idx={self.index}"


@dataclass(frozen=True)
class HuPRDecodedSample:
    label_payload: Any


class HuPRDataset(Dataset[tuple[RadioTensor, dict[str, torch.Tensor]]]):
    """HuPR dataset exposing radar maps plus pose targets."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        variant: str | None = None,
        file_indexes: list[int] | None = None,
        task_binding: str | None = None,
        task_kind: str | None = None,
        target_kind: str | None = None,
        target_schema: dict[str, object] | None = None,
    ) -> None:
        self.dataset_path = resolve_dataset_root("hupr", override=dataset_path)
        self.variant = _hupr_resolved_variant(variant)
        self.file_indexes = set(file_indexes or [])
        self.task_binding = None if task_binding in {None, ""} else str(task_binding)
        self.task_kind = (
            str(task_kind)
            if task_kind not in {None, ""}
            else _require_string(
                _hupr_label_semantics(),
                "kind",
                owner="HuPR schema.yaml label_semantics",
            )
        )
        self.target_kind = str(target_kind or "joints")
        self._concrete_target_layout = (
            dict(target_schema)
            if isinstance(target_schema, dict)
            else self.get_target_schema()
        )
        self.samples = self._index_samples()
        if not self.samples:
            raise ValueError("No HuPR samples matched the requested file indexes")
        self._reader_id = _require_string(
            _hupr_signal_profile(),
            "reader_id",
            owner="HuPR schema.yaml signal_profile",
        )
        self._signal_reader = load_reader(self._reader_id)
        self._signal_capture_device = _require_string(
            _hupr_signal_profile(),
            "capture_device",
            owner="HuPR schema.yaml signal_profile",
        )
        self._signal_modality = _require_string(
            _hupr_signal_profile(),
            "signal_modality",
            owner="HuPR schema.yaml signal_profile",
        )
        self._dataset_metadata = DatasetMetadata(
            name="HuPR",
            sample_count=len(self.samples),
            device_type=_require_string(
                _hupr_signal_profile(),
                "device_type",
                owner="HuPR schema.yaml signal_profile",
            ),
            center_freq=HUPR_CENTER_FREQ,
            bandwidth=HUPR_BANDWIDTH,
            extra={
                "task": self.task_kind,
                "task_binding": self.task_binding,
                "variant": self.variant,
                "target_kind": self.target_kind,
                "dataset_target_layout": dict(self._concrete_target_layout),
            },
        )

    def _index_samples(self) -> list[HuPRSample]:
        radar_dir = self.dataset_path / _hupr_required_path(self.variant, "radar_dir")
        if not radar_dir.exists():
            raise FileNotFoundError(f"radar_maps not found at {radar_dir}")

        samples: list[HuPRSample] = []
        for radar_file in sorted(radar_dir.glob(_hupr_required_path(self.variant, "file_pattern"))):
            if self.file_indexes:
                try:
                    file_idx = int(radar_file.stem.split("_")[-1])
                except ValueError:
                    continue
                if file_idx not in self.file_indexes:
                    continue
            with open(radar_file, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"HuPR pickle payload must be a mapping, got {type(payload)!r} in {radar_file}"
                )
            count = _hupr_indexable_sample_count(
                payload,
                file_path=radar_file,
                variant=self.variant,
            )
            for index in range(count):
                samples.append(HuPRSample(file_path=str(radar_file), index=index))
        return samples

    def _decode_sample(self, sample: HuPRSample) -> HuPRDecodedSample:
        with open(sample.file_path, "rb") as f:
            payload: dict[str, Any] = pickle.load(f)
        labels_key = _hupr_required_payload_key(self.variant, "labels")
        label = payload[labels_key][sample.index]
        return HuPRDecodedSample(label_payload=label)

    def _materialize_signal(self, sample: HuPRSample) -> RadioTensor:
        signal = self._signal_reader.read(
            sample.file_path,
            sample_index=sample.index,
            horizontal_key=_hupr_required_payload_key(self.variant, "horizontal"),
            vertical_key=_hupr_required_payload_key(self.variant, "vertical"),
            signal_modality=self._signal_modality,
            capture_device=self._signal_capture_device,
            center_freq_hz=HUPR_CENTER_FREQ,
            bandwidth_hz=HUPR_BANDWIDTH,
            coord_units=_hupr_coord_units(),
            extra_metadata={
                "dataset": "hupr",
                "variant": self.variant,
                "sample_id": sample.sample_id(),
            },
        )
        return signal

    def _materialize_target(self, decoded: HuPRDecodedSample) -> dict[str, torch.Tensor]:
        label = decoded.label_payload
        return {
            "joints": torch.tensor(
                np.asarray(label[_hupr_required_target_key(self.variant, "joints")]),
                dtype=torch.float32,
            ),
            "bbox": torch.tensor(
                np.asarray(label[_hupr_required_target_key(self.variant, "bbox")]),
                dtype=torch.float32,
            ),
        }

    def materialize_sample(self, sample: HuPRSample) -> tuple[RadioTensor, dict[str, torch.Tensor]]:
        decoded = self._decode_sample(sample)
        return (
            self._materialize_signal(sample),
            self._materialize_target(decoded),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, dict[str, torch.Tensor]]:
        return self.materialize_sample(self.samples[idx])

    def get_sample(self, idx: int) -> HuPRSample:
        return self.samples[idx]

    def get_sample_id(self, idx: int) -> str:
        return self.samples[idx].sample_id()

    def get_group_id(self, idx: int) -> str:
        return str(Path(self.samples[idx].file_path).name)

    def get_target_schema(self) -> dict[str, object]:
        payload = getattr(self, "_concrete_target_layout", None)
        if isinstance(payload, dict):
            return dict(payload)
        return _hupr_declared_concrete_target_layout()

    def metadata_rows(self) -> list[dict[str, object]]:
        return [
            {
                "sample_index": int(index),
                "sample_id": sample.sample_id(),
                "file_path": str(sample.file_path),
                "file_name": str(Path(sample.file_path).name),
                "file_index": int(sample.index),
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
        return _hupr_card()


__all__ = [
    "HUPR_BANDWIDTH",
    "HUPR_CENTER_FREQ",
    "HuPRDataset",
    "HuPRSample",
]
