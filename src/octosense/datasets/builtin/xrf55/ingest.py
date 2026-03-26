"""Ingest owner for the XRF55 builtin dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from torch.utils.data import Dataset

from octosense.datasets.base import resolve_dataset_root
from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor


@lru_cache(maxsize=1)
def _xrf55_card() -> DatasetCard:
    return get_dataset_card("xrf55")


def _xrf55_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields("xrf55"))


def _xrf55_rf_profile() -> dict[str, object]:
    profile = _xrf55_schema_fields().get("rf_profile")
    if not isinstance(profile, dict):
        raise ValueError("XRF55 schema.yaml must define mapping field 'rf_profile'")
    return dict(profile)


def _xrf55_signal_profile() -> dict[str, object]:
    profile = _xrf55_schema_fields().get("signal_profile")
    if not isinstance(profile, dict):
        raise ValueError("XRF55 schema.yaml must define mapping field 'signal_profile'")
    return dict(profile)


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


def _xrf55_variant_payload(variant_key: str) -> dict[str, object]:
    resolved_variant_key = str(variant_key).strip()
    if not resolved_variant_key:
        raise ValueError("XRF55 variant must be a non-empty canonical binding id.")

    payload = _xrf55_card().variants.get(resolved_variant_key)
    if not isinstance(payload, dict):
        available = ", ".join(sorted(_xrf55_card().variants))
        raise ValueError(
            f"Unsupported XRF55 variant '{resolved_variant_key}'. Available variants: {available}"
        )
    return payload


XRF55_WIFI_CENTER_FREQ = _require_float(
    _xrf55_rf_profile(),
    "center_freq_hz",
    owner="XRF55 schema.yaml rf_profile",
)
XRF55_WIFI_BANDWIDTH = _require_float(
    _xrf55_rf_profile(),
    "bandwidth_hz",
    owner="XRF55 schema.yaml rf_profile",
)


@dataclass(frozen=True)
class XRF55RawSample:
    relative_path: str
    label_name: str
    split: str | None = None
    subject_id: str | None = None

    def sample_id(self) -> str:
        parts = [self.relative_path, f"label={self.label_name}"]
        if self.split:
            parts.append(f"split={self.split}")
        if self.subject_id:
            parts.append(f"subject={self.subject_id}")
        return "|".join(parts)

    def group_id(self) -> str:
        parts = [f"label={self.label_name}"]
        if self.split:
            parts.append(f"split={self.split}")
        if self.subject_id:
            parts.append(f"subject={self.subject_id}")
        return "|".join(parts)


def _load_manifest_samples(dataset_path: Path) -> list[XRF55RawSample]:
    manifest_path = dataset_path / "manifest.csv"
    if not manifest_path.exists():
        raise ValueError(
            f"XRF55 canonical manifest is required: {manifest_path.as_posix()}"
        )
    samples: list[XRF55RawSample] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relative_path = (row.get("relative_path") or "").strip()
            label_name = (row.get("label_name") or "").strip()
            if not relative_path or not label_name:
                continue
            samples.append(
                XRF55RawSample(
                    relative_path=relative_path,
                    label_name=label_name,
                    split=(row.get("split") or "").strip() or None,
                    subject_id=(row.get("subject_id") or "").strip() or None,
                )
            )
    return samples


def _augment_xrf55_sample_metadata(
    signal: RadioTensor,
    sample: XRF55RawSample,
    *,
    variant_name: str,
    variant_key: str,
) -> RadioTensor:
    metadata = signal.metadata.copy()
    metadata.extra.update(
        {
            "dataset": _xrf55_card().dataset_id,
            "sample_id": sample.sample_id(),
            "label_name": sample.label_name,
            "relative_path": sample.relative_path,
            "split": sample.split,
            "subject_id": sample.subject_id,
            "variant": variant_name,
            "variant_key": variant_key,
        }
    )
    return signal.with_metadata(metadata)


class _XRF55WiFiDataset(Dataset[tuple[RadioTensor, int]]):
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        variant: str,
        labels: list[str] | None = None,
    ) -> None:
        self.dataset_path = resolve_dataset_root("xrf55", override=dataset_path)
        resolved_variant = str(variant).strip()
        if not resolved_variant:
            raise ValueError("XRF55 WiFi dataset requires an explicit canonical variant binding id.")
        variant_payload = _xrf55_variant_payload(resolved_variant)
        signal_profile = _xrf55_signal_profile()
        self.variant = resolved_variant
        self.variant_name = str(variant_payload.get("variant", resolved_variant))
        self.variant_key = str(variant_payload.get("variant_key", resolved_variant))
        self._signal_profile = signal_profile
        self.reader = io_readers.load(
            _require_string(
                signal_profile,
                "reader_id",
                owner="XRF55 schema.yaml signal_profile",
            )
        )

        samples = _load_manifest_samples(self.dataset_path)
        if labels is not None:
            allowed = set(labels)
            samples = [sample for sample in samples if sample.label_name in allowed]
        self.samples = samples
        if not self.samples:
            raise ValueError("No XRF55 raw WiFi samples matched the requested filters")

        unique_labels = sorted({sample.label_name for sample in self.samples})
        self.label_mapping = {label_name: idx for idx, label_name in enumerate(unique_labels)}
        self._dataset_metadata = DatasetMetadata(
            name=_xrf55_card().display_name,
            sample_count=len(self.samples),
            gestures=unique_labels,
            device_type=_require_string(
                signal_profile,
                "device_type",
                owner="XRF55 schema.yaml signal_profile",
            ),
            center_freq=XRF55_WIFI_CENTER_FREQ,
            bandwidth=XRF55_WIFI_BANDWIDTH,
            extra={
                "label_count": len(unique_labels),
                "modality": "wifi",
                "variant": self.variant_name,
                "variant_key": self.variant_key,
            },
        )

    def __len__(self) -> int:
        return len(self.samples)

    def materialize_sample(self, sample: XRF55RawSample) -> tuple[RadioTensor, int]:
        signal = self.reader.read(self.dataset_path / sample.relative_path)
        signal = _augment_xrf55_sample_metadata(
            signal,
            sample,
            variant_name=self.variant_name,
            variant_key=self.variant_key,
        )
        return signal, self.label_mapping[sample.label_name]

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        sample = self.samples[idx]
        return self.materialize_sample(sample)

    def get_labels(self) -> list[int]:
        return [self.label_mapping[sample.label_name] for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        return self.label_mapping.copy()

    def get_sample(self, idx: int) -> XRF55RawSample:
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
                "relative_path": str(sample.relative_path),
                "label_name": str(sample.label_name),
                "label": int(self.label_mapping[sample.label_name]),
                "split": sample.split,
                "subject_id": sample.subject_id,
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
        return get_dataset_card("xrf55")


class XRF55Dataset(Dataset[tuple[RadioTensor, int]]):
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        modality: str | None = None,
        variant: str,
        labels: list[str] | None = None,
    ) -> None:
        requested_modality = None if modality in {None, ""} else str(modality).strip().lower()
        resolved_variant = str(variant).strip()
        if not resolved_variant:
            raise ValueError("XRF55Dataset requires an explicit canonical variant binding id.")
        variant_payload = _xrf55_variant_payload(resolved_variant)
        resolved_modality = str(variant_payload.get("modality", "")).strip().lower()
        if resolved_modality != "wifi":
            raise ValueError("XRF55Dataset currently supports only modality='wifi'")
        if requested_modality not in {None, "wifi"}:
            raise ValueError("XRF55Dataset currently supports only modality='wifi'")
        self.modality = resolved_modality
        self.variant = resolved_variant
        self.variant_key = str(variant_payload.get("variant_key", resolved_variant))
        self._dataset = _XRF55WiFiDataset(
            dataset_path,
            variant=resolved_variant,
            labels=labels,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        return self._dataset[idx]

    def __getattr__(self, name: str) -> object:
        return getattr(self._dataset, name)


def open_xrf55_dataset(
    dataset_path: str | Path | None = None,
    *,
    modality: str | None = None,
    variant: str,
    labels: list[str] | None = None,
) -> XRF55Dataset:
    return XRF55Dataset(
        dataset_path,
        modality=modality,
        variant=variant,
        labels=labels,
    )


__all__ = [
    "XRF55Dataset",
    "XRF55_WIFI_BANDWIDTH",
    "XRF55_WIFI_CENTER_FREQ",
    "open_xrf55_dataset",
]
