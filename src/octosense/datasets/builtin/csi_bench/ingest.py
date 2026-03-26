"""Raw ingest helpers for the CSI-Bench builtin dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from octosense.datasets.base import resolve_dataset_root
from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor

DATASET_ID = "csi_bench"


@dataclass(frozen=True)
class _CSIBenchSample:
    task_name: str
    sample_id: str
    relative_path: str
    label_name: str
    metadata: dict[str, object]
    position: tuple[float, ...] | None = None


def _csi_bench_card() -> DatasetCard:
    return get_dataset_card(DATASET_ID)


def csi_bench_dataset_card() -> DatasetCard:
    return _csi_bench_card()


def _csi_bench_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields(DATASET_ID))


def _csi_bench_rf_profile() -> dict[str, object]:
    payload = _csi_bench_schema_fields().get("rf_profile")
    if not isinstance(payload, dict):
        raise ValueError("CSI-Bench schema.yaml must define mapping field 'rf_profile'")
    return dict(payload)


def _csi_bench_rf_profile_value_contract() -> dict[str, object]:
    payload = _csi_bench_schema_fields().get("rf_profile_value_contract", {})
    if not isinstance(payload, dict):
        raise ValueError(
            "CSI-Bench schema.yaml rf_profile_value_contract must be a mapping when present"
        )
    return dict(payload)


def _csi_bench_signal_profile() -> dict[str, object]:
    payload = _csi_bench_schema_fields().get("signal_profile")
    if not isinstance(payload, dict):
        raise ValueError("CSI-Bench schema.yaml must define mapping field 'signal_profile'")
    return dict(payload)


def _require_string(payload: dict[str, object], key: str, *, owner: str) -> str:
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"{owner} must define non-empty field '{key}'")
    return str(value)


def _optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value in {None, ""}:
        return None
    return float(value)


def _explicit_nullable_fields() -> set[str]:
    payload = _csi_bench_rf_profile_value_contract().get("nullable_fields", [])
    if not isinstance(payload, list):
        raise ValueError(
            "CSI-Bench schema.yaml rf_profile_value_contract.nullable_fields must be a list"
        )
    return {str(item) for item in payload if item not in {None, ""}}


def _schema_owned_optional_float(payload: dict[str, object], key: str, *, owner: str) -> float | None:
    value = payload.get(key)
    if value not in {None, ""}:
        return float(value)
    if key not in _explicit_nullable_fields():
        raise ValueError(
            f"{owner} must either define '{key}' or list it in rf_profile_value_contract.nullable_fields"
        )
    return None


def _csi_bench_storage_layout() -> dict[str, object]:
    payload = _csi_bench_schema_fields().get("storage_layout")
    if not isinstance(payload, dict):
        raise ValueError("CSI-Bench schema.yaml must define mapping field 'storage_layout'")
    return dict(payload)


def _csi_bench_variant_payload(variant: str) -> dict[str, object]:
    payload = _csi_bench_card().variants.get(variant)
    if isinstance(payload, dict):
        return payload
    available = ", ".join(sorted(_csi_bench_card().variants))
    raise ValueError(f"Unsupported CSI-Bench variant '{variant}'. Supported: {available}")


def _csi_bench_variant_paths(variant: str) -> dict[str, str]:
    payload = _csi_bench_variant_payload(variant).get("paths", {})
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if value not in {None, ""}
    }


def _csi_bench_signal_modality() -> str:
    return _require_string(
        _csi_bench_signal_profile(),
        "signal_modality",
        owner="CSI-Bench schema.yaml signal_profile",
    )


def _csi_bench_reader_id_prefix() -> str:
    return _require_string(
        _csi_bench_signal_profile(),
        "reader_id_prefix",
        owner="CSI-Bench schema.yaml signal_profile",
    )


def _csi_bench_required_path(variant: str, key: str) -> str:
    paths = _csi_bench_variant_paths(variant)
    value = paths.get(key)
    if value in {None, ""}:
        raise ValueError(f"CSI-Bench variant '{variant}' is missing paths.{key}")
    return str(value)


CSI_BENCH_WIFI_CENTER_FREQ = _schema_owned_optional_float(
    _csi_bench_rf_profile(),
    "center_freq_hz",
    owner="CSI-Bench schema.yaml rf_profile",
)
CSI_BENCH_WIFI_BANDWIDTH = _optional_float(_csi_bench_rf_profile(), "bandwidth_hz")
CSI_BENCH_WIFI_NOMINAL_SAMPLE_RATE = _schema_owned_optional_float(
    _csi_bench_rf_profile(),
    "nominal_sample_rate_hz",
    owner="CSI-Bench schema.yaml rf_profile",
)


def resolve_csi_bench_storage_root(root: Path) -> Path:
    layout = _csi_bench_storage_layout()
    candidates = [root]
    for relpath in layout.get("root_candidates", []):
        candidate = root / str(relpath)
        if candidate not in candidates:
            candidates.append(candidate)
    probe_subdir = _require_string(
        layout,
        "task_probe_subdir",
        owner="CSI-Bench schema.yaml storage_layout",
    )
    for candidate in candidates:
        if candidate.exists() and (candidate / probe_subdir).exists():
            return candidate
    return root


def _label_mapping(mapping_path: Path) -> dict[str, int]:
    if not mapping_path.exists():
        return {}
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        label_to_idx = payload.get("label_to_idx")
        if isinstance(label_to_idx, dict):
            return {str(key): int(value) for key, value in label_to_idx.items()}
        idx_to_label = payload.get("idx_to_label")
        if isinstance(idx_to_label, dict):
            return {str(value): int(key) for key, value in idx_to_label.items()}
        return {str(key): int(value) for key, value in payload.items()}
    raise TypeError(f"Expected dict payload in {mapping_path}, got {type(payload)!r}")


def _coordinate_mapping(mapping_path: Path) -> dict[str, tuple[float, ...]]:
    if not mapping_path.exists():
        raise FileNotFoundError(
            "CSI-Bench localization requires metadata/label_mapping.json with explicit coordinates. "
            f"Missing: {mapping_path}"
        )
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {mapping_path}, got {type(payload)!r}")
    coordinates: dict[str, tuple[float, ...]] = {}
    for label_name, raw_value in payload.items():
        if hasattr(raw_value, "tolist"):
            raw_value = raw_value.tolist()
        if not isinstance(raw_value, (list, tuple)):
            raise TypeError(
                "CSI-Bench localization label_mapping.json must map label names to coordinate lists. "
                f"Field {label_name!r} resolved to {type(raw_value)!r}."
            )
        coordinate_values: list[float] = []
        for index, coordinate in enumerate(raw_value):
            if isinstance(coordinate, bool):
                raise TypeError(
                    "CSI-Bench localization coordinate values must be numeric scalars. "
                    f"Field {label_name!r}[{index}] resolved to bool."
                )
            if isinstance(coordinate, (int, float, np.integer, np.floating)):
                coordinate_values.append(float(coordinate))
                continue
            raise TypeError(
                "CSI-Bench localization coordinate values must be numeric scalars. "
                f"Field {label_name!r}[{index}] resolved to {type(coordinate)!r}."
            )
        if not coordinate_values:
            raise ValueError(
                "CSI-Bench localization label_mapping.json must map every label to a non-empty "
                f"coordinate list. Field {label_name!r} resolved to []."
            )
        coordinates[str(label_name)] = tuple(coordinate_values)
    return coordinates


def _normalize_record_path(value: object) -> str:
    candidate = str(value).strip()
    while candidate.startswith("./"):
        candidate = candidate[2:]
    return candidate


def _canonical_row_identity(value: object) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (bool, np.bool_)) and not bool(value):
        candidate = ""
    else:
        candidate = _normalize_record_path(value)
    if candidate and candidate.lower() != "nan":
        return candidate
    raise ValueError("CSI-Bench metadata rows must define explicit id as record identity.")


def _dataset_metadata_rf_kwargs() -> dict[str, float]:
    """Only project schema-declared RF values that exist onto DatasetMetadata."""

    kwargs: dict[str, float] = {}
    if CSI_BENCH_WIFI_CENTER_FREQ is not None:
        kwargs["center_freq"] = CSI_BENCH_WIFI_CENTER_FREQ
    if CSI_BENCH_WIFI_BANDWIDTH is not None:
        kwargs["bandwidth"] = CSI_BENCH_WIFI_BANDWIDTH
    if CSI_BENCH_WIFI_NOMINAL_SAMPLE_RATE is not None:
        kwargs["nominal_sample_rate"] = CSI_BENCH_WIFI_NOMINAL_SAMPLE_RATE
    return kwargs


class CSIBenchDataset(Dataset[tuple[RadioTensor, Any]]):
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        variant: str,
        task_name: str,
        split_name: str = "train_id",
    ) -> None:
        root = resolve_csi_bench_storage_root(
            resolve_dataset_root(DATASET_ID, override=dataset_path)
        )
        self.root = root
        self.variant = str(variant).strip()
        if not self.variant:
            raise ValueError("CSI-Bench dataset requires an explicit config variant binding id.")
        self._structured_target_kind = "coordinates" if self.variant == "localization" else None
        self.task_name = str(task_name).strip()
        if not self.task_name:
            raise ValueError("CSI-Bench dataset requires an explicit task_name from task binding.")
        _csi_bench_variant_payload(self.variant)
        self.split_name = str(split_name)
        self.task_root = self.root / self.task_name
        metadata_path = self.task_root / _csi_bench_required_path(self.variant, "metadata_csv")
        label_mapping_path = self.task_root / _csi_bench_required_path(self.variant, "label_mapping")
        split_dir = self.task_root / _csi_bench_required_path(self.variant, "splits_dir")
        split_path = split_dir / f"{self.split_name}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"CSI-Bench metadata CSV not found: {metadata_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"CSI-Bench split JSON not found: {split_path}")

        split_ids = {
            _normalize_record_path(split_id)
            for split_id in json.loads(split_path.read_text(encoding="utf-8"))
        }
        label_mapping = (
            {}
            if self._structured_target_kind == "coordinates"
            else _label_mapping(label_mapping_path)
        )
        coordinate_mapping = (
            _coordinate_mapping(label_mapping_path)
            if self._structured_target_kind == "coordinates"
            else {}
        )
        rows = np.genfromtxt(
            metadata_path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        if rows.size == 0:
            raise ValueError(f"CSI-Bench metadata CSV is empty: {metadata_path}")
        if rows.shape == ():
            rows = np.asarray([rows], dtype=rows.dtype)

        samples: list[_CSIBenchSample] = []
        for row in rows:
            relative_path = _normalize_record_path(row["file_path"])
            sample_id = _canonical_row_identity(row["id"])
            if sample_id not in split_ids:
                continue
            label_name = str(row["label"])
            metadata = {
                name: (row[name].item() if hasattr(row[name], "item") else row[name])
                for name in row.dtype.names
            }
            position = coordinate_mapping.get(label_name)
            if self._structured_target_kind == "coordinates" and position is None:
                raise ValueError(
                    "CSI-Bench localization metadata rows must reference labels declared in "
                    f"metadata/label_mapping.json. Missing coordinate mapping for {label_name!r}."
                )
            samples.append(
                _CSIBenchSample(
                    task_name=self.task_name,
                    sample_id=sample_id,
                    relative_path=relative_path,
                    label_name=label_name,
                    metadata=metadata,
                    position=position,
                )
            )
        if not samples:
            raise ValueError(
                f"CSI-Bench split '{self.split_name}' for task '{self.task_name}' resolved no samples"
            )

        if not label_mapping:
            if self._structured_target_kind != "coordinates":
                unique_labels = sorted({sample.label_name for sample in samples})
                label_mapping = {label: index for index, label in enumerate(unique_labels)}

        self.samples = samples
        self._label_mapping = label_mapping
        self._target_schema = {}
        if self._structured_target_kind == "coordinates":
            coordinate_dims = sorted(
                {
                    len(sample.position)
                    for sample in samples
                    if sample.position is not None
                }
            )
            if not coordinate_dims:
                raise ValueError(
                    "CSI-Bench localization split resolved no coordinate targets after parsing "
                    "metadata/label_mapping.json."
                )
            if len(coordinate_dims) != 1:
                raise ValueError(
                    "CSI-Bench localization requires a consistent coordinate dimensionality per split. "
                    f"Observed dimensions: {coordinate_dims}."
                )
            self._target_schema = {"label": [coordinate_dims[0]]}
        self.reader = io_readers.load(_csi_bench_reader_id_prefix())
        users = sorted(
            {
                int(str(sample.metadata["user"]).lstrip("U"))
                for sample in samples
                if str(sample.metadata.get("user", "")).lstrip("U").isdigit()
            }
        )
        self.dataset_metadata = DatasetMetadata(
            name=f"CSI-Bench {self.task_name}",
            sample_count=len(samples),
            users=users,
            gestures=sorted(label_mapping.keys()),
            device_type=_require_string(
                _csi_bench_signal_profile(),
                "device_type",
                owner="CSI-Bench schema.yaml signal_profile",
            ),
            extra={
                "variant": self.variant,
                "task_name": self.task_name,
                "split_name": self.split_name,
                "task_root": str(self.task_root),
                "target_kind": self._structured_target_kind,
                "target_schema": dict(self._target_schema),
            },
            **_dataset_metadata_rf_kwargs(),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, Any]:
        sample = self.samples[idx]
        sample_path = self.task_root / sample.relative_path
        tensor = self.reader.read(
            sample_path,
            capture_device=_require_string(
                sample.metadata,
                "device",
                owner="CSI-Bench metadata row",
            ),
            center_freq_hz=CSI_BENCH_WIFI_CENTER_FREQ,
            bandwidth_hz=CSI_BENCH_WIFI_BANDWIDTH,
            sample_rate_hz=CSI_BENCH_WIFI_NOMINAL_SAMPLE_RATE,
            sample_id=sample.sample_id,
            label_name=sample.label_name,
            extra_metadata={
                "dataset": DATASET_ID,
                "variant": self.variant,
                "task_name": self.task_name,
                "split_name": self.split_name,
                **({"position": list(sample.position)} if sample.position is not None else {}),
                **sample.metadata,
            },
        )
        target: Any
        if self._structured_target_kind == "coordinates":
            if sample.position is None:
                raise ValueError(
                    "CSI-Bench localization sample is missing structured coordinates after "
                    "metadata parsing."
                )
            target = {"label": torch.tensor(sample.position, dtype=torch.float32)}
        else:
            target = self._label_mapping[sample.label_name]
        return tensor.with_metadata_updates(
            modality=_csi_bench_signal_modality(),
        ), target

    def get_labels(self) -> list[int]:
        if self._structured_target_kind == "coordinates":
            raise AttributeError(
                "CSI-Bench localization targets are structured coordinates; scalar labels are unavailable."
            )
        return [self._label_mapping[sample.label_name] for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        if self._structured_target_kind == "coordinates":
            raise AttributeError(
                "CSI-Bench localization targets are structured coordinates; label mappings are unavailable."
            )
        return dict(self._label_mapping)

    def get_target_schema(self) -> dict[str, object]:
        return dict(self._target_schema)

    def get_sample_id(self, idx: int) -> str:
        return self.samples[idx].sample_id

    def metadata_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for sample in self.samples:
            row = dict(sample.metadata)
            row["sample_id"] = sample.sample_id
            row.setdefault("record_id", _canonical_row_identity(row.get("id")))
            row["variant"] = self.variant
            row["label_name"] = sample.label_name
            if self._structured_target_kind == "coordinates":
                if sample.position is None:
                    raise ValueError(
                        "CSI-Bench localization metadata row is missing structured coordinates after "
                        "metadata parsing."
                    )
                row["label"] = list(sample.position)
            else:
                row["label"] = self._label_mapping[sample.label_name]
            rows.append(row)
        return rows

    def sample_describe_tree(self):
        first_sample, _ = self[0]
        return first_sample.describe_tree().with_name("sample")


__all__ = [
    "CSI_BENCH_WIFI_BANDWIDTH",
    "CSIBenchDataset",
    "DATASET_ID",
    "csi_bench_dataset_card",
    "resolve_csi_bench_storage_root",
]
