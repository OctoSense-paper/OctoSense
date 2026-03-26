"""Ingest owner for the SignFi builtin dataset."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset

from octosense.datasets.base import resolve_dataset_root, resolve_declared_variant
from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor


@lru_cache(maxsize=1)
def _signfi_card() -> DatasetCard:
    return get_dataset_card("signfi")


def _signfi_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields("signfi"))


def _signfi_signal_profile() -> dict[str, object]:
    payload = _signfi_schema_fields().get("signal_profile")
    if not isinstance(payload, dict):
        raise ValueError("SignFi schema.yaml must define mapping field 'signal_profile'")
    return dict(payload)


def _signfi_rf_profile() -> dict[str, object]:
    profile = _signfi_schema_fields().get("rf_profile")
    if not isinstance(profile, dict):
        raise ValueError("SignFi schema.yaml must define mapping field 'rf_profile'")
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


def _resolve_signfi_variant(variant: str | None) -> str:
    candidate = resolve_declared_variant(
        variant,
        supported_variants=list(_signfi_card().supported_variants or tuple(_signfi_card().variants)),
        owner="SignFi",
    )
    if candidate in _signfi_card().variants:
        return candidate
    supported = ", ".join(sorted(_signfi_card().variants))
    raise ValueError(f"Unsupported SignFi variant '{variant}'. Supported: {supported}")


def _signfi_variant_payload(variant: str | None) -> dict[str, object]:
    normalized = _resolve_signfi_variant(variant)
    payload = _signfi_card().variants.get(normalized)
    if not isinstance(payload, dict):
        raise ValueError(f"SignFi variant '{normalized}' did not resolve to a mapping payload")
    return payload


def _signfi_label_payload() -> dict[str, object]:
    payload = _signfi_schema_fields().get("label_names")
    if not isinstance(payload, dict):
        raise ValueError("SignFi schema.yaml must define mapping field 'label_names'")
    return dict(payload)


def _signfi_signal_modality() -> str:
    return _require_string(_signfi_signal_profile(), "signal_modality", owner="SignFi schema.yaml signal_profile")


def _signfi_capture_device() -> str:
    return _require_string(_signfi_signal_profile(), "capture_device", owner="SignFi schema.yaml signal_profile")


def _signfi_reader_id_prefix() -> str:
    return _require_string(_signfi_signal_profile(), "reader_id_prefix", owner="SignFi schema.yaml signal_profile")


SIGNFI_CENTER_FREQ = _require_float(_signfi_rf_profile(), "center_freq_hz", owner="SignFi schema.yaml rf_profile")
SIGNFI_BANDWIDTH = _require_float(_signfi_rf_profile(), "bandwidth_hz", owner="SignFi schema.yaml rf_profile")
SIGNFI_NOMINAL_SAMPLE_RATE = _require_float(
    _signfi_rf_profile(),
    "nominal_sample_rate_hz",
    owner="SignFi schema.yaml rf_profile",
)


@dataclass(frozen=True)
class SignFiSample:
    variant: str
    environment: str
    direction: str
    sample_index: int
    label_index: int
    label_name: str

    def sample_id(self) -> str:
        return (
            f"{self.variant}|env={self.environment}|dir={self.direction}"
            f"|sample={self.sample_index:06d}|label={self.label_index}"
        )

    def group_id(self) -> str:
        return f"{self.variant}|label={self.label_index}"


def _normalize_variant(variant: str | None) -> str:
    return _resolve_signfi_variant(variant)


def _label_names(variant: str, class_count: int) -> list[str]:
    payload = _signfi_label_payload()
    raw_names = payload.get(variant)
    if not isinstance(raw_names, (list, tuple)):
        raise ValueError(f"SignFi schema.yaml must define label_names.{variant} as a string list")

    names = [str(name).strip() for name in raw_names]
    if any(not name for name in names):
        raise ValueError(f"SignFi schema.yaml label_names.{variant} cannot contain empty values")
    if len(names) != class_count:
        raise ValueError(
            f"SignFi schema.yaml label_names.{variant} must declare exactly {class_count} names, "
            f"got {len(names)}"
        )
    return names


def _mat_variables(mat_path: Path) -> dict[str, tuple[int, ...]]:
    meta = scio.whosmat(mat_path)
    return {
        str(name): tuple(int(dim) for dim in shape)
        for name, shape, _dtype in meta
    }


def _mat_keys(variables: dict[str, tuple[int, ...]]) -> tuple[list[str], list[str]]:
    data_keys: list[str] = []
    label_keys: list[str] = []
    for name in variables:
        if name.startswith("label"):
            label_keys.append(name)
        else:
            data_keys.append(name)
    return data_keys, label_keys


def _choose_data_key(keys: list[str], preferred_prefix: str) -> str:
    for key in keys:
        if key.lower().startswith(preferred_prefix.lower()):
            return key
    if not keys:
        raise ValueError("SignFi MAT file does not expose any CSI payload keys")
    return keys[0]


def _normalize_labels(raw_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(raw_labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("SignFi labels cannot be empty")
    if np.issubdtype(labels.dtype, np.floating):
        labels = labels.astype(np.int64)
    if int(labels.min()) >= 1:
        labels = labels - 1
    return labels.astype(np.int64, copy=False)


class SignFiDataset(Dataset[tuple[RadioTensor, int]]):
    """SignFi segmented MAT adapter with semantic sample conversion."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        variant: str | None = None,
    ) -> None:
        self.root = resolve_dataset_root("signfi", override=dataset_path)
        self.variant = _normalize_variant(variant)
        cfg = _signfi_variant_payload(self.variant)
        self.environment = str(cfg["environment"])
        self.direction = str(cfg["direction"])
        self._mat_path = self.root / Path(str(cfg["relpath"]))
        if not self._mat_path.exists():
            raise FileNotFoundError(f"SignFi variant '{self.variant}' is missing file: {self._mat_path}")

        variables = _mat_variables(self._mat_path)
        data_keys, label_keys = _mat_keys(variables)
        self._data_key = _choose_data_key(data_keys, str(cfg["preferred_key"]))
        if not label_keys:
            raise ValueError(f"SignFi MAT file '{self._mat_path}' is missing label_* payloads")
        self._label_key = label_keys[0]
        data_shape = variables.get(self._data_key)
        if data_shape is None:
            raise ValueError(f"SignFi MAT file '{self._mat_path}' is missing data key '{self._data_key}'")
        if len(data_shape) != 4:
            raise ValueError(
                f"Expected SignFi CSI tensor with 4 dims (time, subc, rx, sample), got {data_shape}"
            )

        payload = scio.loadmat(
            self._mat_path,
            variable_names=[self._label_key],
        )
        self._labels = _normalize_labels(payload[self._label_key])
        if int(data_shape[-1]) != self._labels.shape[0]:
            raise ValueError(
                f"SignFi sample count mismatch: data has {int(data_shape[-1])} samples but "
                f"labels has {self._labels.shape[0]}"
            )
        self.reader = io_readers.load(_signfi_reader_id_prefix())

        expected_class_count = int(cfg["class_count"])
        self._label_names = _label_names(self.variant, expected_class_count)
        if np.any(self._labels < 0):
            raise ValueError(f"SignFi labels must be non-negative after normalization, got min={int(self._labels.min())}")
        max_label = int(self._labels.max(initial=-1))
        if max_label >= len(self._label_names):
            raise ValueError(
                f"SignFi labels for variant '{self.variant}' reference class index {max_label}, "
                f"but schema declares only {len(self._label_names)} label names"
            )
        self._label_mapping = {name: index for index, name in enumerate(self._label_names)}
        self.samples = [
            SignFiSample(
                variant=self.variant,
                environment=self.environment,
                direction=self.direction,
                sample_index=index,
                label_index=int(label),
                label_name=self._label_names[int(label)],
            )
            for index, label in enumerate(self._labels.tolist())
        ]
        self.dataset_metadata = DatasetMetadata(
            name="SignFi",
            sample_count=len(self.samples),
            gestures=sorted({sample.label_name for sample in self.samples}),
            device_type=_require_string(
                _signfi_signal_profile(),
                "device_type",
                owner="SignFi schema.yaml signal_profile",
            ),
            center_freq=SIGNFI_CENTER_FREQ,
            bandwidth=SIGNFI_BANDWIDTH,
            nominal_sample_rate=SIGNFI_NOMINAL_SAMPLE_RATE,
            extra={
                "variant": self.variant,
                "environment": self.environment,
                "direction": self.direction,
                "mat_path": str(self._mat_path),
                "data_key": self._data_key,
                "label_key": self._label_key,
            },
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        sample = self.samples[idx]
        tensor = self.reader.read(
            self._mat_path,
            data_key=self._data_key,
            sample_index=sample.sample_index,
            sample_id=sample.sample_id(),
            label_name=sample.label_name,
            extra_metadata={
                "dataset": "signfi",
                "variant": self.variant,
                "environment": self.environment,
                "direction": self.direction,
            },
        )
        return (
            tensor.with_metadata_updates(
                modality=_signfi_signal_modality(),
                capture_device=_signfi_capture_device(),
            ),
            int(sample.label_index),
        )

    def get_labels(self) -> list[int]:
        return [int(sample.label_index) for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        return dict(self._label_mapping)

    def get_sample_id(self, idx: int) -> str:
        return self.samples[idx].sample_id()

    def metadata_rows(self) -> list[dict[str, object]]:
        return [
            {
                "sample_index": sample.sample_index,
                "environment": sample.environment,
                "direction": sample.direction,
                "variant": sample.variant,
                "sample_id": sample.sample_id(),
                "label": sample.label_index,
                "label_name": sample.label_name,
            }
            for sample in self.samples
        ]

    def sample_describe_tree(self):
        first_sample, _ = self[0]
        return first_sample.describe_tree().with_name("sample")


def signfi_dataset_card() -> DatasetCard:
    return get_dataset_card("signfi")


def load_signfi_dataset(
    dataset_path: str | Path | None = None,
    *,
    variant: str | None = None,
) -> SignFiDataset:
    return SignFiDataset(dataset_path, variant=variant)


__all__ = [
    "SIGNFI_BANDWIDTH",
    "SIGNFI_CENTER_FREQ",
    "SIGNFI_NOMINAL_SAMPLE_RATE",
    "SignFiDataset",
    "SignFiSample",
    "load_signfi_dataset",
    "signfi_dataset_card",
]
