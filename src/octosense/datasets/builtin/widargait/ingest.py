"""Ingest owner for the WidarGait builtin dataset."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from torch.utils.data import Dataset

from octosense.datasets.base import resolve_declared_variant
from octosense.datasets.catalog import DatasetCard, get_dataset_card, get_dataset_schema_fields
from octosense.datasets.core.schema import DatasetMetadata
from octosense.io import readers as io_readers
from octosense.io.tensor import RadioTensor


@lru_cache(maxsize=1)
def _widargait_card() -> DatasetCard:
    return get_dataset_card("widargait")


def _widargait_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields("widargait"))


def _widargait_rf_profile() -> dict[str, object]:
    profile = _widargait_schema_fields().get("rf_profile")
    if not isinstance(profile, dict):
        raise ValueError("WidarGait schema.yaml must define mapping field 'rf_profile'")
    return dict(profile)


def _widargait_signal_profile() -> dict[str, object]:
    payload = _widargait_schema_fields().get("signal_profile")
    if not isinstance(payload, dict):
        raise ValueError("WidarGait schema.yaml must define mapping field 'signal_profile'")
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


def _widargait_resolved_variant(variant: str | None) -> str:
    return resolve_declared_variant(
        variant,
        supported_variants=list(
            _widargait_card().supported_variants or tuple(_widargait_card().variants)
        ),
        owner="WidarGait",
    )


def _widargait_variant_payload(variant: str) -> dict[str, object]:
    payload = _widargait_card().variants.get(variant)
    if isinstance(payload, dict):
        return payload
    lowered = variant.lower()
    for key, value in _widargait_card().variants.items():
        if key.lower() == lowered and isinstance(value, dict):
            return value
    available = ", ".join(sorted(_widargait_card().variants))
    raise ValueError(
        f"Unsupported WidarGait variant '{variant}'. Available variants: {available}"
    )


WIDARGAIT_CENTER_FREQ = _require_float(
    _widargait_rf_profile(),
    "center_freq_hz",
    owner="WidarGait schema.yaml rf_profile",
)
WIDARGAIT_BANDWIDTH = _require_float(
    _widargait_rf_profile(),
    "bandwidth_hz",
    owner="WidarGait schema.yaml rf_profile",
)
WIDARGAIT_NOMINAL_SAMPLE_RATE = _require_float(
    _widargait_rf_profile(),
    "nominal_sample_rate_hz",
    owner="WidarGait schema.yaml rf_profile",
)


def _widargait_runtime_profile(variant: str) -> dict[str, object]:
    payload = _widargait_variant_payload(variant).get("runtime_profile", {})
    if not isinstance(payload, dict):
        return {}
    return payload


def _widargait_reader_payload(variant: str) -> dict[str, object]:
    payload = _widargait_variant_payload(variant).get("reader", {})
    if not isinstance(payload, dict):
        return {}
    return payload


def _widargait_paths_payload(variant: str) -> dict[str, object]:
    payload = _widargait_variant_payload(variant).get("paths", {})
    if not isinstance(payload, dict):
        return {}
    return payload


def _widargait_target_length(variant: str) -> int:
    payload = _widargait_runtime_profile(variant)
    value = payload.get("target_length")
    if value in {None, ""}:
        raise ValueError(f"WidarGait variant '{variant}' is missing runtime_profile.target_length")
    return int(value)


def _widargait_channel(variant: str) -> int:
    payload = _widargait_reader_payload(variant)
    value = payload.get("channel")
    if value in {None, ""}:
        raise ValueError(f"WidarGait variant '{variant}' is missing reader.channel")
    return int(value)


def _widargait_required_path(variant: str, key: str) -> str:
    payload = _widargait_paths_payload(variant)
    value = payload.get(key)
    if value in {None, ""}:
        raise ValueError(f"WidarGait variant '{variant}' is missing paths.{key}")
    return str(value)


def _widargait_signal_modality() -> str:
    return _require_string(
        _widargait_signal_profile(),
        "signal_modality",
        owner="WidarGait schema.yaml signal_profile",
    )


def _widargait_device_type() -> str:
    return _require_string(
        _widargait_signal_profile(),
        "device_type",
        owner="WidarGait schema.yaml signal_profile",
    )


def _widargait_capture_device() -> str:
    return _require_string(
        _widargait_signal_profile(),
        "capture_device",
        owner="WidarGait schema.yaml signal_profile",
    )


def _augment_widargait_sample_metadata(
    signal: RadioTensor,
    sample: WidarGaitSample,
    *,
    sample_path: Path,
) -> RadioTensor:
    return signal.with_metadata_updates(
        extra_defaults={
            "dataset": _widargait_card().dataset_id,
            "sample_id": sample.sample_id(),
            "sample_path": str(sample_path),
        }
    )


@dataclass(frozen=True)
class WidarGaitSample:
    relative_path: str
    user_id: int
    track_id: int
    rep_id: int
    rx_id: int
    date: str

    def sample_id(self) -> str:
        return (
            f"{self.date}|user={self.user_id}|track={self.track_id}|rep={self.rep_id}|rx={self.rx_id}"
        )

    def group_id(self) -> str:
        return f"{self.date}|user={self.user_id}|track={self.track_id}|rep={self.rep_id}"


class WidarGaitDataset(Dataset[tuple[RadioTensor, int]]):
    """WidarGait dataset returning canonical ``(time, subc, tx, rx)`` RadioTensors."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        variant: str | None = None,
        users: list[int] | None = None,
        tracks: list[int] | None = None,
        rx_ids: list[int] | None = None,
        target_length: int | None = None,
        preload: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.variant = _widargait_resolved_variant(variant)
        self.users = users
        self.tracks = tracks
        self.rx_ids = rx_ids
        self.target_length = (
            _widargait_target_length(self.variant)
            if target_length is None
            else int(target_length)
        )
        self.preload = preload
        variant_payload = _widargait_variant_payload(self.variant)
        self.reader = io_readers.load("wifi/iwl5300", channel=_widargait_channel(self.variant))
        self.processed_reader = io_readers.load("wifi/widargait_processed")
        self._raw_dir = _widargait_required_path(self.variant, "raw_dir").strip("/")
        self._processed_dir = _widargait_required_path(self.variant, "processed_dir").strip("/")
        self._metadata_csv = _widargait_required_path(self.variant, "metadata_csv")
        self.invalid_samples: list[tuple[WidarGaitSample, str]] = []

        self.samples = self._filter_invalid_samples(self._load_samples())
        if not self.samples:
            raise ValueError("No WidarGait samples matched the requested filters")

        unique_users = sorted({sample.user_id for sample in self.samples})
        self.label_mapping = {f"user{user_id}": idx for idx, user_id in enumerate(unique_users)}
        self._dataset_metadata = DatasetMetadata(
            name="WidarGait",
            sample_count=len(self.samples),
            users=unique_users,
            gestures=[f"user{user_id}" for user_id in unique_users],
            collection_dates=sorted({sample.date for sample in self.samples}),
            device_type=_widargait_device_type(),
            center_freq=WIDARGAIT_CENTER_FREQ,
            bandwidth=WIDARGAIT_BANDWIDTH,
            nominal_sample_rate=WIDARGAIT_NOMINAL_SAMPLE_RATE,
            extra={
                "tracks": sorted({sample.track_id for sample in self.samples}),
                "invalid_sample_count": len(self.invalid_samples),
                "variant": self.variant,
                "variant_description": variant_payload.get("description"),
                "target_length": self.target_length,
                "reader_channel": _widargait_channel(self.variant),
            },
        )
        self._cache: list[RadioTensor | None] | None = None
        if preload:
            self._cache = [self._load_radiotensor(sample) for sample in self.samples]

    def _load_samples(self) -> list[WidarGaitSample]:
        meta_path = self.dataset_path / self._metadata_csv
        if not meta_path.exists():
            raise FileNotFoundError(f"metas.csv not found at {meta_path}")

        samples: list[WidarGaitSample] = []
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = int(row["user_id"])
                track_id = int(row["track_id"])
                rx_id = int(row["rx_id"])
                if self.users is not None and user_id not in self.users:
                    continue
                if self.tracks is not None and track_id not in self.tracks:
                    continue
                if self.rx_ids is not None and rx_id not in self.rx_ids:
                    continue

                file_name = row["file_name"].strip()
                date = (
                    Path(file_name).parts[0]
                    if "/" in file_name
                    else Path(file_name).stem.split("-")[0]
                )
                samples.append(
                    WidarGaitSample(
                        relative_path=str(Path(self._raw_dir) / file_name),
                        user_id=user_id,
                        track_id=track_id,
                        rep_id=int(row["rep_id"]),
                        rx_id=rx_id,
                        date=date,
                    )
                )
        return samples

    def _processed_path(self, sample: WidarGaitSample) -> Path:
        relative = Path(sample.relative_path)
        relative_parts = relative.parts
        if relative_parts and relative_parts[0] == self._raw_dir:
            relative = Path(*relative_parts[1:])
        return self.dataset_path / self._processed_dir / relative.with_suffix(".npy")

    def _raw_path(self, sample: WidarGaitSample) -> Path:
        return self.dataset_path / sample.relative_path

    def _validate_raw_sample(self, sample: WidarGaitSample) -> str | None:
        processed = self._processed_path(sample)
        if processed.exists():
            return None

        raw_path = self._raw_path(sample)
        try:
            size_bytes = raw_path.stat().st_size
        except FileNotFoundError:
            return f"missing raw CSI file: {raw_path}"
        except OSError as exc:
            return f"failed to stat raw CSI file: {exc}"

        if size_bytes < 3:
            return f"raw CSI file too short: {size_bytes} bytes"

        is_valid, error = self.reader.validate_format(raw_path)
        if not is_valid:
            return error or "raw CSI file failed IWL5300 format validation"
        return None

    def _filter_invalid_samples(self, samples: list[WidarGaitSample]) -> list[WidarGaitSample]:
        valid_samples: list[WidarGaitSample] = []
        invalid_samples: list[tuple[WidarGaitSample, str]] = []

        for sample in samples:
            reason = self._validate_raw_sample(sample)
            if reason is None:
                valid_samples.append(sample)
            else:
                invalid_samples.append((sample, reason))

        self.invalid_samples = invalid_samples
        if invalid_samples:
            preview = ", ".join(
                f"{sample.sample_id()} [{reason}]" for sample, reason in invalid_samples[:3]
            )
            warnings.warn(
                "Skipped "
                f"{len(invalid_samples)} invalid WidarGait raw samples before training. "
                f"Examples: {preview}",
                RuntimeWarning,
                stacklevel=2,
            )
        return valid_samples

    def _load_processed_tensor(self, sample: WidarGaitSample) -> RadioTensor | None:
        processed = self._processed_path(sample)
        if not processed.exists():
            return None
        return self.processed_reader.read(
            processed,
            sample_id=sample.sample_id(),
            dataset_name=_widargait_card().dataset_id,
            signal_modality=_widargait_signal_modality(),
            capture_device=_widargait_capture_device(),
            center_freq_hz=WIDARGAIT_CENTER_FREQ,
            bandwidth_hz=WIDARGAIT_BANDWIDTH,
            sample_rate_hz=WIDARGAIT_NOMINAL_SAMPLE_RATE,
        )

    def _load_radiotensor(self, sample: WidarGaitSample) -> RadioTensor:
        rt = self._load_processed_tensor(sample)
        if rt is None:
            full_path = self._raw_path(sample)
            rt = self.reader.read(str(full_path))
            return _augment_widargait_sample_metadata(rt, sample, sample_path=full_path)
        return rt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        rt = (
            self._cache[idx]
            if self._cache is not None
            else self._load_radiotensor(self.samples[idx])
        )
        if rt is None:
            raise RuntimeError(f"Failed to load WidarGait sample {self.samples[idx].sample_id()}")
        label = self.label_mapping[f"user{self.samples[idx].user_id}"]
        return rt, label

    def get_labels(self) -> list[int]:
        return [self.label_mapping[f"user{sample.user_id}"] for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        return self.label_mapping.copy()

    def get_sample(self, idx: int) -> WidarGaitSample:
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
                "user_id": int(sample.user_id),
                "track_id": int(sample.track_id),
                "rep_id": int(sample.rep_id),
                "rx_id": int(sample.rx_id),
                "date": str(sample.date),
                "label": int(self.label_mapping[f"user{sample.user_id}"]),
            }
            for index, sample in enumerate(self.samples)
        ]

    def sample_describe_tree(self):
        sample = self._cache[0] if self._cache is not None else self._load_radiotensor(self.samples[0])
        return sample.describe_tree().with_name("sample")

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        return self._dataset_metadata

    @property
    def dataset_card(self) -> DatasetCard:
        return get_dataset_card("widargait")


def load_widargait_dataset(
    dataset_path: str | Path,
    *,
    variant: str | None = None,
    users: list[int] | None = None,
    tracks: list[int] | None = None,
    rx_ids: list[int] | None = None,
    target_length: int | None = None,
    preload: bool = False,
) -> WidarGaitDataset:
    return WidarGaitDataset(
        dataset_path,
        variant=variant,
        users=users,
        tracks=tracks,
        rx_ids=rx_ids,
        target_length=target_length,
        preload=preload,
    )


__all__ = [
    "WIDARGAIT_BANDWIDTH",
    "WIDARGAIT_CENTER_FREQ",
    "WIDARGAIT_NOMINAL_SAMPLE_RATE",
    "WidarGaitDataset",
    "WidarGaitSample",
    "load_widargait_dataset",
]
