"""Ingest owner for the MM-Fi builtin dataset."""

from __future__ import annotations

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
def _mmfi_card() -> DatasetCard:
    return get_dataset_card("mmfi")


def _mmfi_schema_fields() -> dict[str, object]:
    return dict(get_dataset_schema_fields("mmfi"))


@dataclass(frozen=True)
class MMFiModalityProfile:
    modality: str
    sample_subdir: str
    signal_modality: str
    device_type: str
    reader_id: str
    axis_schema: tuple[str, ...]
    coord_units: dict[str, str]
    time_coord_unit: str
    center_freq_hz: float = 0.0
    bandwidth_hz: float = 0.0
    nominal_sample_rate_hz: float = 0.0


def _normalize_modality(modality: str) -> str:
    normalized = modality.strip().lower().replace("_", "")
    if normalized not in {"wifi", "mmwave"}:
        raise ValueError("MMFiDataset currently supports modality='wifi' or 'mmwave'")
    return normalized


def _mmfi_modality_profile(modality: str) -> MMFiModalityProfile:
    normalized = _normalize_modality(modality)
    profiles = _mmfi_schema_fields().get("rf_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("MMFi schema.yaml must define mapping field 'rf_profiles'")
    payload = profiles.get(normalized)
    if not isinstance(payload, dict):
        raise ValueError(f"MMFi schema.yaml is missing rf_profiles.{normalized!r}")

    axis_schema = payload.get("axis_schema")
    if not isinstance(axis_schema, list) or not all(isinstance(axis, str) for axis in axis_schema):
        raise ValueError(
            f"MMFi schema.yaml rf_profiles.{normalized!r}.axis_schema must be a string list"
        )

    coord_units_raw = payload.get("coord_units")
    if not isinstance(coord_units_raw, dict):
        raise ValueError(
            f"MMFi schema.yaml rf_profiles.{normalized!r}.coord_units must be a mapping"
        )
    coord_units = {str(axis): str(unit) for axis, unit in coord_units_raw.items()}

    return MMFiModalityProfile(
        modality=normalized,
        sample_subdir=str(payload.get("sample_subdir", normalized)),
        signal_modality=str(payload.get("signal_modality", normalized)),
        device_type=str(payload.get("device_type", "")),
        reader_id=str(payload.get("reader_id", "")),
        axis_schema=tuple(axis_schema),
        coord_units=coord_units,
        time_coord_unit=str(payload.get("time_coord_unit", "index")),
        center_freq_hz=float(payload.get("center_freq_hz", 0.0)),
        bandwidth_hz=float(payload.get("bandwidth_hz", 0.0)),
        nominal_sample_rate_hz=float(payload.get("nominal_sample_rate_hz", 0.0)),
    )


def _mmfi_variant_payload(variant_key: str) -> dict[str, object]:
    resolved_variant_key = str(variant_key).strip()
    if not resolved_variant_key:
        raise ValueError("MMFi variant must be a non-empty canonical binding id.")

    payload = _mmfi_card().variants.get(resolved_variant_key)
    if isinstance(payload, dict):
        return payload

    available = ", ".join(sorted(_mmfi_card().variants))
    raise ValueError(
        f"Unsupported MMFi variant '{resolved_variant_key}'. Available variants: {available}"
    )


def _mmfi_dataset_instance_name(modality: str) -> str:
    normalized = _normalize_modality(modality)
    return "MMFiWiFi" if normalized == "wifi" else "MMFiMmwave"


MMFI_WIFI_CENTER_FREQ = _mmfi_modality_profile("wifi").center_freq_hz
MMFI_WIFI_BANDWIDTH = _mmfi_modality_profile("wifi").bandwidth_hz
MMFI_WIFI_SAMPLE_RATE = _mmfi_modality_profile("wifi").nominal_sample_rate_hz
MMFI_MMWAVE_CENTER_FREQ = _mmfi_modality_profile("mmwave").center_freq_hz
MMFI_PROTOCOL_ACTIONS: dict[str, tuple[str, ...]] = {
    str(name): tuple(str(action) for action in actions)
    for name, actions in dict(_mmfi_schema_fields().get("protocol_actions", {})).items()
}


@dataclass(frozen=True)
class MMFiSequenceSample:
    scene: str
    subject: str
    action: str
    modality: str
    relative_path: str

    def sample_id(self) -> str:
        modality_token = Path(self.relative_path).name
        return f"{self.scene}|{self.subject}|{self.action}|modality={modality_token}"

    def group_id(self) -> str:
        return f"{self.scene}|{self.subject}|{self.action}"


def _scan_mmfi_sequences(
    dataset_root: Path,
    *,
    profile: MMFiModalityProfile,
    scenes: list[str] | None,
    subjects: list[str] | None,
    actions: list[str],
) -> list[MMFiSequenceSample]:
    allowed_scenes = set(scenes) if scenes is not None else None
    allowed_subjects = set(subjects) if subjects is not None else None
    allowed_actions = set(actions)
    samples: list[MMFiSequenceSample] = []
    for scene_dir in sorted(dataset_root.glob("E*")):
        if not scene_dir.is_dir():
            continue
        if allowed_scenes is not None and scene_dir.name not in allowed_scenes:
            continue
        for subject_dir in sorted(scene_dir.glob("S*")):
            if not subject_dir.is_dir():
                continue
            if allowed_subjects is not None and subject_dir.name not in allowed_subjects:
                continue
            for action_dir in sorted(subject_dir.glob("A*")):
                if not action_dir.is_dir() or action_dir.name not in allowed_actions:
                    continue
                rel_path = action_dir.relative_to(dataset_root) / profile.sample_subdir
                if not (dataset_root / rel_path).is_dir():
                    continue
                samples.append(
                    MMFiSequenceSample(
                        scene=scene_dir.name,
                        subject=subject_dir.name,
                        action=action_dir.name,
                        modality=profile.modality,
                        relative_path=rel_path.as_posix(),
                    )
                )
    return samples


class MMFiDataset(Dataset[tuple[RadioTensor, int]]):
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        modality: str = "wifi",
        variant: str,
        scenes: list[str] | None = None,
        subjects: list[str] | None = None,
        actions: list[str] | None = None,
        max_points: int | None = None,
    ) -> None:
        self.dataset_path = resolve_dataset_root("mmfi", override=dataset_path)
        self.modality = _normalize_modality(modality)
        self.variant = str(variant).strip()
        if not self.variant:
            raise ValueError("MMFiDataset requires an explicit canonical variant binding id.")
        self.max_points = max_points
        self._profile = _mmfi_modality_profile(self.modality)
        self._variant_payload = _mmfi_variant_payload(self.variant)
        payload_modality = str(self._variant_payload.get("modality", "")).strip().lower()
        if payload_modality != self.modality:
            raise ValueError(
                f"MMFi variant '{self.variant}' declares modality {payload_modality!r}, "
                f"got dataset modality {self.modality!r}."
            )
        self.variant_name = str(self._variant_payload.get("variant", self.variant))

        resolved_actions = actions or [
            str(action) for action in self._variant_payload.get("actions", [])
        ]
        self.samples = _scan_mmfi_sequences(
            self.dataset_path,
            profile=self._profile,
            scenes=scenes,
            subjects=subjects,
            actions=resolved_actions,
        )
        if not self.samples:
            raise ValueError(
                f"No MM-Fi {self.modality} samples matched the requested filters"
            )

        unique_actions = sorted({sample.action for sample in self.samples})
        self.label_mapping = {action: idx for idx, action in enumerate(unique_actions)}
        self._dataset_metadata = DatasetMetadata(
            name=_mmfi_dataset_instance_name(self.modality),
            sample_count=len(self.samples),
            users=sorted({int(sample.subject[1:]) for sample in self.samples}),
            gestures=unique_actions,
            rooms=sorted({int(sample.scene[1:]) for sample in self.samples}),
            device_type=self._profile.device_type,
            center_freq=self._profile.center_freq_hz,
            bandwidth=self._profile.bandwidth_hz,
            nominal_sample_rate=self._profile.nominal_sample_rate_hz,
            extra={
                "modality": self.modality,
                "variant": self.variant,
                "protocol": self.variant_name,
                "variant_key": str(self._variant_payload.get("variant_key", "")),
                "max_points": max_points,
            },
        )
        self._reader = self._build_reader()

    def __len__(self) -> int:
        return len(self.samples)

    def _build_reader(self):
        reader_id = str(self._profile.reader_id).strip()
        if not reader_id:
            raise ValueError(
                f"MMFi schema.yaml rf_profiles.{self.modality!r}.reader_id must be non-empty."
            )
        reader_kwargs: dict[str, object] = {}
        if self.modality == "mmwave" and self.max_points is not None:
            reader_kwargs["max_points"] = int(self.max_points)
        return io_readers.load(reader_id, **reader_kwargs)

    def _materialize_radiotensor(self, sample: MMFiSequenceSample) -> RadioTensor:
        full_root = self.dataset_path / sample.relative_path
        radiotensor = self._reader.read(full_root)
        radiotensor.metadata.extra.setdefault("dataset", "MMFi")
        radiotensor.metadata.extra.setdefault("sample_id", sample.sample_id())
        radiotensor.metadata.extra.setdefault("scene", sample.scene)
        radiotensor.metadata.extra.setdefault("subject", sample.subject)
        radiotensor.metadata.extra.setdefault("action", sample.action)
        radiotensor.metadata.extra.setdefault("protocol", self.variant_name)
        radiotensor.metadata.extra.setdefault("variant", self.variant)
        return radiotensor

    def materialize_sample(self, sample: MMFiSequenceSample) -> tuple[RadioTensor, int]:
        return self._materialize_radiotensor(sample), self.label_mapping[sample.action]

    def __getitem__(self, idx: int) -> tuple[RadioTensor, int]:
        sample = self.samples[idx]
        return self.materialize_sample(sample)

    def get_labels(self) -> list[int]:
        return [self.label_mapping[sample.action] for sample in self.samples]

    def get_label_mapping(self) -> dict[str, int]:
        return self.label_mapping.copy()

    def get_sample(self, idx: int) -> MMFiSequenceSample:
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
                "scene": str(sample.scene),
                "subject": str(sample.subject),
                "action": str(sample.action),
                "modality": self.modality,
                "variant": self.variant,
                "label": int(self.label_mapping[sample.action]),
            }
            for index, sample in enumerate(self.samples)
        ]

    def sample_describe_tree(self):
        sample, _ = self[0]
        return sample.describe_tree().with_name("sample")

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._reader = self._build_reader()

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        return self._dataset_metadata

    @property
    def dataset_card(self) -> DatasetCard:
        return get_dataset_card("mmfi")


__all__ = [
    "MMFI_MMWAVE_CENTER_FREQ",
    "MMFI_PROTOCOL_ACTIONS",
    "MMFI_WIFI_BANDWIDTH",
    "MMFI_WIFI_CENTER_FREQ",
    "MMFiDataset",
]
