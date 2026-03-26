"""Builtin manifest generation ownership for Widar3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from octosense.datasets.catalog import get_dataset_binding_payload, list_dataset_binding_ids
from octosense.datasets.core.builder import annotate_split_rows, stratified_train_val_indices

from .ingest import parse_widar3_capture_filename

DATASET_ID = "widar3"


@dataclass
class Widar3Sample:
    """Single Widar3 CSI sample metadata."""

    file_path: str
    user_id: int
    gesture_id: int
    gesture_name: str
    orientation: int
    trial: int
    rx_id: int
    room: int
    date: str

    def sample_id(self) -> str:
        return (
            f"{self.date}|user={self.user_id}|gesture={self.gesture_id}"
            f"|orientation={self.orientation}|trial={self.trial}|rx={self.rx_id}"
        )

    def group_id(self) -> str:
        return (
            f"{self.date}|user={self.user_id}|gesture={self.gesture_id}"
            f"|orientation={self.orientation}|trial={self.trial}"
        )

    @classmethod
    def from_file_path(
        cls, file_path: str, room: int, gesture_map: dict[int, str]
    ) -> "Widar3Sample":
        path = Path(file_path)
        filename = path.stem
        parts = path.parts
        if len(parts) < 3:
            raise ValueError(f"Invalid file path format: {file_path}")

        date = parts[-3] if not parts[-3].startswith("CSI") else parts[-2]
        try:
            parsed = parse_widar3_capture_filename(filename, gesture_map=gesture_map)
            return cls(
                file_path=file_path,
                user_id=int(parsed["user_id"]),
                gesture_id=int(parsed["gesture_id"]),
                gesture_name=str(parsed["gesture_name"]),
                orientation=int(parsed.get("orientation", parsed.get("orientation_id", 0))),
                trial=int(parsed.get("trial", parsed.get("repeat_id", 0))),
                rx_id=int(parsed["rx_id"]),
                room=room,
                date=date,
            )
        except ValueError as exc:
            raise ValueError(f"Cannot parse file path {file_path}: {exc}") from exc


class Widar3SampleLike(Protocol):
    """High-level sample contract consumed by Widar3 ingest/build surfaces."""

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


@dataclass(frozen=True)
class Widar3ManifestPlan:
    """Manifest-owner assembly result for one explicit Widar3 split scheme."""

    split_scheme_id: str
    split_partitions: dict[str, dict[str, Any]]
    train_ratio: float
    seed: int

    def dataset_filters(self, split: str) -> dict[str, list[int] | None]:
        """Return dataset-consumable split filters for one manifest-owned split."""

        payload = self.split_partitions.get(split)
        if not isinstance(payload, dict):
            raise KeyError(
                f"Widar3 manifest plan '{self.split_scheme_id}' is missing split {split!r}."
            )
        return {
            "users": _coerce_optional_int_list(payload.get("users")),
            "rooms": _coerce_optional_int_list(payload.get("rooms")),
            "rx_ids": _coerce_optional_int_list(payload.get("rx_ids")),
        }

    def materialization_payload(self) -> dict[str, object]:
        """Return the manifest-owned payload persisted into the build artifact."""

        return {
            "split_partitions": {
                split_name: dict(payload)
                for split_name, payload in self.split_partitions.items()
            },
            "train_ratio": self.train_ratio,
            "seed": self.seed,
        }

    def materialize_rows(
        self,
        *,
        train_source_rows: Sequence[dict[str, object]],
        train_labels: Sequence[int],
        test_source_rows: Sequence[dict[str, object]],
        task_binding_id: str,
    ) -> dict[str, list[dict[str, object]]]:
        train_indices, val_indices = stratified_train_val_indices(
            [int(label) for label in train_labels],
            train_ratio=self.train_ratio,
            seed=self.seed,
        )
        return {
            "train": _annotate_widar3_rows(
                _subset_rows(train_source_rows, train_indices),
                split="train",
                split_scheme_id=self.split_scheme_id,
                task_binding_id=task_binding_id,
            ),
            "val": _annotate_widar3_rows(
                _subset_rows(train_source_rows, val_indices),
                split="val",
                split_scheme_id=self.split_scheme_id,
                task_binding_id=task_binding_id,
            ),
            "test": _annotate_widar3_rows(
                test_source_rows,
                split="test",
                split_scheme_id=self.split_scheme_id,
                task_binding_id=task_binding_id,
            ),
        }


def load_widar3_sample(
    *,
    file_path: str,
    room: int,
    gesture_map: dict[int, str],
) -> Widar3SampleLike:
    """Create one manifest-owned sample object from a relative Widar3 file path."""

    return Widar3Sample.from_file_path(
        file_path=file_path,
        room=room,
        gesture_map=gesture_map,
    )


def _resolve_widar3_binding(
    *,
    binding_kind: str,
    binding_id: str | None,
) -> dict[str, Any]:
    candidate = "" if binding_id in {None, ""} else str(binding_id).strip()
    available = list_dataset_binding_ids(DATASET_ID, binding_kind=binding_kind)
    if not candidate:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"Widar3 requires an explicit {binding_kind}; "
            "implicit default/singleton fallback is not supported. "
            f"Supported bindings: {supported}."
        )
    if candidate not in available:
        supported = ", ".join(available) or "<none>"
        raise ValueError(
            f"Widar3 {binding_kind} must be one of: {supported}. Received {candidate!r}."
        )
    return get_dataset_binding_payload(
        DATASET_ID,
        binding_kind=binding_kind,
        binding_id=candidate,
    )


def _resolve_widar3_split_scheme_contract(binding_id: str | None = None) -> dict[str, Any]:
    return _resolve_widar3_binding(binding_kind="split_scheme", binding_id=binding_id)


def _selector_payload(
    split_contract: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    partitions = split_contract.get("partitions", {})
    if not isinstance(partitions, dict):
        raise TypeError("Widar3 split contract must define mapping field 'partitions'.")
    payload = partitions.get(split_name)
    if not isinstance(payload, dict):
        raise KeyError(
            f"Widar3 split contract '{split_contract['binding_id']}' is missing partition {split_name!r}."
    )
    return dict(payload)


def _coerce_optional_int_list(payload: Any) -> list[int] | None:
    if payload is None or payload == "":
        return None
    if not isinstance(payload, list):
        raise TypeError(f"Expected integer list payload, got {type(payload)!r}")
    return [int(value) for value in payload]


def _subset_rows(
    rows: Sequence[dict[str, object]],
    indices: Sequence[int],
) -> list[dict[str, object]]:
    return [dict(rows[index]) for index in indices]


def _annotate_widar3_rows(
    rows: Sequence[dict[str, object]],
    *,
    split: str,
    split_scheme_id: str,
    task_binding_id: str,
) -> list[dict[str, object]]:
    return annotate_split_rows(
        [dict(row) for row in rows],
        split=split,
        split_scheme=split_scheme_id,
        task_binding=task_binding_id,
    )


def build_widar3_manifest_plan(
    *,
    split_scheme_id: str | None,
) -> Widar3ManifestPlan:
    """Resolve one explicit split scheme into a manifest-owned assembly plan."""

    split_contract = _resolve_widar3_split_scheme_contract(split_scheme_id)
    resolved_split_scheme_id = str(split_contract["binding_id"])
    split_partitions = {
        "train": _selector_payload(split_contract, "train"),
        "test": _selector_payload(split_contract, "test"),
    }
    val_materialization = split_contract.get("val_materialization", {})
    if not isinstance(val_materialization, dict):
        raise TypeError("Widar3 split scheme must define mapping field 'val_materialization'.")
    train_ratio = float(val_materialization.get("train_ratio", 0.0))
    seed = int(val_materialization.get("seed", 0))
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("Widar3 split scheme val_materialization.train_ratio must be between 0 and 1.")
    return Widar3ManifestPlan(
        split_scheme_id=resolved_split_scheme_id,
        split_partitions=split_partitions,
        train_ratio=train_ratio,
        seed=seed,
    )


__all__ = [
    "DATASET_ID",
    "Widar3ManifestPlan",
    "build_widar3_manifest_plan",
    "load_widar3_sample",
]
