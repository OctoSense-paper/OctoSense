"""Dataset-owned builder contracts for metadata-first assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

from octosense.datasets.core.manifest import DatasetManifest

_GroupT = TypeVar("_GroupT")

if TYPE_CHECKING:
    from octosense.datasets.views.dataset_view import DatasetView


@dataclass(frozen=True)
class DatasetBuildRequest:
    """Normalized request passed to builtin dataset builders."""

    dataset_id: str
    variant: str | None = None
    split_scheme: str | None = None
    task_binding: str | None = None
    modalities: tuple[str, ...] = ()
    dataset_root: Path | None = None
    source_root: Path | None = None
    cache_root: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetBuildArtifact:
    """Metadata artifact emitted by a dataset builder before materialization."""

    dataset_id: str
    variant: str | None
    manifest: DatasetManifest
    dataset_root: Path | None = None
    materialization: dict[str, Any] = field(default_factory=dict)


DatasetMaterializationPayload = tuple[
    str,
    dict[str, int] | None,
    str,
    str,
    dict[str, object] | None,
    dict[str, "DatasetView"],
]


class DatasetBuilderProtocol(Protocol):
    """Protocol for dataset-specific metadata-first build entrypoints."""

    def build(self, request: DatasetBuildRequest) -> DatasetBuildArtifact:
        ...


class MetadataFirstDatasetBuilder:
    """Small OO adapter for builders that emit collection metadata first."""

    dataset_id: str

    def build_manifest(self, request: DatasetBuildRequest) -> DatasetManifest:
        raise NotImplementedError

    def build(self, request: DatasetBuildRequest) -> DatasetBuildArtifact:
        manifest = self.build_manifest(request)
        return DatasetBuildArtifact(
            dataset_id=request.dataset_id,
            variant=request.variant,
            manifest=manifest,
            dataset_root=request.dataset_root,
        )


def annotate_split_rows(
    rows: list[dict[str, object]],
    *,
    split: str,
    split_scheme: str | None = None,
    task_binding: str | None = None,
) -> list[dict[str, object]]:
    """Attach canonical split markers before manifest assembly."""

    annotated: list[dict[str, object]] = []
    for row in rows:
        partitions = dict(row.get("partitions", {})) if isinstance(row.get("partitions"), dict) else {}
        partitions["split"] = split
        normalized = {
            **dict(row),
            "split": split,
            "assigned_split": split,
            "partitions": partitions,
        }
        if split_scheme not in {None, ""}:
            normalized["split_scheme"] = str(split_scheme)
        if task_binding not in {None, ""}:
            normalized["task_binding"] = str(task_binding)
        annotated.append(normalized)
    return annotated


def stratified_train_val_indices(
    labels: list[int],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic stratified train/val positions."""

    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    grouped_positions: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped_positions.setdefault(int(label), []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    rng = np.random.default_rng(int(seed))
    for positions in grouped_positions.values():
        shuffled = list(positions)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.extend(shuffled)
            continue
        cutoff = int(round(len(shuffled) * float(train_ratio)))
        cutoff = max(1, min(len(shuffled) - 1, cutoff))
        train_indices.extend(shuffled[:cutoff])
        val_indices.extend(shuffled[cutoff:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def partition_sorted_groups(
    groups: list[_GroupT],
    *,
    train_ratio: float,
) -> tuple[list[_GroupT], list[_GroupT]]:
    """Split a sorted group list without shuffling group identity."""

    if not groups:
        return [], []
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    cutoff = int(round(len(groups) * float(train_ratio)))
    cutoff = max(1, min(len(groups) - 1, cutoff))
    return list(groups[:cutoff]), list(groups[cutoff:])


def subset_indices_by_group(
    group_values: list[_GroupT],
    *,
    train_ratio: float,
) -> tuple[list[int], list[int], list[_GroupT], list[_GroupT]]:
    """Partition positions by unique group identity."""

    ordered_groups = sorted({value for value in group_values}, key=lambda value: str(value))
    train_groups, val_groups = partition_sorted_groups(ordered_groups, train_ratio=train_ratio)
    train_group_set = set(train_groups)
    train_indices = [index for index, value in enumerate(group_values) if value in train_group_set]
    val_indices = [index for index, value in enumerate(group_values) if value not in train_group_set]
    return train_indices, val_indices, train_groups, val_groups


def build_builtin_artifact(
    request: DatasetBuildRequest,
    *,
    module_path: str,
) -> DatasetBuildArtifact:
    """Resolve and execute one builtin dataset builder module."""

    module = import_module(module_path)
    builder = getattr(module, "build", None)
    if not callable(builder):
        raise AttributeError(
            f"Builtin dataset builder module {module_path!r} does not export build(request)."
        )
    artifact = builder(request)
    if not isinstance(artifact, DatasetBuildArtifact):
        raise TypeError(
            f"Builtin dataset builder {module_path!r} returned {type(artifact)!r}; "
            "expected DatasetBuildArtifact."
        )
    return artifact


__all__ = [
    "annotate_split_rows",
    "build_builtin_artifact",
    "DatasetBuildArtifact",
    "DatasetBuildRequest",
    "DatasetMaterializationPayload",
    "DatasetBuilderProtocol",
    "MetadataFirstDatasetBuilder",
    "partition_sorted_groups",
    "stratified_train_val_indices",
    "subset_indices_by_group",
]
