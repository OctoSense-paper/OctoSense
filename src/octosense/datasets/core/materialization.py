"""Dataset-owned view/materialization helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from torch.utils.data import ConcatDataset

from octosense.datasets.core.builder import annotate_split_rows

if TYPE_CHECKING:
    from octosense.datasets.core.builder import DatasetBuildArtifact
    from octosense.datasets.views.dataset_view import DatasetView


def _materialized_task_id(
    *,
    task_id: str | None,
    target_schema: dict[str, object] | None,
    views_by_split: dict[str, "DatasetView"],
) -> str | None:
    if task_id not in {None, ""}:
        return str(task_id)
    for view in views_by_split.values():
        embedded_task_id = str(getattr(view, "task_id", "") or "").strip()
        if embedded_task_id:
            return embedded_task_id
    return None


def _attach_task_identity(view: "DatasetView", *, task_id: str | None) -> "DatasetView":
    if task_id in {None, ""}:
        return view
    if getattr(view, "task_id", None) in {None, ""}:
        view.task_id = str(task_id)
    return view


def artifact_split_payload(
    artifact: "DatasetBuildArtifact",
    *,
    split: str,
) -> tuple[list[dict[str, object]], list[object]]:
    """Project one split from the canonical artifact manifest."""

    split_positions = artifact.manifest.split_positions(split, candidate_fields=("split",))
    if not split_positions:
        return [], []
    split_manifest = artifact.manifest.select_positions(split_positions)
    return split_manifest.metadata_rows(), split_manifest.manifest_rows()


def build_manifest_backed_view(
    *,
    dataset: Any,
    indices: list[int],
    dataset_id: str,
    variant: str,
    split: str,
    task_id: str | None = None,
    task_kind: str | None = None,
    target_kind: str | None = None,
    label_mapping: dict[str, int] | None = None,
    target_schema: dict[str, object] | None = None,
    target_field_bridge: dict[str, str] | None = None,
    metadata_rows: list[dict[str, object]] | None = None,
    manifest_rows: list[object] | None = None,
    metadata_kwargs: dict[str, object] | None = None,
) -> "DatasetView":
    from octosense.datasets.views.dataset_view import DatasetView, copy_dataset_metadata

    metadata_overrides = dict(metadata_kwargs or {})
    users = metadata_overrides.pop("users", None)
    dataset_metadata = copy_dataset_metadata(dataset, sample_count=len(indices), users=users)
    if dataset_metadata is not None:
        for key, value in metadata_overrides.items():
            if hasattr(dataset_metadata, key):
                setattr(dataset_metadata, key, copy.deepcopy(value))
                continue
            extra = getattr(dataset_metadata, "extra", None)
            if isinstance(extra, dict):
                extra[key] = copy.deepcopy(value)
    sample_describe_tree = None
    provider = getattr(dataset, "sample_describe_tree", None)
    if callable(provider):
        sample_describe_tree = provider
    return DatasetView(
        dataset,
        indices,
        dataset_id=dataset_id,
        variant=variant,
        split=split,
        task_id=task_id,
        task_kind=task_kind,
        target_kind=target_kind,
        label_mapping=label_mapping,
        target_schema=target_schema,
        target_field_bridge=target_field_bridge,
        dataset_metadata=dataset_metadata,
        metadata_rows=metadata_rows,
        manifest_rows=manifest_rows,
        sample_describe_tree=sample_describe_tree,
    )


def merge_materialized_split_views(
    *,
    dataset_id: str,
    variant: str,
    split: str | None,
    task_id: str | None = None,
    task_kind: str | None = None,
    target_kind: str | None = None,
    label_mapping: dict[str, int] | None = None,
    target_schema: dict[str, object] | None = None,
    target_field_bridge: dict[str, str] | None = None,
    views_by_split: dict[str, "DatasetView"],
) -> "DatasetView":
    from octosense.datasets.views.dataset_view import DatasetView, copy_dataset_metadata

    resolved_task_id = _materialized_task_id(
        task_id=task_id,
        target_schema=target_schema,
        views_by_split=views_by_split,
    )
    if split == "train+val":
        requested_splits = tuple(
            split_name for split_name in ("train", "val") if split_name in views_by_split
        )
    elif split not in {None, "all"}:
        try:
            return _attach_task_identity(views_by_split[split], task_id=resolved_task_id)
        except KeyError as exc:
            supported = ", ".join(sorted(views_by_split))
            raise ValueError(f"Unsupported split '{split}'. Supported splits: {supported}") from exc
    else:
        requested_splits = tuple(
            split_name for split_name in ("train", "val", "test") if split_name in views_by_split
        )

    if not requested_splits:
        requested_splits = tuple(sorted(views_by_split))
    if len(requested_splits) == 1:
        return _attach_task_identity(
            views_by_split[requested_splits[0]],
            task_id=resolved_task_id,
        )

    ordered_views = [views_by_split[split_name] for split_name in requested_splits]
    merged_rows: list[dict[str, object]] = []
    merged_manifests: list[object] = []
    total_samples = 0
    for split_name, view in zip(requested_splits, ordered_views, strict=True):
        total_samples += len(view)
        metadata_rows = view.metadata_rows()
        if metadata_rows:
            merged_rows.extend(annotate_split_rows(metadata_rows, split=split_name))
        manifest_payload = view.manifest_rows()
        if manifest_payload:
            merged_manifests.extend(manifest_payload)

    root_split = split or ("all" if "test" in requested_splits else "train+val")
    return DatasetView(
        ConcatDataset(ordered_views),
        list(range(total_samples)),
        dataset_id=dataset_id,
        variant=variant,
        split=root_split,
        task_id=resolved_task_id,
        task_kind=task_kind,
        target_kind=target_kind,
        label_mapping=label_mapping,
        target_schema=target_schema,
        target_field_bridge=target_field_bridge,
        dataset_metadata=copy_dataset_metadata(ordered_views[0], sample_count=total_samples),
        metadata_rows=merged_rows or None,
        manifest_rows=merged_manifests or None,
        sample_describe_tree=ordered_views[0].sample_describe_tree,
    )


__all__ = [
    "artifact_split_payload",
    "build_manifest_backed_view",
    "merge_materialized_split_views",
]
