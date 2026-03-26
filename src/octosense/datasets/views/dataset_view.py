"""Metadata-first ``DatasetView`` owner for ``octosense.datasets``."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from octosense.core import DescribeNode, Describable, ensure_describe_node
from octosense.datasets.core.manifest import DatasetManifest
from octosense.datasets.views.filter import compile_metadata_filter
from octosense.datasets.views.group_split import GroupSplitPlan, resolve_group_split
from octosense.datasets.views.leakage import normalize_leakage_fields, summarize_metadata_overlap
from octosense.datasets.views.partition import partition_positions
from octosense.io.tensor import RadioTensor

_SUPPORTED_SPLIT_NAMES = {"all", "train", "val", "test", "train+val"}
_LABEL_TARGET_KINDS = frozenset({"categorical_label"})
_STRUCTURED_TARGET_KINDS = frozenset({"structured_pose", "coordinates", "respiration_signal"})


def _ensure_named_node(
    value: DescribeNode | dict[str, object],
    *,
    name: str,
    kind: str | None = None,
) -> DescribeNode:
    node = ensure_describe_node(value)
    if kind is not None and node.kind != kind:
        node = node.with_kind(kind)
    if node.name != name:
        node = node.with_name(name)
    return node


def _sample_describe_tree_from_sample(sample: RadioTensor) -> DescribeNode:
    return sample.describe_tree().with_name("sample")


def _deferred_sample_describe_node() -> DescribeNode:
    return DescribeNode(
        kind="sample",
        name="sample",
        fields={
            "status": "deferred",
            "materialization": "call sample_describe_tree() to resolve sample IO",
        },
    )


def _normalize_split_name(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    if normalized not in _SUPPORTED_SPLIT_NAMES:
        supported = ", ".join(sorted(_SUPPORTED_SPLIT_NAMES))
        raise ValueError(
            f"Unsupported split '{split_name}'. Expected one of: {supported}"
        )
    return normalized


def _dataset_metadata_node(dataset: Any) -> DescribeNode | None:
    metadata = getattr(dataset, "dataset_metadata", None)
    if metadata is None:
        return None
    describe_tree = getattr(metadata, "describe_tree", None)
    if callable(describe_tree):
        return _ensure_named_node(describe_tree(), name="dataset_metadata", kind="dataset_metadata")
    to_dict = getattr(metadata, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return DescribeNode(kind="dataset_metadata", name="dataset_metadata", fields=payload)
    return None


def _target_kind_uses_label_mapping(target_kind: str | None) -> bool:
    return str(target_kind or "").strip() in _LABEL_TARGET_KINDS


def _target_kind_uses_structured_schema(target_kind: str | None) -> bool:
    return str(target_kind or "").strip() in _STRUCTURED_TARGET_KINDS


def _infer_target_schema(target: dict[str, torch.Tensor] | Any) -> dict[str, object]:
    if not isinstance(target, dict):
        raise TypeError(f"Expected structured target dict, got {type(target)}")
    return {key: list(torch.as_tensor(value).shape) for key, value in target.items()}


def _canonical_task_semantics(task_id: str | None) -> tuple[str | None, str | None]:
    if task_id in {None, ""}:
        return None, None
    from octosense.datasets.core.task_binding import canonical_task_semantics

    task_kind, target_kind = canonical_task_semantics(str(task_id))
    return task_kind, target_kind


def _normalize_optional_semantic(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_modalities_payload(
    value: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    raw_items: Sequence[str]
    if isinstance(value, str):
        raw_items = (value,)
    else:
        raw_items = value
    normalized = tuple(str(item).strip() for item in raw_items if str(item).strip())
    return normalized or None


def _validate_explicit_task_semantics(
    *,
    task_id: str | None,
    task_kind: str | None,
    target_kind: str | None,
) -> None:
    if task_id is not None:
        return
    explicit_fields = [
        field_name
        for field_name, field_value in (
            ("task_kind", task_kind),
            ("target_kind", target_kind),
        )
        if field_value is not None
    ]
    if not explicit_fields:
        return
    field_text = ", ".join(explicit_fields)
    raise ValueError(
        "DatasetView requires canonical task_id when explicit "
        f"{field_text} is provided."
    )


def _resolve_task_semantics(
    *,
    task_id: str | None,
    task_kind: str | None,
    target_kind: str | None,
) -> tuple[str | None, str | None]:
    canonical_task_kind, canonical_target_kind = _canonical_task_semantics(task_id)
    return canonical_task_kind or task_kind, canonical_target_kind or target_kind


def _schema_declared_field_names(schema: object) -> tuple[str, ...]:
    declared: list[str] = []
    for collection_name in ("coordinates", "columns"):
        collection = getattr(schema, collection_name, None)
        if not isinstance(collection, Sequence):
            continue
        for item in collection:
            field_name = getattr(item, "name", None)
            normalized = _normalize_optional_semantic(field_name)
            if normalized is not None and normalized not in declared:
                declared.append(normalized)
    return tuple(declared)


def _declared_metadata_field_names(
    owner: object,
    *,
    dataset_metadata: Any | None = None,
    manifest: DatasetManifest | None = None,
) -> tuple[str, ...]:
    declared: list[str] = []

    manifest_schema = getattr(manifest, "schema", None)
    if manifest_schema is not None:
        declared.extend(_schema_declared_field_names(manifest_schema))

    owner_manifest = getattr(owner, "manifest", None)
    owner_manifest_schema = getattr(owner_manifest, "schema", None)
    if owner_manifest_schema is not None:
        declared.extend(_schema_declared_field_names(owner_manifest_schema))

    for provider_name in ("manifest_schema", "get_manifest_schema"):
        provider = getattr(owner, provider_name, None)
        payload = provider() if callable(provider) else provider
        if payload is not None:
            declared.extend(_schema_declared_field_names(payload))

    metadata = dataset_metadata if dataset_metadata is not None else getattr(owner, "dataset_metadata", None)
    metadata_extra = getattr(metadata, "extra", None)
    if isinstance(metadata_extra, Mapping):
        metadata_fields = metadata_extra.get("metadata_fields")
        if isinstance(metadata_fields, Sequence) and not isinstance(metadata_fields, (str, bytes)):
            for field_name in metadata_fields:
                normalized = _normalize_optional_semantic(str(field_name))
                if normalized is not None:
                    declared.append(normalized)

    normalized_declared: list[str] = []
    for field_name in declared:
        normalized = _normalize_optional_semantic(field_name)
        if normalized is not None and normalized not in normalized_declared:
            normalized_declared.append(normalized)
    return tuple(normalized_declared)


def _sanitize_dataset_target_schema(
    target_schema: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if target_schema is None:
        return None
    payload = {str(key): copy.deepcopy(value) for key, value in target_schema.items()}
    slots_payload = payload.get("slots")
    if isinstance(slots_payload, Mapping):
        concrete_layout: dict[str, object] = {}
        for slot_payload in slots_payload.values():
            if not isinstance(slot_payload, Mapping):
                continue
            concrete_field = str(slot_payload.get("concrete_field", "") or "").strip()
            if not concrete_field:
                continue
            if "shape" in slot_payload:
                concrete_layout[concrete_field] = copy.deepcopy(slot_payload["shape"])
                continue
            slot_metadata = {
                str(key): copy.deepcopy(value)
                for key, value in slot_payload.items()
                if str(key) != "concrete_field"
            }
            concrete_layout[concrete_field] = slot_metadata
        if concrete_layout:
            return concrete_layout
    payload.pop("task_id", None)
    payload.pop("task_kind", None)
    payload.pop("target_kind", None)
    return payload


def _extract_leakage_keys(
    target_schema: Mapping[str, object] | None,
) -> tuple[str, ...] | None:
    if target_schema is None:
        return None
    raw_keys = target_schema.get("leakage_keys")
    if raw_keys is None:
        return None
    if isinstance(raw_keys, str) and raw_keys == "":
        return None
    if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, (str, bytes)):
        raise TypeError("DatasetView target_schema.leakage_keys must be a sequence of field names.")
    return normalize_leakage_fields(
        raw_keys,
        owner="DatasetView target_schema.leakage_keys",
    )


def _extract_target_field_bridge(
    target_schema: Mapping[str, object] | None,
) -> dict[str, str] | None:
    if target_schema is None:
        return None
    slots_payload = target_schema.get("slots")
    if not isinstance(slots_payload, Mapping):
        return None
    bridge: dict[str, str] = {}
    for semantic_field, slot_payload in slots_payload.items():
        if not isinstance(slot_payload, Mapping):
            raise TypeError("DatasetView target_schema.slots entries must be mappings.")
        canonical_field = str(semantic_field).strip()
        if not canonical_field:
            raise ValueError("DatasetView target_schema.slots keys must be non-empty field names.")
        concrete_field = str(slot_payload.get("concrete_field", "") or "").strip()
        if not concrete_field:
            raise ValueError(
                "DatasetView target_schema.slots entries must declare non-empty concrete_field values."
            )
        bridge[canonical_field] = concrete_field
    return bridge or None


def _resolve_dataset_binding_target_field_bridge(
    *,
    dataset_id: str | None,
    task_id: str | None,
    dataset_metadata: Any | None,
    metadata_rows: Sequence[dict[str, object]] | None,
    manifest_rows: Sequence[object] | None,
) -> dict[str, str] | None:
    resolved_dataset_id = str(dataset_id or "").strip()
    resolved_task_id = str(task_id or "").strip()
    if not resolved_dataset_id or not resolved_task_id:
        return None

    binding_candidates: list[str] = []

    def _remember_task_binding(value: object) -> None:
        candidate = str(value or "").strip()
        if candidate and candidate not in binding_candidates:
            binding_candidates.append(candidate)

    for rows in (manifest_rows, metadata_rows):
        if rows is None:
            continue
        for row in rows:
            if isinstance(row, Mapping):
                _remember_task_binding(row.get("task_binding"))

    metadata_extra = getattr(dataset_metadata, "extra", None)
    if isinstance(metadata_extra, Mapping):
        _remember_task_binding(metadata_extra.get("task_binding"))

    if not binding_candidates:
        return None
    if len(binding_candidates) > 1:
        raise ValueError(
            "DatasetView task_binding metadata is ambiguous; expected exactly one non-empty "
            f"binding id, got {binding_candidates}."
        )

    from octosense.datasets.core.task_binding import resolve_target_field_bridge_from_task_binding

    return resolve_target_field_bridge_from_task_binding(
        resolved_dataset_id,
        binding_candidates[0],
        owner=(
            "DatasetView execution target bridge "
            f"(dataset_id={resolved_dataset_id!r}, task_id={resolved_task_id!r})"
        ),
    )


def _resolve_sample_describe_tree(
    dataset: Any,
    sample_describe_tree: DescribeNode | dict[str, object] | None,
) -> DescribeNode:
    if sample_describe_tree is not None:
        return _ensure_named_node(sample_describe_tree, name="sample")

    for provider_name in ("sample_describe_tree", "get_sample_describe_tree"):
        provider = getattr(dataset, provider_name, None)
        if callable(provider):
            payload = provider()
            if isinstance(payload, (dict, DescribeNode)):
                return _ensure_named_node(payload, name="sample")

    sample, _ = dataset[0]
    if not isinstance(sample, RadioTensor):
        raise TypeError(
            "Dataset describe_tree expects the dataset to yield RadioTensor samples, "
            f"got {type(sample)}"
        )
    return _sample_describe_tree_from_sample(sample)


def _target_describe_node(
    *,
    target_kind: str | None,
    metadata_rows: Sequence[dict[str, object]] | None,
    label_mapping: dict[str, int] | None,
    target_schema: dict[str, object] | None,
) -> DescribeNode | None:
    if _target_kind_uses_label_mapping(target_kind):
        fields: dict[str, object] = {
            "kind": str(target_kind),
            "attached_axis": "sample",
        }
        if label_mapping is not None:
            fields["label_mapping"] = dict(label_mapping)
        if metadata_rows:
            fields["metadata_fields"] = sorted(metadata_rows[0].keys())
        return DescribeNode(kind="target", name="target", fields=fields)

    if _target_kind_uses_structured_schema(target_kind):
        schema_payload = dict(target_schema or {})
        return DescribeNode(
            kind="target",
            name="target",
            fields={
                "kind": str(target_kind),
                "attached_axis": "sample",
                "schema": schema_payload,
            },
        )

    return None


def build_dataset_describe_tree(
    *,
    dataset_id: str,
    variant: str | None,
    split: str,
    dataset: Any,
    task_id: str | None,
    task_kind: str | None = None,
    target_kind: str | None = None,
    label_mapping: dict[str, int] | None,
    target_schema: dict[str, object] | None,
    metadata_fields: Sequence[str] | None = None,
    metadata_rows: Sequence[dict[str, object]] | None = None,
    sample_describe_tree: DescribeNode | dict[str, object] | None = None,
) -> DescribeNode:
    normalized_task_kind = _normalize_optional_semantic(task_kind)
    normalized_target_kind = _normalize_optional_semantic(target_kind)
    _validate_explicit_task_semantics(
        task_id=task_id,
        task_kind=normalized_task_kind,
        target_kind=normalized_target_kind,
    )
    task_kind, target_kind = _resolve_task_semantics(
        task_id=task_id,
        task_kind=normalized_task_kind,
        target_kind=normalized_target_kind,
    )
    fields: dict[str, object] = {
        "dataset_id": dataset_id,
        "variant": variant,
        "split": split,
        "task_id": task_id,
    }
    if task_kind is not None:
        fields["task_kind"] = task_kind
    if target_kind is not None:
        fields["target_kind"] = target_kind
    children: list[DescribeNode] = []
    dataset_metadata_node = _dataset_metadata_node(dataset)
    if dataset_metadata_node is not None:
        children.append(dataset_metadata_node)
    children.append(_resolve_sample_describe_tree(dataset, sample_describe_tree))
    if metadata_fields:
        children.append(
            DescribeNode(
                kind="metadata_fields",
                name="metadata_fields",
                fields={"fields": list(metadata_fields)},
            )
        )
    if _target_kind_uses_structured_schema(target_kind) and target_schema is None:
        target_schema_provider = getattr(dataset, "get_target_schema", None)
        if callable(target_schema_provider):
            target_schema = dict(target_schema_provider())
    target_node = _target_describe_node(
        target_kind=target_kind,
        metadata_rows=metadata_rows,
        label_mapping=label_mapping,
        target_schema=target_schema,
    )
    if target_node is not None:
        children.append(target_node)
    if target_schema is not None:
        children.append(
            DescribeNode(
                kind="target_schema",
                name="target_schema",
                fields={"fields": dict(target_schema)},
            )
        )
    return DescribeNode(
        kind="dataset_split",
        name=f"{dataset_id}[{split}]",
        fields=fields,
        children=tuple(children),
    )


def copy_dataset_metadata(
    dataset: Dataset[Any],
    *,
    sample_count: int | None = None,
    users: Sequence[int] | object = None,
) -> Any:
    """Deep-copy ``dataset_metadata`` while optionally overriding common fields."""

    metadata = getattr(dataset, "dataset_metadata", None)
    if metadata is None:
        return None
    copied = copy.deepcopy(metadata)
    if sample_count is not None and hasattr(copied, "sample_count"):
        copied.sample_count = int(sample_count)
    if users is not None and hasattr(copied, "users"):
        copied.users = list(users)
    return copied


def _sorted_unique(values: Sequence[object]) -> list[object]:
    unique_values = {value for value in values if value is not None}
    if not unique_values:
        return []
    try:
        return sorted(unique_values)
    except TypeError:
        return sorted(unique_values, key=lambda value: (type(value).__name__, str(value)))


def _metadata_unique_values(
    rows: Sequence[dict[str, object]],
    keys: Sequence[str],
) -> list[object]:
    values: list[object] = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None:
                values.append(value)
                break
    return _sorted_unique(values)


def _maybe_int_list(values: Sequence[object]) -> list[int]:
    coerced: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            coerced.append(int(value))
            continue
        if isinstance(value, str) and value.isdigit():
            coerced.append(int(value))
    return coerced


def _normalize_partition_field_names(fields: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for field in fields:
        candidate = str(field).strip()
        if not candidate:
            continue
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized or ["split"])


def _subset_dataset_metadata(
    dataset: Dataset[Any],
    *,
    rows: Sequence[dict[str, object]] | None,
    sample_count: int,
) -> Any:
    metadata = copy_dataset_metadata(dataset, sample_count=sample_count)
    if metadata is None or not rows:
        return metadata

    users = _maybe_int_list(_metadata_unique_values(rows, ("user_id", "subject", "user")))
    rooms = _maybe_int_list(_metadata_unique_values(rows, ("room", "environment")))
    collection_dates = [
        str(value)
        for value in _metadata_unique_values(rows, ("date", "collection_date", "session_date"))
    ]

    if users and hasattr(metadata, "users"):
        metadata.users = list(users)
    if rooms and hasattr(metadata, "rooms"):
        metadata.rooms = list(rooms)
    if collection_dates and hasattr(metadata, "collection_dates"):
        metadata.collection_dates = list(collection_dates)
    metadata_extra = getattr(metadata, "extra", None)
    if isinstance(metadata_extra, dict):
        summary_specs = metadata_extra.get("row_summary_fields")
        if isinstance(summary_specs, dict):
            row_summaries: dict[str, list[object]] = {}
            for summary_name, declared_keys in summary_specs.items():
                if not isinstance(summary_name, str):
                    continue
                if isinstance(declared_keys, str):
                    keys = (declared_keys,)
                elif isinstance(declared_keys, Sequence):
                    keys = tuple(str(key) for key in declared_keys)
                else:
                    continue
                values = _metadata_unique_values(rows, keys)
                if values:
                    row_summaries[summary_name] = list(values)
            if row_summaries:
                metadata_extra["row_summaries"] = row_summaries
    return metadata


class DatasetView(Dataset[tuple[RadioTensor, Any]], Describable):
    """Metadata-first dataset view over a concrete sample source."""

    def __init__(
        self,
        dataset: Dataset[tuple[RadioTensor, Any]],
        indices: Sequence[int],
        *,
        dataset_id: str | None = None,
        variant: str | None = None,
        split: str | None = None,
        task_id: str | None = None,
        task_kind: str | None = None,
        target_kind: str | None = None,
        label_mapping: dict[str, int] | None = None,
        target_schema: dict[str, object] | None = None,
        target_field_bridge: dict[str, str] | None = None,
        dataset_metadata: Any | None = None,
        metadata_rows: Sequence[dict[str, object]] | None = None,
        manifest_rows: Sequence[object] | None = None,
        sample_describe_tree: (
            DescribeNode
            | dict[str, object]
            | Callable[[], DescribeNode | dict[str, object] | None]
            | None
        ) = None,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.dataset_id = dataset_id
        self.variant = variant
        self.split = split
        if task_id is None:
            self.task_id = None
        else:
            normalized_task_id = str(task_id).strip()
            self.task_id = normalized_task_id or None
        normalized_task_kind = _normalize_optional_semantic(task_kind)
        normalized_target_kind = _normalize_optional_semantic(target_kind)
        _validate_explicit_task_semantics(
            task_id=self.task_id,
            task_kind=normalized_task_kind,
            target_kind=normalized_target_kind,
        )
        self._task_kind = normalized_task_kind
        self._target_kind = normalized_target_kind
        self._label_mapping = dict(label_mapping) if label_mapping is not None else None
        self._target_field_bridge = (
            dict(target_field_bridge)
            if target_field_bridge is not None
            else _extract_target_field_bridge(target_schema)
        )
        self._leakage_keys = _extract_leakage_keys(target_schema)
        self._target_schema = _sanitize_dataset_target_schema(target_schema)
        self._dataset_metadata = dataset_metadata
        self._metadata_rows = [dict(row) for row in metadata_rows] if metadata_rows is not None else None
        self._manifest_rows = (
            [copy.deepcopy(row) for row in manifest_rows] if manifest_rows is not None else None
        )
        self._manifest = self._build_manifest()
        self._sample_describe_tree_provider: (
            Callable[[], DescribeNode | dict[str, object] | None] | None
        ) = None
        if callable(sample_describe_tree):
            self._sample_describe_tree = None
            self._sample_describe_tree_provider = sample_describe_tree
        else:
            self._sample_describe_tree = (
                _ensure_named_node(sample_describe_tree, name="sample")
                if sample_describe_tree is not None
                else None
            )
        self._describe_tree_cache: DescribeNode | None = None

    @property
    def task_kind(self) -> str | None:
        canonical_task_kind, _ = _canonical_task_semantics(self.task_id)
        return canonical_task_kind or self._task_kind

    @property
    def target_kind(self) -> str | None:
        _, canonical_target_kind = _canonical_task_semantics(self.task_id)
        return canonical_target_kind or self._target_kind

    def _build_manifest(self) -> DatasetManifest | None:
        rows: list[object] | None = None
        if self._manifest_rows is not None:
            rows = list(self._manifest_rows)
        elif self._metadata_rows is not None:
            rows = list(self._metadata_rows)
        if rows is None:
            return None
        try:
            return DatasetManifest.from_rows(
                rows,
                dataset_id=self.dataset_id,
                variant=self.variant,
            )
        except (TypeError, ValueError):
            return None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[RadioTensor, Any]:
        return self.dataset[self.indices[idx]]

    @property
    def dataset_metadata(self) -> Any:
        if self._dataset_metadata is not None:
            return self._dataset_metadata
        return getattr(self.dataset, "dataset_metadata", None)

    @property
    def modalities(self) -> tuple[str, ...] | None:
        metadata_modalities = _normalize_modalities_payload(
            getattr(self.dataset_metadata, "modalities", None)
        )
        if metadata_modalities is not None:
            return metadata_modalities
        return _normalize_modalities_payload(getattr(self.dataset, "modalities", None))

    def get_labels(self) -> list[int]:
        if self.target_kind is not None and not _target_kind_uses_label_mapping(self.target_kind):
            raise AttributeError(
                "DatasetView.get_labels() is only available for categorical-label targets."
            )
        if hasattr(self.dataset, "get_labels"):
            labels = self.dataset.get_labels()  # type: ignore[attr-defined]
            return [int(labels[index]) for index in self.indices]
        return [int(self[idx][1]) for idx in range(len(self))]

    def get_label_mapping(self) -> dict[str, int]:
        if self._label_mapping is not None:
            return dict(self._label_mapping)
        concat_children = getattr(self.dataset, "datasets", None)
        if isinstance(concat_children, Sequence):
            for child in concat_children:
                child_mapping = getattr(child, "get_label_mapping", None)
                if callable(child_mapping):
                    return dict(child_mapping())
        if not hasattr(self.dataset, "get_label_mapping"):
            raise AttributeError("Underlying dataset does not expose get_label_mapping()")
        return dict(self.dataset.get_label_mapping())  # type: ignore[attr-defined]

    def get_target_schema(self) -> dict[str, object]:
        if self._target_schema is not None:
            return dict(self._target_schema)
        if not hasattr(self.dataset, "get_target_schema"):
            raise AttributeError("Underlying dataset does not expose get_target_schema()")
        target_schema = self.dataset.get_target_schema()  # type: ignore[attr-defined]
        if not isinstance(target_schema, Mapping):
            raise TypeError(
                "Underlying dataset get_target_schema() must return a mapping describing "
                "dataset-local target layout."
            )
        sanitized = _sanitize_dataset_target_schema(target_schema)
        return dict(sanitized or {})

    def get_execution_target_bridge(self) -> dict[str, str] | None:
        if self._target_field_bridge is None:
            self._target_field_bridge = _resolve_dataset_binding_target_field_bridge(
                dataset_id=self.dataset_id,
                task_id=self.task_id,
                dataset_metadata=self.dataset_metadata,
                metadata_rows=self._metadata_rows,
                manifest_rows=self._manifest_rows,
            )
        if self._target_field_bridge is None:
            return None
        return dict(self._target_field_bridge)

    def get_leakage_keys(self) -> tuple[str, ...] | None:
        if self._leakage_keys is None:
            return None
        return tuple(self._leakage_keys)

    def metadata_rows(self) -> list[dict[str, object]]:
        if self._metadata_rows is None:
            if self._manifest is None:
                return []
            return [dict(row) for row in self._manifest.metadata_rows()]
        return [dict(row) for row in self._metadata_rows]

    def manifest_rows(self) -> list[object]:
        if self._manifest_rows is None:
            if self._manifest is None:
                return []
            return [copy.deepcopy(row) for row in self._manifest.manifest_rows()]
        return [copy.deepcopy(row) for row in self._manifest_rows]

    def sample_describe_tree(self) -> DescribeNode:
        if self._sample_describe_tree is None:
            payload = (
                self._sample_describe_tree_provider()
                if self._sample_describe_tree_provider is not None
                else None
            )
            self._sample_describe_tree = _resolve_sample_describe_tree(self.dataset, payload)
            self._describe_tree_cache = None
        return self._sample_describe_tree

    def _spawn(
        self,
        positions: Sequence[int],
        *,
        split: str | None = None,
    ) -> "DatasetView":
        selected_positions = [int(position) for position in positions]
        new_indices = [self.indices[position] for position in selected_positions]
        new_rows = None
        if self._metadata_rows is not None:
            new_rows = [dict(self._metadata_rows[position]) for position in selected_positions]
        new_manifest_rows = None
        if self._manifest_rows is not None:
            new_manifest_rows = [
                copy.deepcopy(self._manifest_rows[position]) for position in selected_positions
            ]
        new_metadata = _subset_dataset_metadata(
            self,
            rows=new_rows,
            sample_count=len(new_indices),
        )
        return DatasetView(
            self.dataset,
            new_indices,
            dataset_id=self.dataset_id,
            variant=self.variant,
            split=self.split if split is None else split,
            task_id=self.task_id,
            task_kind=self.task_kind,
            target_kind=self.target_kind,
            label_mapping=self._label_mapping,
            target_schema=self._target_schema,
            target_field_bridge=self._target_field_bridge,
            dataset_metadata=new_metadata,
            metadata_rows=new_rows,
            manifest_rows=new_manifest_rows,
            sample_describe_tree=(
                self._sample_describe_tree
                if self._sample_describe_tree is not None
                else self._sample_describe_tree_provider
            ),
        )

    def select(self, positions: Sequence[int]) -> "DatasetView":
        return self._spawn(positions)

    def limit(self, max_samples: int | None) -> "DatasetView":
        """Return a sample-limited view without touching dataset internals."""

        if max_samples is None:
            return self
        limit = int(max_samples)
        if limit <= 0:
            raise ValueError("DatasetView.limit(...) expects a positive integer.")
        if len(self) <= limit:
            return self
        return self.select(range(limit))

    def get_split(
        self,
        split_name: str,
        *,
        split_field: str | None = None,
        candidate_fields: Sequence[str] = ("split",),
    ) -> "DatasetView":
        normalized_split_name = _normalize_split_name(split_name)
        partition_fields = _normalize_partition_field_names(
            (split_field,) if split_field is not None else candidate_fields
        )
        positions = self._resolve_split_positions(
            normalized_split_name,
            split_field=split_field,
            partition_fields=partition_fields,
        )
        return self._spawn(positions, split=normalized_split_name)

    def _resolve_split_positions(
        self,
        normalized_split_name: str,
        *,
        split_field: str | None,
        partition_fields: Sequence[str],
        max_samples: int | None = None,
    ) -> list[int]:
        limit: int | None = None
        if max_samples is not None:
            limit = int(max_samples)
            if limit <= 0:
                raise ValueError("DatasetView split materialization expects a positive max_samples.")

        if self._manifest is not None:
            positions = self._manifest.split_positions(
                normalized_split_name,
                split_field=split_field,
                candidate_fields=partition_fields,
            )
            if positions:
                return positions if limit is None else positions[:limit]

        rows = self._metadata_rows if self._metadata_rows is not None else self.metadata_rows()
        if not rows:
            raise ValueError(
                "DatasetView.get_split(...) requires metadata_rows or manifest_rows with split assignments."
            )

        positions: list[int] = []
        for index, row in enumerate(rows):
            row_partitions = row.get("partitions")
            if isinstance(row_partitions, dict) and any(
                row_partitions.get(field_name) == normalized_split_name
                for field_name in partition_fields
            ):
                positions.append(index)
            elif "split" in partition_fields and (
                row.get("split") == normalized_split_name
                or row.get("assigned_split") == normalized_split_name
            ):
                positions.append(index)
            if limit is not None and len(positions) >= limit:
                break
        if positions:
            return positions

        available_fields = sorted(
            {
                field_name
                for row in rows
                for field_name in (
                    tuple(
                        str(key)
                        for key in row.get("partitions", {}).keys()
                        if isinstance(row.get("partitions"), dict)
                    )
                    + tuple(
                        key
                        for key in ("split", "assigned_split")
                        if key in row
                    )
                )
            }
        )
        raise ValueError(
            f"Split '{normalized_split_name}' is not available in this DatasetView. "
            f"Available partition fields: {available_fields}"
        )

    def filter(
        self,
        predicate: Callable[[dict[str, object]], bool] | None = None,
        **equals: object,
    ) -> "DatasetView":
        if self._metadata_rows is None:
            raise ValueError(
                "DatasetView.filter(...) requires metadata_rows. "
                "Build the view with machine-readable metadata first."
            )
        positions = compile_metadata_filter(predicate=predicate, **equals).select_positions(
            self._metadata_rows
        )
        return self.select(positions)

    def materialize_split_mapping(
        self,
        *,
        split_names: Sequence[str] = ("train", "val", "test"),
        max_samples_per_split: int | None = None,
        required_splits: Sequence[str] = ("train", "val"),
    ) -> dict[str, "DatasetView"]:
        """Return an explicit split mapping through canonical DatasetView accessors."""

        split_mapping: dict[str, DatasetView] = {}
        normalized_required = {
            _normalize_split_name(split_name)
            for split_name in required_splits
        }
        for split_name in split_names:
            normalized_split_name = _normalize_split_name(split_name)
            try:
                positions = self._resolve_split_positions(
                    normalized_split_name,
                    split_field=None,
                    partition_fields=("split",),
                    max_samples=max_samples_per_split,
                )
            except (AttributeError, KeyError, ValueError):
                continue
            split_mapping[normalized_split_name] = self._spawn(
                positions,
                split=normalized_split_name,
            )
        if not normalized_required.issubset(split_mapping):
            required_text = ", ".join(sorted(normalized_required))
            available_text = ", ".join(sorted(split_mapping)) or "<none>"
            raise ValueError(
                "DatasetView.materialize_split_mapping(...) requires explicit "
                f"{required_text} splits. Available splits: {available_text}."
            )
        return split_mapping

    def select_input(
        self,
        modality: str,
        *,
        node_id: int | None = None,
    ) -> Dataset[tuple[RadioTensor, Any]]:
        if self._manifest_rows is None:
            raise AttributeError("DatasetView.select_input(...) requires structured manifest_rows.")
        projector = getattr(self.dataset, "project_input", None)
        if not callable(projector):
            raise AttributeError(
                "Underlying dataset does not expose project_input(...). "
                "This view is not a session-level multimodal dataset."
            )

        projected_dataset = projector(
            [copy.deepcopy(record) for record in self._manifest_rows],
            modality=modality,
            node_id=node_id,
        )
        if not isinstance(projected_dataset, Dataset):
            raise TypeError(
                "project_input(...) must return a Dataset, "
                f"got {type(projected_dataset)!r}"
            )

        label_mapping = dict(self._label_mapping) if self._label_mapping is not None else None
        get_label_mapping = getattr(projected_dataset, "get_label_mapping", None)
        if callable(get_label_mapping):
            label_mapping = dict(get_label_mapping())

        metadata_rows_provider = getattr(projected_dataset, "metadata_rows", None)
        manifest_rows_provider = getattr(projected_dataset, "manifest_rows", None)
        sample_describe_tree_provider = getattr(projected_dataset, "sample_describe_tree", None)
        target_schema_provider = getattr(projected_dataset, "get_target_schema", None)
        dataset_metadata = getattr(projected_dataset, "dataset_metadata", None)

        projected_metadata_rows = metadata_rows_provider() if callable(metadata_rows_provider) else None
        projected_manifest_rows = manifest_rows_provider() if callable(manifest_rows_provider) else None
        target_schema = (
            dict(target_schema_provider()) if callable(target_schema_provider) else self._target_schema
        )
        sample_describe_tree = (
            sample_describe_tree_provider
            if callable(sample_describe_tree_provider)
            else self._sample_describe_tree
        )
        return DatasetView(
            projected_dataset,
            list(range(len(projected_dataset))),
            dataset_id=self.dataset_id,
            variant=self.variant,
            split=self.split,
            task_id=self.task_id,
            task_kind=self.task_kind,
            target_kind=self.target_kind,
            label_mapping=label_mapping,
            target_schema=target_schema,
            target_field_bridge=self._target_field_bridge,
            dataset_metadata=dataset_metadata,
            metadata_rows=projected_metadata_rows,
            manifest_rows=projected_manifest_rows,
            sample_describe_tree=sample_describe_tree,
        )

    def partition(
        self,
        predicate: Callable[[dict[str, object]], bool] | None = None,
        *,
        matched_split_name: str | None = None,
        remainder_split_name: str | None = None,
        **equals: object,
    ) -> tuple["DatasetView", "DatasetView"]:
        if self._metadata_rows is None:
            raise ValueError(
                "DatasetView.partition(...) requires metadata_rows. "
                "Build the view with machine-readable metadata first."
            )
        matched_positions, remainder_positions = partition_positions(
            self._metadata_rows,
            predicate=predicate,
            **equals,
        )
        return (
            self._spawn(matched_positions, split=matched_split_name),
            self._spawn(remainder_positions, split=remainder_split_name),
        )

    def group_split(
        self,
        group_by: str,
        *,
        ratio: float = 0.8,
        left_values: Sequence[object] | None = None,
        right_values: Sequence[object] | None = None,
        left_split_name: str | None = None,
        right_split_name: str | None = None,
    ) -> tuple["DatasetView", "DatasetView"]:
        if self._metadata_rows is None:
            raise ValueError(
                "DatasetView.group_split(...) requires metadata_rows. "
                "Build the view with machine-readable metadata first."
            )
        if not group_by:
            raise ValueError("group_by must not be empty")
        normalized_left_values = (
            tuple((value,) for value in left_values) if left_values is not None else None
        )
        normalized_right_values = (
            tuple((value,) for value in right_values) if right_values is not None else None
        )
        result = resolve_group_split(
            self._metadata_rows,
            plan=GroupSplitPlan(
                fields=(group_by,),
                ratio=ratio,
                left_values=normalized_left_values,
                right_values=normalized_right_values,
            ),
        )
        return (
            self._spawn(result.left_positions, split=left_split_name),
            self._spawn(result.right_positions, split=right_split_name),
        )

    def metadata_overlap(
        self,
        other: "DatasetView | object",
        *,
        fields: Sequence[str] | None = None,
    ) -> dict[str, list[object]]:
        if self._metadata_rows is None:
            raise ValueError(
                "DatasetView.metadata_overlap(...) requires metadata_rows on the left view."
            )
        other_provider = getattr(other, "metadata_rows", None)
        if not callable(other_provider):
            raise TypeError(
                "metadata_overlap(...) expects another metadata-aware dataset view or "
                "split handle exposing metadata_rows()."
            )
        other_rows = other_provider()
        resolved_fields = tuple(fields) if fields is not None else self.get_leakage_keys()
        if not resolved_fields:
            raise ValueError(
                "DatasetView.metadata_overlap(...) requires explicit fields or task-binding "
                "leakage_keys in target_schema."
            )
        left_declared_fields = _declared_metadata_field_names(
            self.dataset,
            dataset_metadata=self.dataset_metadata,
            manifest=self._manifest,
        )
        other_declared_fields = _declared_metadata_field_names(other)
        if isinstance(other, DatasetView):
            other_declared_fields = _declared_metadata_field_names(
                other.dataset,
                dataset_metadata=other.dataset_metadata,
                manifest=other._manifest,
            )
        return summarize_metadata_overlap(
            self._metadata_rows,
            other_rows,
            fields=resolved_fields,
            left_declared_fields=left_declared_fields,
            right_declared_fields=other_declared_fields,
        )

    def describe_tree(self) -> DescribeNode:
        if self._describe_tree_cache is None:
            if self.dataset_id is None or self.split is None:
                raise ValueError(
                    "DatasetView.describe_tree() requires dataset_id / split context."
                )
            resolved_label_mapping = self._label_mapping
            if resolved_label_mapping is None and _target_kind_uses_label_mapping(self.target_kind):
                resolved_label_mapping = self.get_label_mapping()
            self._describe_tree_cache = build_dataset_describe_tree(
                dataset_id=self.dataset_id,
                variant=self.variant,
                split=self.split,
                dataset=self,
                task_id=self.task_id,
                task_kind=self.task_kind,
                target_kind=self.target_kind,
                label_mapping=resolved_label_mapping,
                target_schema=self._target_schema,
                metadata_fields=sorted(self._metadata_rows[0].keys()) if self._metadata_rows else None,
                metadata_rows=self._metadata_rows,
                sample_describe_tree=(
                    self._sample_describe_tree
                    if self._sample_describe_tree is not None
                    else _deferred_sample_describe_node()
                ),
            ).with_kind("dataset_view")
        return self._describe_tree_cache
