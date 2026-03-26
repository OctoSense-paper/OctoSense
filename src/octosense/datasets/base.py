"""Shared dataset adapter/base contracts for the dataset public surface."""

from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass, field, replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from collections.abc import Mapping, Sequence

from octosense.datasets.core.materialization import (
    artifact_split_payload,
    build_manifest_backed_view,
    merge_materialized_split_views,
)

if TYPE_CHECKING:
    from octosense.datasets.core.builder import (
        DatasetBuildArtifact,
        DatasetBuildRequest,
        DatasetMaterializationPayload,
    )
    from octosense.datasets.views.dataset_view import DatasetView


OCTOSENSE_DATASETS_ROOT_ENV = "OCTOSENSE_DATASETS_ROOT"


@dataclass(frozen=True)
class _DeclaredDatasetContract:
    dataset_id: str
    modalities: tuple[str, ...]
    supported_variants: tuple[str, ...]
    split_schemes: tuple[str, ...]
    task_bindings: tuple[str, ...]


def _declared_dataset_contract_from_sidecars(dataset_id: str) -> _DeclaredDatasetContract:
    from octosense.datasets.catalog import get_dataset_card, list_dataset_binding_ids

    card = get_dataset_card(dataset_id)
    return _DeclaredDatasetContract(
        dataset_id=dataset_id,
        modalities=tuple(card.modalities),
        supported_variants=tuple(card.supported_variants),
        split_schemes=list_dataset_binding_ids(dataset_id, binding_kind="split_scheme"),
        task_bindings=list_dataset_binding_ids(dataset_id, binding_kind="task_binding"),
    )


def resolve_declared_variant(
    variant: str | None,
    *,
    supported_variants: tuple[str, ...] | list[str],
    owner: str,
) -> str:
    """Resolve one declared variant from an explicit request only, never by fallback."""

    available = tuple(str(item) for item in supported_variants if str(item))
    candidate = "" if variant in {None, ""} else str(variant).strip()
    if candidate:
        if candidate not in available:
            supported = ", ".join(sorted(available)) or "<none>"
            raise ValueError(
                f"{owner} variant must be one of: {supported}. Received {candidate!r}."
            )
        return candidate
    supported = ", ".join(sorted(available)) or "<none>"
    raise ValueError(
        f"{owner} requires an explicit variant; "
        f"implicit default/singleton fallback is not supported. "
        f"Supported variants: {supported}."
    )


def _canonical_modalities(
    dataset_id: str,
    *,
    modalities: Sequence[str] | None,
    supported: tuple[str, ...],
) -> tuple[str, ...]:
    if modalities is None:
        if len(supported) == 1:
            return (supported[0],)
        available = ", ".join(supported) or "<none>"
        raise ValueError(
            f"Dataset '{dataset_id}' requires explicit modalities because it exposes multiple "
            f"modalities. Supported modalities: {available}"
        )

    if isinstance(modalities, str):
        raise TypeError("datasets.load(..., modalities=...) expects a sequence of modality names")

    canonical: list[str] = []
    for item in modalities:
        token = str(item).strip()
        if not token:
            continue
        canonical.append(token)
    if not canonical:
        if len(supported) == 1:
            return (supported[0],)
        available = ", ".join(supported) or "<none>"
        raise ValueError(
            "datasets.load(..., modalities=...) requires at least one non-empty modality for "
            f"dataset '{dataset_id}'. Supported modalities: {available}"
        )

    deduplicated = tuple(dict.fromkeys(canonical))
    if len(deduplicated) != len(canonical):
        raise ValueError("datasets.load(..., modalities=...) received duplicate modality entries")
    return deduplicated


def _canonical_variant_key(
    dataset_id: str,
    *,
    variant: str | None,
    supported: tuple[str, ...],
) -> str | None:
    if not supported:
        return None if variant in {None, ""} else str(variant)
    if variant in {None, ""}:
        choices = ", ".join(str(item) for item in supported)
        raise ValueError(
            f"Dataset '{dataset_id}' requires an explicit variant. "
            "Implicit default/singleton fallback is not supported. "
            f"Expected one of the canonical variant keys: {choices}"
        )

    resolved_variant = str(variant)
    if resolved_variant not in supported:
        choices = ", ".join(str(item) for item in supported)
        raise ValueError(
            f"Unsupported variant '{variant}' for dataset '{dataset_id}'. "
            f"Expected one of the canonical variant keys: {choices}"
        )
    return resolved_variant


def _canonical_binding(
    dataset_id: str,
    *,
    binding_kind: str,
    binding_id: str | None,
    supported: tuple[str, ...],
) -> str | None:
    if binding_id in {None, ""}:
        return None
    resolved_binding = str(binding_id)
    if resolved_binding not in supported:
        choices = ", ".join(supported)
        raise ValueError(
            f"Unsupported {binding_kind} '{resolved_binding}' for dataset '{dataset_id}'. "
            f"Supported {binding_kind}s: {choices}"
        )
    return resolved_binding


def _canonical_task_binding(
    dataset_id: str,
    *,
    task_binding: str | None,
    supported: tuple[str, ...],
) -> str | None:
    if task_binding in {None, ""}:
        return None
    return _canonical_binding(
        dataset_id,
        binding_kind="task_binding",
        binding_id=str(task_binding).strip(),
        supported=supported,
    )


def _task_id_for_request(request: "DatasetLoadRequest") -> str | None:
    if request.task_binding in {None, ""}:
        return None
    from octosense.datasets.catalog import get_dataset_binding_payload

    payload = get_dataset_binding_payload(
        request.dataset_id,
        binding_kind="task_binding",
        binding_id=str(request.task_binding),
    )
    task_id = str(payload.get("task_id", "") or "").strip()
    return task_id or None


def _resolve_common_target_field_bridge(
    views_by_split: Mapping[str, "DatasetView"],
) -> dict[str, str] | None:
    resolved_bridge: dict[str, str] | None = None
    resolved_split: str | None = None
    for split_name, view in views_by_split.items():
        provider = getattr(view, "get_execution_target_bridge", None)
        bridge = provider() if callable(provider) else None
        if bridge is None:
            continue
        normalized_bridge = dict(bridge)
        if resolved_bridge is None:
            resolved_bridge = normalized_bridge
            resolved_split = split_name
            continue
        if normalized_bridge != resolved_bridge:
            raise ValueError(
                "Materialized split views must agree on target_field_bridge; "
                f"split '{split_name}' does not match '{resolved_split}'."
            )
    return resolved_bridge


def normalize_public_load_request(
    dataset_id: str,
    *,
    modalities: Sequence[str] | None,
    variant: str | None,
    split_scheme: str | None,
    task_binding: str | None,
    path: str | Path | None,
) -> dict[str, Any]:
    """Normalize public datasets.load(...) selectors through the request owner."""
    contract = _declared_dataset_contract_from_sidecars(dataset_id)
    resolved_modalities = _canonical_modalities(
        dataset_id,
        modalities=modalities,
        supported=contract.modalities,
    )
    unsupported = [item for item in resolved_modalities if item not in contract.modalities]
    if unsupported:
        available = ", ".join(contract.modalities)
        raise ValueError(
            f"Unsupported modalities {unsupported!r} for dataset '{dataset_id}'. "
            f"Supported modalities: {available}"
        )

    return {
        "dataset_id": dataset_id,
        "variant": _canonical_variant_key(
            dataset_id,
            variant=variant,
            supported=contract.supported_variants,
        ),
        "modalities": resolved_modalities,
        "split_scheme": _canonical_binding(
            dataset_id,
            binding_kind="split_scheme",
            binding_id=split_scheme,
            supported=contract.split_schemes,
        ),
        "task_binding": _canonical_task_binding(
            dataset_id,
            task_binding=task_binding,
            supported=contract.task_bindings,
        ),
        "dataset_root": (None if path is None else Path(path).expanduser().resolve()),
        "source_root": None,
    }


def _builtin_owner_module(adapter: "BaseDatasetAdapter"):
    module_path = adapter.builtin_build_module
    if module_path in {None, ""}:
        raise NotImplementedError(
            f"Dataset adapter '{adapter.dataset_id}' does not declare builtin_build_module."
        )
    return import_module(str(module_path))


def get_dataset_storage_subdir(dataset_id: str) -> str:
    from octosense.datasets.catalog import get_dataset_card

    return str(get_dataset_card(dataset_id).default_root_subdir)


def resolve_dataset_root(
    dataset_id: str,
    *,
    override: str | Path | None,
) -> Path:
    """Resolve the canonical on-disk root for one builtin dataset."""

    if override is not None:
        return Path(override).expanduser().resolve()

    default_root_subdir = get_dataset_storage_subdir(dataset_id)
    datasets_root = os.environ.get(OCTOSENSE_DATASETS_ROOT_ENV)
    if not datasets_root:
        raise ValueError(
            f"Dataset '{dataset_id}' requires an explicit path or "
            f"{OCTOSENSE_DATASETS_ROOT_ENV}=... so the dataset owner can resolve "
            f"'{default_root_subdir}'."
        )
    return (Path(datasets_root).expanduser().resolve() / default_root_subdir).resolve()


def normalize_public_dataset_options(
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize public dataset options into a plain request-owned mapping."""

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("datasets.load(..., options=...) expects a mapping when provided.")
    return {
        str(key): _normalize_public_dataset_option_value(value)
        for key, value in options.items()
    }


def _normalize_public_dataset_option_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_public_dataset_option_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_normalize_public_dataset_option_value(item) for item in value)
    return value


def freeze_public_dataset_options_for_cache(
    options: Mapping[str, Any],
) -> tuple[tuple[str, object], ...]:
    """Convert normalized public dataset options into a stable hash-safe cache key."""

    return tuple(
        sorted(
            (str(key), _freeze_public_dataset_option_value(value))
            for key, value in options.items()
        )
    )


def _freeze_public_dataset_option_value(value: Any) -> object:
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_public_dataset_option_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_public_dataset_option_value(item) for item in value)
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            "datasets.load(..., options=...) only accepts hash-safe leaf values plus nested "
            f"mappings/sequences; received unsupported value type {type(value).__name__}."
        ) from exc
    return value


@dataclass(frozen=True)
class DatasetLoadRequest:
    """Canonical dataset request shared by config, api, and registry."""

    dataset_id: str
    variant: str | None = None
    modalities: tuple[str, ...] = ()
    split_scheme: str | None = None
    task_binding: str | None = None
    dataset_root: Path | None = None
    source_root: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_public_load(
        cls,
        dataset_id: str,
        *,
        modalities: tuple[str, ...] | list[str] | None,
        variant: str | None,
        split_scheme: str | None,
        task_binding: str | None,
        path: str | Path | None,
        options: Mapping[str, Any] | None = None,
    ) -> "DatasetLoadRequest":
        normalized_request = normalize_public_load_request(
            dataset_id,
            modalities=modalities,
            variant=variant,
            split_scheme=split_scheme,
            task_binding=task_binding,
            path=path,
        )
        normalized_request["options"] = normalize_public_dataset_options(options)
        return cls(**normalized_request)

    def validate_adapter_support(self, adapter: "BaseDatasetAdapter") -> None:
        if self.split_scheme not in {None, ""} and not adapter.supports_split_scheme:
            raise NotImplementedError(
                "datasets.load(..., split_scheme=...) now resolves through DatasetLoadRequest, "
                f"but dataset '{adapter.dataset_id}' has not wired builtin split binding materialization yet."
            )
        if self.task_binding not in {None, ""} and not adapter.supports_task_binding:
            raise NotImplementedError(
                "datasets.load(..., task_binding=...) now resolves through DatasetLoadRequest, "
                f"but dataset '{adapter.dataset_id}' has not wired builtin task binding materialization yet."
            )

    def selected_modality(self) -> str | None:
        """Return the modality when the request already selects exactly one."""

        if len(self.modalities) != 1:
            return None
        return self.modalities[0]

    def require_single_modality(self, *, owner: str) -> str:
        """Internal helper for dataset owners that can project only one modality."""

        if not self.modalities:
            raise ValueError(f"{owner} requires exactly one modality selection.")
        if len(self.modalities) != 1:
            raise ValueError(
                f"{owner} requires exactly one modality selection; "
                f"received {list(self.modalities)!r}."
            )
        return self.modalities[0]

    def to_build_request(
        self,
        *,
        cache_root: Path | None = None,
    ) -> "DatasetBuildRequest":
        from octosense.datasets.core.builder import DatasetBuildRequest

        return DatasetBuildRequest(
            dataset_id=self.dataset_id,
            variant=self.variant,
            split_scheme=self.split_scheme,
            task_binding=self.task_binding,
            modalities=self.modalities,
            dataset_root=self.dataset_root,
            source_root=self.source_root,
            cache_root=cache_root,
            options=dict(self.options),
        )


class BaseDatasetAdapter(ABC):
    """OO boundary for dataset registration and view materialization."""

    dataset_id: str
    modality: str
    request_tokens: tuple[str, ...]
    builtin_build_module: str | None = None
    supports_split_scheme: bool = False
    supports_task_binding: bool = False

    def build_artifact(
        self,
        request: DatasetLoadRequest,
        *,
        dataset_root: Path | None = None,
    ) -> "DatasetBuildArtifact":
        if self.builtin_build_module in {None, ""}:
            raise NotImplementedError(
                f"Dataset adapter '{self.dataset_id}' does not declare builtin_build_module."
            )
        from octosense.datasets.core.builder import build_builtin_artifact

        build_request = request.to_build_request()
        if dataset_root is not None:
            build_request = replace(build_request, dataset_root=dataset_root)
        return build_builtin_artifact(
            build_request,
            module_path=str(self.builtin_build_module),
        )

    def build_view(
        self,
        request: DatasetLoadRequest,
        *,
        split: str | None = None,
    ) -> "DatasetView":
        if self.builtin_build_module in {None, ""}:
            raise NotImplementedError(
                f"Dataset adapter '{self.dataset_id}' must implement build_view() "
                "or declare builtin_build_module for artifact-backed view materialization."
            )

        artifact = self.build_artifact(request)
        (
            variant,
            label_mapping,
            task_kind,
            target_kind,
            target_schema,
            views_by_split,
        ) = self.materialize_views_from_artifact(request, artifact)
        resolved_task_id = _task_id_for_request(request)
        target_field_bridge = _resolve_common_target_field_bridge(views_by_split)
        return merge_materialized_split_views(
            dataset_id=self.dataset_id,
            variant=variant,
            split=split,
            task_id=resolved_task_id,
            task_kind=task_kind,
            target_kind=target_kind,
            label_mapping=label_mapping,
            target_schema=target_schema,
            target_field_bridge=target_field_bridge,
            views_by_split=views_by_split,
        )

    def materialize_views_from_artifact(
        self,
        request: DatasetLoadRequest,
        artifact: "DatasetBuildArtifact",
    ) -> "DatasetMaterializationPayload":
        module = _builtin_owner_module(self)
        hook = getattr(module, "materialize_views_from_artifact", None)
        if not callable(hook):
            raise NotImplementedError(
                f"Builtin dataset module '{module.__name__}' must own artifact-backed "
                "view materialization via "
                "'materialize_views_from_artifact(request=..., artifact=...)'."
            )
        payload = hook(request=request, artifact=artifact)
        if not isinstance(payload, tuple) or len(payload) != 6:
            raise TypeError(
                f"Builtin dataset module '{module.__name__}' hook "
                "'materialize_views_from_artifact' returned an invalid materialization payload."
            )
        return payload

    def download_card_id(self, request: DatasetLoadRequest) -> str | None:
        del request
        return None
