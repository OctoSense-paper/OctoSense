"""Builtin-first dataset catalog metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml


DatasetBindingKind = Literal["config", "task_binding", "split_scheme"]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping payload in {path}, got {type(payload)!r}")
    return {str(key): value for key, value in payload.items()}


def _optional_file(path: Path) -> Path | None:
    return path.resolve() if path.is_file() else None


def _binding_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path.resolve()
        for path in sorted(directory.glob("*.yaml"))
        if path.is_file()
    }


def _binding_label(binding_kind: DatasetBindingKind) -> str:
    if binding_kind == "config":
        return "config"
    if binding_kind == "task_binding":
        return "task_binding"
    if binding_kind == "split_scheme":
        return "split_scheme"
    raise ValueError(f"Unsupported dataset binding kind '{binding_kind}'.")


def _read_declared_dataset_id(card_path: Path | None, *, fallback: str) -> str:
    if card_path is None:
        raise ValueError(
            f"Builtin dataset directory '{fallback}' is missing required card.yaml."
        )
    payload = _load_yaml_mapping(card_path)
    declared = payload.get("dataset_id")
    if declared in {None, ""}:
        raise ValueError(
            f"Builtin dataset directory '{fallback}' must declare non-empty dataset_id in "
            f"{card_path}."
        )
    dataset_id = str(declared)
    if dataset_id != fallback:
        raise ValueError(
            f"Builtin dataset directory '{fallback}' must declare matching dataset_id in "
            f"{card_path}; received {dataset_id!r}."
        )
    return dataset_id


@dataclass(frozen=True)
class _BuiltinBindingSet:
    """Structured paths for builtin dataset binding sidecars."""

    configs: dict[str, Path] = field(default_factory=dict)
    task_bindings: dict[str, Path] = field(default_factory=dict)
    splits: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class _BuiltinDatasetLayout:
    """Catalog-private builtin dataset definition layout."""

    dataset_id: str
    root: Path
    card_path: Path | None = None
    schema_path: Path | None = None
    citation_path: Path | None = None
    bindings: _BuiltinBindingSet = field(default_factory=_BuiltinBindingSet)


def _load_builtin_layout(layout_root: Path) -> _BuiltinDatasetLayout:
    root = layout_root.resolve()
    card_path = _optional_file(root / "card.yaml")
    bindings_root = root / "bindings"
    return _BuiltinDatasetLayout(
        dataset_id=_read_declared_dataset_id(card_path, fallback=root.name),
        root=root,
        card_path=card_path,
        schema_path=_optional_file(root / "schema.yaml"),
        citation_path=_optional_file(root / "citation.bib"),
        bindings=_BuiltinBindingSet(
            configs=_binding_map(bindings_root / "configs"),
            task_bindings=_binding_map(bindings_root / "task_bindings"),
            splits=_binding_map(bindings_root / "splits"),
        ),
    )


def _discover_builtin_layouts(builtin_root: Path) -> dict[str, _BuiltinDatasetLayout]:
    root = builtin_root.resolve()
    layouts: dict[str, _BuiltinDatasetLayout] = {}
    if not root.exists():
        return layouts
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("_"):
            continue
        layout = _load_builtin_layout(candidate)
        layouts[str(layout.dataset_id)] = layout
    return layouts


@lru_cache(maxsize=None)
def _load_builtin_layouts(builtin_root: Path) -> dict[str, _BuiltinDatasetLayout]:
    return _discover_builtin_layouts(builtin_root.resolve())


def _get_builtin_layout(builtin_root: Path, dataset_id: str) -> _BuiltinDatasetLayout | None:
    return _load_builtin_layouts(builtin_root.resolve()).get(dataset_id)


@dataclass(frozen=True)
class DatasetAccessInfo:
    """How one dataset can be accessed."""

    kind: Literal["public_download", "application_required", "manual_only"]
    public_urls: tuple[str, ...] = field(default_factory=tuple)
    application_url: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "public_urls": list(self.public_urls),
            "application_url": self.application_url,
            "note": self.note,
        }


@dataclass(frozen=True)
class DatasetLicenseInfo:
    """License metadata used for download policy decisions."""

    name: str = ""
    spdx_id: str | None = None
    source_url: str | None = None
    redistribution_allowed: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spdx_id": self.spdx_id,
            "source_url": self.source_url,
            "redistribution_allowed": self.redistribution_allowed,
            "note": self.note,
        }


@dataclass(frozen=True)
class DatasetDownloadPolicy:
    """Policy for whether OctoSense may automate dataset preparation."""

    allow_automated_download: bool = False
    requires_explicit_opt_in: bool = True
    supported_parts: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_automated_download": self.allow_automated_download,
            "requires_explicit_opt_in": self.requires_explicit_opt_in,
            "supported_parts": list(self.supported_parts),
            "note": self.note,
        }


@dataclass(frozen=True)
class DatasetPaperInfo:
    """Paper metadata for one dataset."""

    title: str
    url: str
    authors: tuple[str, ...]
    citation_text: str
    citation_bibtex: str


@dataclass(frozen=True)
class DatasetStoragePolicy:
    """Default on-disk storage contract for one dataset."""

    default_root_subdir: str
    canonical_display_name: str | None = None
    downloads_dirname: str = "_downloads"
    cache_dirname: str = "_octosense_cache"


@dataclass(frozen=True)
class DatasetCard:
    """Queryable dataset card for the datasets public surface."""

    dataset_id: str
    display_name: str
    modalities: tuple[str, ...]
    access: DatasetAccessInfo
    paper: DatasetPaperInfo
    storage: DatasetStoragePolicy
    license: DatasetLicenseInfo = field(default_factory=DatasetLicenseInfo)
    download: DatasetDownloadPolicy = field(default_factory=DatasetDownloadPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    supported_variants: tuple[str, ...] = field(default_factory=tuple)
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def default_root_subdir(self) -> str:
        return self.storage.default_root_subdir


@dataclass(frozen=True)
class DatasetDownloadConfig:
    """Catalog-owned raw and structured download metadata for one dataset card."""

    dataset_id: str
    handler: str | None = None
    sources: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    parts: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _copy_catalog_value(self.raw_payload)


@dataclass(frozen=True)
class DatasetSchema:
    """Queryable dataset schema parsed from the builtin catalog."""

    dataset_id: str
    schema_version: int
    summary: str = ""
    variant: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "summary": self.summary,
            "variant": self.variant,
            "fields": dict(self.fields),
        }


def _datasets_root() -> Path:
    return Path(__file__).resolve().parent


def _builtin_root() -> Path:
    return _datasets_root() / "builtin"


def _catalog_dataset_ids() -> tuple[str, ...]:
    return tuple(sorted(_load_builtin_layouts(_builtin_root())))


def _binding_paths(
    dataset_id: str,
    *,
    binding_kind: DatasetBindingKind,
) -> dict[str, Path]:
    layout = _get_builtin_layout(_builtin_root(), dataset_id)
    if layout is None:
        raise ValueError(
            f"Dataset '{dataset_id}' is missing builtin catalog metadata under datasets/builtin."
        )
    if binding_kind == "config":
        return dict(layout.bindings.configs)
    if binding_kind == "task_binding":
        return dict(layout.bindings.task_bindings)
    if binding_kind == "split_scheme":
        return dict(layout.bindings.splits)
    raise ValueError(f"Unsupported dataset binding kind '{binding_kind}'.")


def _binding_inventory(
    dataset_id: str,
    *,
    binding_kind: DatasetBindingKind,
) -> tuple[str, ...]:
    return tuple(sorted(_binding_paths(dataset_id, binding_kind=binding_kind)))


def _binding_payload(
    dataset_id: str,
    *,
    binding_kind: DatasetBindingKind,
    binding_id: str,
) -> dict[str, Any]:
    bindings = _binding_paths(dataset_id, binding_kind=binding_kind)
    path = bindings.get(binding_id)
    if path is not None:
        payload = _load_yaml_mapping(path)
        return {"binding_id": str(binding_id), **payload}

    available = ", ".join(_binding_inventory(dataset_id, binding_kind=binding_kind)) or "<none>"
    raise KeyError(
        f"Unknown {_binding_label(binding_kind)} '{binding_id}' for dataset "
        f"'{dataset_id}'. Available: {available}"
    )


def _load_access_info(payload: dict[str, Any]) -> DatasetAccessInfo:
    return DatasetAccessInfo(
        kind=str(payload["kind"]),
        public_urls=tuple(str(url) for url in payload.get("public_urls", [])),
        application_url=(
            None
            if payload.get("application_url") in {None, ""}
            else str(payload["application_url"])
        ),
        note=str(payload.get("note", "")),
    )


def _load_license_info(payload: dict[str, Any]) -> DatasetLicenseInfo:
    return DatasetLicenseInfo(
        name=str(payload.get("name", "")),
        spdx_id=(None if payload.get("spdx_id") in {None, ""} else str(payload["spdx_id"])),
        source_url=(
            None if payload.get("source_url") in {None, ""} else str(payload["source_url"])
        ),
        redistribution_allowed=bool(payload.get("redistribution_allowed", False)),
        note=str(payload.get("note", "")),
    )


def _load_download_policy(payload: dict[str, Any]) -> DatasetDownloadPolicy:
    return DatasetDownloadPolicy(
        allow_automated_download=bool(payload.get("allow_automated_download", False)),
        requires_explicit_opt_in=bool(payload.get("requires_explicit_opt_in", True)),
        supported_parts=tuple(str(part) for part in payload.get("supported_parts", [])),
        note=str(payload.get("note", "")),
    )


def _load_storage_policy(payload: dict[str, Any]) -> DatasetStoragePolicy:
    return DatasetStoragePolicy(
        default_root_subdir=str(payload["default_root_subdir"]),
        canonical_display_name=(
            None
            if payload.get("canonical_display_name") in {None, ""}
            else str(payload["canonical_display_name"])
        ),
        downloads_dirname=str(payload.get("downloads_dirname", "_downloads")),
        cache_dirname=str(payload.get("cache_dirname", "_octosense_cache")),
    )


def _load_paper_info(payload: dict[str, Any], *, bibtex_path: Path) -> DatasetPaperInfo:
    return DatasetPaperInfo(
        title=str(payload["title"]),
        url=str(payload["url"]),
        authors=tuple(str(author) for author in payload.get("authors", [])),
        citation_text=str(payload["citation_text"]),
        citation_bibtex=bibtex_path.read_text(encoding="utf-8").strip(),
    )


def _copy_catalog_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_catalog_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_catalog_value(item) for item in value]
    return value


def _normalize_download_source_list(value: Any, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if value is None or value == "":
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Expected list field '{field_name}' in dataset card download config.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected mapping entry '{field_name}[{index}]' in dataset card download config."
            )
        normalized.append(_copy_catalog_value(item))
    return tuple(normalized)


def _normalize_download_parts(value: Any, *, field_name: str) -> dict[str, dict[str, Any]]:
    if value is None or value == "":
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping field '{field_name}' in dataset card download config.")
    normalized: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected mapping entry '{field_name}.{key}' in dataset card download config."
            )
        normalized[str(key)] = _copy_catalog_value(item)
    return normalized


def _load_dataset_download_config(
    dataset_id: str,
    *,
    card_path: Path,
    payload: Any,
) -> DatasetDownloadConfig:
    if payload is None or payload == "":
        normalized_payload: dict[str, Any] = {}
    elif isinstance(payload, dict):
        normalized_payload = _copy_catalog_value(payload)
    else:
        raise ValueError(f"Expected mapping field 'download' in {card_path}")
    handler = normalized_payload.get("handler")
    normalized_handler = None if handler in {None, ""} else str(handler)
    return DatasetDownloadConfig(
        dataset_id=dataset_id,
        handler=normalized_handler,
        sources=_normalize_download_source_list(
            normalized_payload.get("sources"),
            field_name="download.sources",
        ),
        parts=_normalize_download_parts(
            normalized_payload.get("parts"),
            field_name="download.parts",
        ),
        raw_payload=normalized_payload,
    )


def _load_variant_payloads(
    dataset_id: str,
) -> dict[str, dict[str, Any]]:
    layout = _get_builtin_layout(_builtin_root(), dataset_id)
    if layout is None:
        raise ValueError(
            f"Dataset '{dataset_id}' is missing builtin catalog metadata under datasets/builtin."
        )
    variants: dict[str, dict[str, Any]] = {}
    for binding_id, path in sorted(layout.bindings.configs.items()):
        payload = _load_yaml_mapping(path)
        if payload.get("variant_key") in {None, ""}:
            raise ValueError(
                f"Dataset config binding '{binding_id}' must declare 'variant_key' explicitly."
            )
        variant_key = str(payload["variant_key"])
        normalized = dict(payload)
        if normalized.get("variant") in {None, ""}:
            raise ValueError(
                f"Dataset config binding '{binding_id}' must declare 'variant' explicitly."
            )
        variants[variant_key] = normalized
    return variants


def _load_dataset_schema(
    dataset_id: str,
    *,
    layout: _BuiltinDatasetLayout | None,
) -> DatasetSchema:
    if layout is None:
        raise ValueError(
            f"Dataset '{dataset_id}' is missing builtin catalog metadata under datasets/builtin."
        )
    schema_path = layout.schema_path
    if schema_path is None:
        raise ValueError(f"Dataset '{dataset_id}' is missing builtin schema.yaml.")
    payload = _load_yaml_mapping(schema_path)
    if "fields" not in payload:
        raise ValueError(f"Dataset schema {schema_path} must declare top-level mapping field 'fields'.")
    fields_payload = payload["fields"]
    if not isinstance(fields_payload, dict):
        raise ValueError(f"Expected mapping field 'fields' in {schema_path}")
    if not fields_payload:
        raise ValueError(f"Dataset schema {schema_path} must declare at least one structured field.")
    fields = _copy_catalog_value(fields_payload)
    declared_dataset_id = payload.get("dataset_id")
    if declared_dataset_id not in {None, ""} and str(declared_dataset_id) != dataset_id:
        raise ValueError(
            f"Dataset schema in {schema_path} declares dataset_id={declared_dataset_id!r}, "
            f"expected {dataset_id!r}."
        )
    schema_version = payload.get("schema_version")
    if schema_version in {None, ""}:
        raise ValueError(f"Dataset schema {schema_path} must declare top-level integer 'schema_version'.")
    try:
        normalized_schema_version = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Dataset schema {schema_path} must declare integer 'schema_version', got {schema_version!r}."
        ) from exc
    variant = None if payload.get("variant") in {None, ""} else str(payload["variant"])
    summary_value = fields.get("summary", "")
    if summary_value in {None, ""}:
        summary = ""
    elif isinstance(summary_value, str):
        summary = summary_value
    else:
        raise ValueError(
            f"Dataset schema {schema_path} must declare string 'fields.summary', got "
            f"{type(summary_value)!r}."
        )
    return DatasetSchema(
        dataset_id=dataset_id,
        schema_version=normalized_schema_version,
        summary=summary,
        variant=variant,
        fields=fields,
    )


def _load_dataset_card(
    dataset_id: str,
    *,
    layout: _BuiltinDatasetLayout | None,
) -> tuple[str, dict[str, Any], DatasetCard]:
    if layout is None:
        raise ValueError(
            f"Dataset '{dataset_id}' is missing builtin catalog metadata under datasets/builtin."
        )
    card_path = layout.card_path
    if card_path is None:
        raise ValueError(f"Dataset '{dataset_id}' is missing builtin card.yaml.")
    card_payload = _load_yaml_mapping(card_path)
    bibtex_path = layout.citation_path
    if bibtex_path is None:
        raise ValueError(f"Dataset '{dataset_id}' is missing builtin citation.bib.")
    supported_modalities = tuple(str(item) for item in card_payload.get("modalities", []))
    variant_payloads = _load_variant_payloads(dataset_id)
    metadata = {}
    raw_metadata = card_payload.get("metadata")
    if raw_metadata is not None:
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"Expected mapping field 'metadata' in {card_path}")
        metadata = dict(raw_metadata)

    paper_payload = card_payload.get("paper", {})
    if not isinstance(paper_payload, dict):
        raise ValueError(f"Expected mapping field 'paper' in {card_path}")

    card = DatasetCard(
        dataset_id=dataset_id,
        display_name=str(card_payload["display_name"]),
        modalities=supported_modalities,
        access=_load_access_info(dict(card_payload.get("access", {}))),
        paper=_load_paper_info(dict(paper_payload), bibtex_path=bibtex_path),
        storage=_load_storage_policy(dict(card_payload.get("storage", {}))),
        license=_load_license_info(dict(card_payload.get("license", {}))),
        download=_load_download_policy(dict(card_payload.get("download", {}))),
        metadata=metadata,
        supported_variants=tuple(str(item) for item in card_payload.get("supported_variants", [])),
        variants=variant_payloads,
    )
    return dataset_id, card_payload, card


@lru_cache(maxsize=1)
def dataset_cards() -> dict[str, DatasetCard]:
    cards: dict[str, DatasetCard] = {}
    for dataset_id in _catalog_dataset_ids():
        layout = _get_builtin_layout(_builtin_root(), dataset_id)
        _, _, card = _load_dataset_card(dataset_id, layout=layout)
        cards[dataset_id] = card
    return cards


@lru_cache(maxsize=1)
def dataset_schemas() -> dict[str, DatasetSchema]:
    schemas: dict[str, DatasetSchema] = {}
    for dataset_id in _catalog_dataset_ids():
        layout = _get_builtin_layout(_builtin_root(), dataset_id)
        schemas[dataset_id] = _load_dataset_schema(dataset_id, layout=layout)
    return schemas


@lru_cache(maxsize=1)
def dataset_download_configs() -> dict[str, DatasetDownloadConfig]:
    configs: dict[str, DatasetDownloadConfig] = {}
    for dataset_id in _catalog_dataset_ids():
        layout = _get_builtin_layout(_builtin_root(), dataset_id)
        if layout is None or layout.card_path is None:
            raise ValueError(
                f"Dataset '{dataset_id}' is missing builtin catalog metadata under datasets/builtin."
            )
        card_payload = _load_yaml_mapping(layout.card_path)
        configs[dataset_id] = _load_dataset_download_config(
            dataset_id,
            card_path=layout.card_path,
            payload=card_payload.get("download"),
        )
    return configs


_DATASET_CARD_CACHE: dict[str, DatasetCard] = dict(dataset_cards())
_DATASET_DOWNLOAD_CONFIG_CACHE: dict[str, DatasetDownloadConfig] = dict(dataset_download_configs())
_DATASET_SCHEMA_CACHE: dict[str, DatasetSchema] = dict(dataset_schemas())


def get_dataset_card(dataset_id: str) -> DatasetCard:
    try:
        return _DATASET_CARD_CACHE[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASET_CARD_CACHE))
        raise KeyError(f"Unknown dataset_id '{dataset_id}'. Available: {available}") from exc


def get_dataset_schema(dataset_id: str) -> DatasetSchema:
    try:
        return _DATASET_SCHEMA_CACHE[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASET_SCHEMA_CACHE))
        raise KeyError(f"Unknown dataset_id '{dataset_id}'. Available: {available}") from exc


def get_dataset_schema_fields(dataset_id: str) -> dict[str, Any]:
    """Return the canonical structured ``fields`` payload for one builtin dataset schema."""

    return _copy_catalog_value(get_dataset_schema(dataset_id).fields)


def get_dataset_download_config(dataset_id: str) -> DatasetDownloadConfig:
    try:
        return _DATASET_DOWNLOAD_CONFIG_CACHE[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASET_DOWNLOAD_CONFIG_CACHE))
        raise KeyError(f"Unknown dataset_id '{dataset_id}'. Available: {available}") from exc


def list_dataset_binding_ids(
    dataset_id: str,
    *,
    binding_kind: DatasetBindingKind,
) -> tuple[str, ...]:
    """List declared sidecar ids for one builtin dataset binding family."""

    return _binding_inventory(dataset_id, binding_kind=binding_kind)


def get_dataset_binding_payload(
    dataset_id: str,
    *,
    binding_kind: DatasetBindingKind,
    binding_id: str,
) -> dict[str, Any]:
    """Read one builtin binding sidecar payload by canonical binding id only."""

    candidate = str(binding_id).strip()
    if not candidate:
        raise ValueError("Dataset binding payload lookup requires a non-empty binding_id.")
    return _copy_catalog_value(
        _binding_payload(
            dataset_id,
            binding_kind=binding_kind,
            binding_id=candidate,
        )
    )

__all__ = [
    "DatasetAccessInfo",
    "DatasetCard",
    "DatasetDownloadConfig",
    "DatasetDownloadPolicy",
    "DatasetLicenseInfo",
    "DatasetPaperInfo",
    "DatasetSchema",
    "DatasetStoragePolicy",
    "dataset_cards",
    "dataset_download_configs",
    "dataset_schemas",
    "get_dataset_card",
    "get_dataset_binding_payload",
    "get_dataset_download_config",
    "get_dataset_schema",
    "get_dataset_schema_fields",
    "list_dataset_binding_ids",
]
