"""Internal dataset_id -> builder adapter registry."""

from __future__ import annotations

from functools import lru_cache

from octosense.datasets.base import BaseDatasetAdapter, DatasetLoadRequest
from octosense.datasets.views.dataset_view import DatasetView


class BuiltinDatasetAdapter(BaseDatasetAdapter):
    """Thin discovery entry that delegates dataset materialization to builtin owners."""

    def __init__(
        self,
        *,
        dataset_id: str,
        modality: str,
        builtin_build_module: str,
        request_tokens: tuple[str, ...],
        supports_split_scheme: bool = False,
        supports_task_binding: bool = False,
    ) -> None:
        self.dataset_id = dataset_id
        self.modality = modality
        self.builtin_build_module = builtin_build_module
        self.request_tokens = request_tokens
        self.supports_split_scheme = supports_split_scheme
        self.supports_task_binding = supports_task_binding

    def build_view(
        self,
        request: DatasetLoadRequest,
        *,
        split: str | None = None,
    ) -> DatasetView:
        return BaseDatasetAdapter.build_view(self, request, split=split)


_DATASET_ADAPTERS: tuple[BaseDatasetAdapter, ...] = (
    BuiltinDatasetAdapter(
        dataset_id="widar3",
        modality="wifi",
        builtin_build_module="octosense.datasets.builtin.widar3.build",
        request_tokens=("widar3",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="signfi",
        modality="wifi",
        builtin_build_module="octosense.datasets.builtin.signfi.build",
        request_tokens=("signfi",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="falldar",
        modality="wifi",
        builtin_build_module="octosense.datasets.builtin.falldar.build",
        request_tokens=("falldar",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="csi_bench",
        modality="wifi",
        builtin_build_module="octosense.datasets.builtin.csi_bench.build",
        request_tokens=("csi_bench",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="widargait",
        modality="wifi",
        builtin_build_module="octosense.datasets.builtin.widargait.build",
        request_tokens=("widargait",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="hupr",
        modality="mmwave",
        builtin_build_module="octosense.datasets.builtin.hupr.build",
        request_tokens=("hupr",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="mmfi",
        modality="multimodal",
        builtin_build_module="octosense.datasets.builtin.mmfi.build",
        request_tokens=("mmfi",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="octonet",
        modality="wifi",
        builtin_build_module="octosense.datasets.builtin.octonet.build",
        request_tokens=("octonet",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="xrf55",
        modality="multimodal",
        builtin_build_module="octosense.datasets.builtin.xrf55.build",
        request_tokens=("xrf55",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
    BuiltinDatasetAdapter(
        dataset_id="xrfv2",
        modality="multimodal",
        builtin_build_module="octosense.datasets.builtin.xrfv2.build",
        request_tokens=("xrfv2",),
        supports_split_scheme=True,
        supports_task_binding=True,
    ),
)


def _iter_dataset_adapters() -> tuple[BaseDatasetAdapter, ...]:
    return _DATASET_ADAPTERS


@lru_cache(maxsize=1)
def _dataset_adapters_by_id() -> dict[str, BaseDatasetAdapter]:
    return {adapter.dataset_id: adapter for adapter in _DATASET_ADAPTERS}


def resolve_dataset_adapter(request: DatasetLoadRequest) -> BaseDatasetAdapter:
    adapter = _dataset_adapters_by_id().get(request.dataset_id)
    if adapter is None:
        supported = ", ".join(sorted(_dataset_adapters_by_id()))
        raise ValueError(
            f"Unsupported dataset '{request.dataset_id}'. Expected one of: {supported}"
        )
    request.validate_adapter_support(adapter)
    return adapter
