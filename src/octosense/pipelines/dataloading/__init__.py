"""Canonical public surface for pipeline-owned dataloading helpers.

Only execution-time datamodule, collate, and sampler entrypoints are
re-exported here. Historical compatibility facades are intentionally absent.
"""

from octosense.pipelines.dataloading.collate import (
    collate_scalar_index_batch,
    collate_structured_target_batch,
)
from octosense.pipelines.dataloading.datamodule import (
    DataLoaderConfig,
    DatasetSource,
    DatasetSplitMapping,
    build_data_loader,
    build_execution_dataloaders,
    resolve_loader_runtime_kwargs,
)
from octosense.pipelines.dataloading.samplers import build_sampler

__all__ = [
    "DataLoaderConfig",
    "DatasetSource",
    "DatasetSplitMapping",
    "build_data_loader",
    "build_execution_dataloaders",
    "build_sampler",
    "collate_scalar_index_batch",
    "collate_structured_target_batch",
    "resolve_loader_runtime_kwargs",
]
