"""Stable shared abstractions for OctoSense."""

from __future__ import annotations

from octosense.core.contracts import (
    AxisContract,
    MetadataRequirement,
    ModelInputContract,
    ModelOutputContract,
    TaskOutputSpec,
    TransformContract,
)
from octosense.core.describe import Describable, DescribeNode, ensure_describe_node
from octosense.core.errors import (
    ContractError,
    DimensionError,
    MetadataError,
    OctoSenseError,
    SchemaValidationError,
)
from octosense.core.types import ComplexTensor, DeviceLike, RealTensor, TensorLike

__all__ = [
    # Contracts
    "AxisContract",
    "MetadataRequirement",
    "TransformContract",
    "ModelInputContract",
    "ModelOutputContract",
    "TaskOutputSpec",
    # Description
    "DescribeNode",
    "Describable",
    "ensure_describe_node",
    # Errors
    "OctoSenseError",
    "ContractError",
    "DimensionError",
    "MetadataError",
    "SchemaValidationError",
    # Type aliases
    "ComplexTensor",
    "DeviceLike",
    "RealTensor",
    "TensorLike",
]
