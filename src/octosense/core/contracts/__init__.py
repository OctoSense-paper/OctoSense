"""Shared declarative cross-module contracts."""

from octosense.core.contracts.axis import AxisContract, MetadataRequirement
from octosense.core.contracts.model import ModelInputContract, ModelOutputContract
from octosense.core.contracts.task import TaskOutputSpec
from octosense.core.contracts.transform import TransformContract

__all__ = [
    "AxisContract",
    "MetadataRequirement",
    "ModelInputContract",
    "ModelOutputContract",
    "TaskOutputSpec",
    "TransformContract",
]
