"""Internal canonical reader contracts for the IO sample layer."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ReaderError(RuntimeError):
    """Reader-local failure with optional source context."""

    def __init__(
        self,
        message: str,
        *,
        offset: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.offset = offset
        self.context = dict(context or {})

    def __str__(self) -> str:
        detail = self.message
        if self.offset is not None:
            detail = f"{detail} (offset={self.offset})"
        if self.context:
            detail = f"{detail} context={self.context}"
        return detail


@dataclass(frozen=True)
class ReaderDescriptor:
    """Canonical identity for one reader implementation."""

    modality: str
    device_family: str
    reader_module: str
    reader_class: str
    package_root: Path | None = None
    metadata_path: Path | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def reader_id(self) -> str:
        return f"{self.modality}/{self.device_family}"

    @property
    def import_path(self) -> str:
        return f"{self.reader_module}:{self.reader_class}"


class CanonicalReader(ABC):
    """Shared logical reader identity for canonical IO readers."""

    modality: str = "unknown"
    device_family: str = "unknown"
    device_name: str = "UnknownDevice"
    reader_version: str = "1.0"

    def __init__(self) -> None:
        self.reader_id = self.descriptor().reader_id

    @classmethod
    def descriptor(cls) -> ReaderDescriptor:
        return ReaderDescriptor(
            modality=str(cls.modality),
            device_family=str(cls.device_family),
            reader_module=cls.__module__,
            reader_class=cls.__name__,
        )


__all__ = ["CanonicalReader", "ReaderDescriptor", "ReaderError"]
