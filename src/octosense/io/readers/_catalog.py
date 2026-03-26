"""Internal reader catalog keyed by canonical ``<modality>/<device_family>`` ids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from octosense.io.readers._base import ReaderDescriptor


@dataclass(frozen=True)
class ReaderCatalogEntry:
    descriptor: ReaderDescriptor
    metadata: dict[str, Any]

    @property
    def reader_id(self) -> str:
        return self.descriptor.reader_id

    @property
    def import_path(self) -> str:
        return self.descriptor.import_path


_CANONICAL_READER_ID_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*/[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
)


def readers_root() -> Path:
    return Path(__file__).resolve().parent


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(payload)!r}")
    return payload


def _guess_reader_class(device_family: str) -> str:
    special_cases = {
        "rssi_csv": "BLERSSIReader",
        "ti_dca1000": "TI_DCA1000Reader",
        "qorvo_cir": "QorvoCIRReader",
        "wav": "WAVReader",
    }
    if device_family in special_cases:
        return special_cases[device_family]
    return "".join(part.upper() if part.isdigit() else part.capitalize() for part in device_family.split("_")) + "Reader"


def iter_reader_catalog() -> tuple[ReaderCatalogEntry, ...]:
    entries: list[ReaderCatalogEntry] = []
    root = readers_root()
    for modality_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(("_", "."))):
        for device_dir in sorted(
            path
            for path in modality_dir.iterdir()
            if path.is_dir() and not path.name.startswith(("_", "."))
        ):
            reader_path = device_dir / "reader.py"
            if not reader_path.exists():
                continue
            metadata_path = device_dir / "metadata.yaml"
            metadata = _load_metadata(metadata_path)
            modality = str(metadata.get("modality", modality_dir.name))
            device_family = str(metadata.get("device_family", device_dir.name))
            reader_class = str(metadata.get("reader_class", _guess_reader_class(device_family)))
            descriptor = ReaderDescriptor(
                modality=modality,
                device_family=device_family,
                reader_module=f"octosense.io.readers.{modality_dir.name}.{device_dir.name}.reader",
                reader_class=reader_class,
                package_root=device_dir,
                metadata_path=metadata_path if metadata_path.exists() else None,
                extra=metadata,
            )
            entries.append(ReaderCatalogEntry(descriptor=descriptor, metadata=metadata))
    return tuple(entries)


def reader_catalog_by_id() -> dict[str, ReaderCatalogEntry]:
    return {entry.reader_id: entry for entry in iter_reader_catalog()}

def canonicalize_reader_id(reader_id: str) -> str:
    normalized = reader_id.strip().lower()
    if not _CANONICAL_READER_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Reader id must use canonical '<modality>/<device_family>' form, got {reader_id!r}"
        )
    return normalized


__all__ = [
    "ReaderCatalogEntry",
    "canonicalize_reader_id",
    "iter_reader_catalog",
    "reader_catalog_by_id",
    "readers_root",
]
