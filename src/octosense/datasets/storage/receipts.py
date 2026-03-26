"""Download receipt helpers for dataset storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASET_RECEIPT_NAME = "octosense.dataset.json"
DATASET_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetDownloadReceipt:
    dataset_id: str
    display_name: str | None = None
    resolved_root: str | None = None
    license: dict[str, object] = field(default_factory=dict)
    access: dict[str, object] = field(default_factory=dict)
    download_policy: dict[str, object] = field(default_factory=dict)
    selected_part: str | None = None
    selected_source_url: str | None = None
    transport: str | None = None
    downloaded_at: str | None = None
    downloads_dir: str | None = None
    source_root: str | None = None
    force_download: bool = False
    note: str | None = None
    checksums: list[dict[str, object]] = field(default_factory=list)
    sources: list[dict[str, object]] = field(default_factory=list)
    schema_version: int = DATASET_RECEIPT_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetDownloadReceipt":
        return cls(
            dataset_id=str(payload["dataset_id"]),
            display_name=(
                str(payload["display_name"])
                if payload.get("display_name") is not None
                else None
            ),
            resolved_root=(
                str(payload["resolved_root"])
                if payload.get("resolved_root") is not None
                else None
            ),
            license=dict(payload.get("license", {})),
            access=dict(payload.get("access", {})),
            download_policy=dict(payload.get("download_policy", {})),
            selected_part=(
                str(payload["selected_part"])
                if payload.get("selected_part") is not None
                else None
            ),
            selected_source_url=(
                str(payload["selected_source_url"])
                if payload.get("selected_source_url") is not None
                else None
            ),
            transport=(
                str(payload["transport"])
                if payload.get("transport") is not None
                else None
            ),
            downloaded_at=(
                str(payload["downloaded_at"])
                if payload.get("downloaded_at") is not None
                else None
            ),
            downloads_dir=(
                str(payload["downloads_dir"])
                if payload.get("downloads_dir") is not None
                else None
            ),
            source_root=(
                str(payload["source_root"])
                if payload.get("source_root") is not None
                else None
            ),
            force_download=bool(payload.get("force_download", False)),
            note=str(payload["note"]) if payload.get("note") is not None else None,
            checksums=[dict(item) for item in payload.get("checksums", [])],
            sources=[dict(item) for item in payload.get("sources", [])],
            schema_version=int(payload.get("schema_version", DATASET_RECEIPT_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "resolved_root": self.resolved_root,
            "license": dict(self.license),
            "access": dict(self.access),
            "download_policy": dict(self.download_policy),
            "selected_part": self.selected_part,
            "selected_source_url": self.selected_source_url,
            "transport": self.transport,
            "downloaded_at": self.downloaded_at,
            "downloads_dir": self.downloads_dir,
            "source_root": self.source_root,
            "force_download": self.force_download,
            "note": self.note,
            "checksums": [dict(item) for item in self.checksums],
            "sources": [dict(item) for item in self.sources],
        }


def receipt_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / DATASET_RECEIPT_NAME


def load_receipt(dataset_root: str | Path) -> DatasetDownloadReceipt | None:
    path = receipt_path(dataset_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return DatasetDownloadReceipt.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def write_dataset_receipt(
    dataset_root: str | Path,
    receipt: DatasetDownloadReceipt | dict[str, Any],
) -> Path:
    payload = receipt.to_dict() if isinstance(receipt, DatasetDownloadReceipt) else dict(receipt)
    path = receipt_path(dataset_root)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_receipt(dataset_id: str, payload: dict[str, Any], root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path.home() / ".cache" / "octosense" / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{dataset_id}.receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "DATASET_RECEIPT_NAME",
    "DATASET_RECEIPT_SCHEMA_VERSION",
    "DatasetDownloadReceipt",
    "load_receipt",
    "receipt_path",
    "write_dataset_receipt",
    "write_receipt",
]
