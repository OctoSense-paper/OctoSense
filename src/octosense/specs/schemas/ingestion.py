"""Ingestion schemas owned by ``octosense.specs``."""

from dataclasses import dataclass


@dataclass(slots=True)
class IngestionSpec:
    """Optional custom raw-data import description."""

    reader_id: str = ""
    source_root: str | None = None
    file_glob: str | None = None
    label_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "reader_id": self.reader_id,
            "source_root": self.source_root,
            "file_glob": self.file_glob,
            "label_source": self.label_source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "IngestionSpec":
        if payload is None:
            return cls()
        return cls(
            reader_id=str(payload.get("reader_id", "") or ""),
            source_root=_optional_str(payload.get("source_root")),
            file_glob=_optional_str(payload.get("file_glob")),
            label_source=_optional_str(payload.get("label_source")),
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = ["IngestionSpec"]
