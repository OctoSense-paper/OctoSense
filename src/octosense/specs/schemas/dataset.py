"""Dataset schema owned by ``octosense.specs``."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DatasetSpec:
    """Declarative dataset selection for a benchmark run."""

    dataset_id: str = ""
    modalities: list[str] = field(default_factory=list)
    variant: str | None = None
    split_scheme: str | None = None
    dataset_root: str | None = None
    input_selection: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "modalities": list(self.modalities),
            "variant": self.variant,
            "split_scheme": self.split_scheme,
            "dataset_root": self.dataset_root,
            "input_selection": (
                None if self.input_selection is None else dict(self.input_selection)
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "DatasetSpec":
        if payload is None:
            return cls()
        modalities = payload.get("modalities", [])
        if isinstance(modalities, str):
            modalities = [modalities]
        if not isinstance(modalities, list):
            raise TypeError("DatasetSpec.modalities must be a list of modality ids")
        input_selection = payload.get("input_selection")
        if input_selection is not None and not isinstance(input_selection, dict):
            raise TypeError("DatasetSpec.input_selection must be a mapping when provided")
        return cls(
            dataset_id=str(payload.get("dataset_id", "") or ""),
            modalities=[str(item) for item in modalities],
            variant=_optional_str(payload.get("variant")),
            split_scheme=_optional_str(payload.get("split_scheme")),
            dataset_root=_optional_str(payload.get("dataset_root")),
            input_selection=None if input_selection is None else dict(input_selection),
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = ["DatasetSpec"]
