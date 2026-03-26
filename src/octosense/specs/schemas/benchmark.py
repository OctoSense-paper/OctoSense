"""Benchmark schema owned by ``octosense.specs``."""

from __future__ import annotations

from dataclasses import dataclass, field

from octosense.specs.schemas.dataset import DatasetSpec
from octosense.specs.schemas.ingestion import IngestionSpec
from octosense.specs.schemas.model import ModelSpec
from octosense.specs.schemas.runtime import RuntimeSpec
from octosense.specs.schemas.task import TaskSpec
from octosense.specs.schemas.transform import TransformSpec

BENCHMARK_SPEC_KIND = "BenchmarkSpec"
DEFAULT_BENCHMARK_API_VERSION = "octosense.benchmarks/v2"


@dataclass(slots=True)
class BenchmarkSpec:
    """Declarative benchmark request."""

    api_version: str = DEFAULT_BENCHMARK_API_VERSION
    kind: str = BENCHMARK_SPEC_KIND
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    task: TaskSpec = field(default_factory=TaskSpec)
    transform: TransformSpec = field(default_factory=TransformSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    ingestion: IngestionSpec | None = None
    protocol: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "api_version": self.api_version,
            "kind": self.kind,
            "dataset": self.dataset.to_dict(),
            "task": self.task.to_dict(),
            "transform": self.transform.to_dict(),
            "model": self.model.to_dict(),
            "runtime": self.runtime.to_dict(),
            "ingestion": None if self.ingestion is None else self.ingestion.to_dict(),
            "protocol": dict(self.protocol),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "BenchmarkSpec":
        if payload is None:
            return cls()
        ingestion_payload = payload.get("ingestion")
        if ingestion_payload is not None and not isinstance(ingestion_payload, dict):
            raise TypeError("BenchmarkSpec.ingestion must be a mapping when provided")
        return cls(
            api_version=str(
                payload.get("api_version", DEFAULT_BENCHMARK_API_VERSION)
                or DEFAULT_BENCHMARK_API_VERSION
            ),
            kind=str(payload.get("kind", BENCHMARK_SPEC_KIND) or BENCHMARK_SPEC_KIND),
            dataset=DatasetSpec.from_dict(_mapping(payload.get("dataset"), "dataset")),
            task=TaskSpec.from_dict(_mapping(payload.get("task"), "task")),
            transform=TransformSpec.from_dict(_mapping(payload.get("transform"), "transform")),
            model=ModelSpec.from_dict(_mapping(payload.get("model"), "model")),
            runtime=RuntimeSpec.from_dict(_mapping(payload.get("runtime"), "runtime")),
            ingestion=None if ingestion_payload is None else IngestionSpec.from_dict(ingestion_payload),
            protocol=_plain_mapping(payload.get("protocol"), "protocol"),
        )


def _mapping(value: object, field_name: str) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"BenchmarkSpec.{field_name} must be a mapping")
    return dict(value)


def _plain_mapping(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"BenchmarkSpec.{field_name} must be a mapping")
    return dict(value)


__all__ = [
    "BENCHMARK_SPEC_KIND",
    "DEFAULT_BENCHMARK_API_VERSION",
    "BenchmarkSpec",
]
