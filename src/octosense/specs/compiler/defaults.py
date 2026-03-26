"""Defaulting helpers for benchmark specs."""

from __future__ import annotations

from octosense.specs.schemas.benchmark import (
    BENCHMARK_SPEC_KIND,
    DEFAULT_BENCHMARK_API_VERSION,
    BenchmarkSpec,
)
from octosense.specs.schemas.runtime import RuntimeSpec


def apply_defaults(spec: BenchmarkSpec) -> BenchmarkSpec:
    """Clone a benchmark spec and backfill omitted canonical defaults."""

    resolved = BenchmarkSpec.from_dict(spec.to_dict())
    if not resolved.api_version.strip():
        resolved.api_version = DEFAULT_BENCHMARK_API_VERSION
    if not resolved.kind.strip():
        resolved.kind = BENCHMARK_SPEC_KIND
    if resolved.runtime is None:
        resolved.runtime = RuntimeSpec()
    if not resolved.runtime.device.strip():
        resolved.runtime.device = "cpu"
    return resolved


__all__ = ["apply_defaults"]
