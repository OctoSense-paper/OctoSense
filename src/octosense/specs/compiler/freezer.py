"""Freeze benchmark specs into canonical serialization payloads."""

from __future__ import annotations

from dataclasses import dataclass

from octosense.specs.compiler.defaults import apply_defaults
from octosense.specs.compiler.resolver import ReferenceResolverContext, resolve_references
from octosense.specs.compiler.semantic_validator import validate_semantics
from octosense.specs.schemas.benchmark import BenchmarkSpec
from octosense.specs.serde.canonical import canonical_dump, spec_digest


@dataclass(slots=True, frozen=True)
class FrozenBenchmarkSpec:
    """Frozen benchmark spec plus its canonical serialization metadata."""

    spec: BenchmarkSpec
    canonical_payload: str
    spec_digest: str


def freeze_spec(
    spec: BenchmarkSpec,
    *,
    resolver_context: ReferenceResolverContext | None = None,
) -> FrozenBenchmarkSpec:
    """Apply defaults, resolve references, validate semantics, and seal the digest."""

    resolved = apply_defaults(spec)
    resolved = resolve_references(resolved, context=resolver_context)
    validate_semantics(resolved)
    payload = canonical_dump(resolved)
    return FrozenBenchmarkSpec(
        spec=resolved,
        canonical_payload=payload,
        spec_digest=spec_digest(resolved),
    )


__all__ = ["FrozenBenchmarkSpec", "freeze_spec"]
