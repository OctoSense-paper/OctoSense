"""CLI for exporting canonical spec payloads."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import octosense as octo
from octosense.specs.compiler.defaults import apply_defaults
from octosense.specs.compiler.freezer import freeze_spec
from octosense.specs.schemas.benchmark import BenchmarkSpec
from octosense.specs.serde.canonical import canonical_dump
from octosense.specs.serde.json_io import load_json
from octosense.specs.serde.yaml_io import load_yaml


def _freeze_and_serialize_benchmark_spec(spec: BenchmarkSpec) -> str:
    return freeze_spec(spec).canonical_payload


def _serialize_canonical_benchmark_spec(spec: BenchmarkSpec) -> str:
    return canonical_dump(apply_defaults(spec))


def export_spec(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.suffix.lower() == ".json":
        document = load_json(candidate)
    else:
        document = load_yaml(candidate)
    if not isinstance(document, BenchmarkSpec):
        raise TypeError("export_spec only accepts BenchmarkSpec documents")
    return _freeze_and_serialize_benchmark_spec(document)


def export_spec_from_recipe(
    recipe_id: str,
    *,
    modalities: Sequence[str],
    variant: str | None = None,
    path: str | Path | None = None,
) -> str:
    pipeline = octo.pipelines.load(
        recipe_id=recipe_id,
        modalities=modalities,
        variant=variant,
        path=path,
    )
    benchmark_spec = getattr(pipeline, "benchmark_spec", None)
    if not isinstance(benchmark_spec, BenchmarkSpec):
        raise ValueError("Canonical recipe pipeline did not expose benchmark_spec after load().")
    return _serialize_canonical_benchmark_spec(benchmark_spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a canonical benchmark spec")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="Path to a YAML or JSON BenchmarkSpec")
    source.add_argument("--recipe", help="Canonical recipe id like <dataset_id>/<model_id>@<task_id>")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Canonical modalities required for --recipe",
    )
    parser.add_argument("--variant", default=None, help="Optional dataset variant for --recipe")
    parser.add_argument("--path", default=None, help="Optional dataset root override for --recipe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.spec is not None:
        print(export_spec(args.spec))
        return 0
    if not args.modalities:
        parser.error("--recipe requires --modalities")
    try:
        print(
            export_spec_from_recipe(
                args.recipe,
                modalities=args.modalities,
                variant=args.variant,
                path=args.path,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
