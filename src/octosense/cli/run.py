"""Canonical CLI for spec-native OctoSense BenchmarkSpec execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from octosense.pipelines.execution.runner import execute_target
from octosense.specs.schemas.benchmark import BenchmarkSpec
from octosense.specs.serde.json_io import load_json
from octosense.specs.serde.yaml_io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a BenchmarkSpec through the OctoSense pipelines execution owner"
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Canonical BenchmarkSpec path or inline payload",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["train", "evaluate"], default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    return parser


def _coerce_cli_spec_argument(spec: str) -> str | Path:
    candidate = Path(spec)
    return candidate if candidate.exists() else spec


def _read_cli_spec_payload(spec: str | Path) -> str:
    if isinstance(spec, Path):
        return spec.read_text(encoding="utf-8")
    return str(spec)


def _load_cli_target(*, spec: str | None) -> BenchmarkSpec:
    if spec is None or not str(spec).strip():
        raise ValueError("Pass one canonical execution target: <BenchmarkSpec>.")
    candidate = _coerce_cli_spec_argument(spec)
    try:
        if isinstance(candidate, Path) and candidate.suffix.lower() == ".json":
            document = load_json(candidate)
        else:
            payload = _read_cli_spec_payload(candidate)
            stripped = payload.lstrip()
            document = load_json(payload) if stripped.startswith("{") else load_yaml(payload)
        if not isinstance(document, BenchmarkSpec):
            raise ValueError(f"Expected BenchmarkSpec input, got {type(document).__name__}.")
        return document
    except Exception as exc:  # pragma: no cover - CLI error path
        raise ValueError(f"Unable to parse canonical BenchmarkSpec input: {exc}") from exc


def _run_loaded_pipeline(
    target: BenchmarkSpec,
    *,
    output_root: str | None,
    run_name: str | None,
    device: str | None,
    seed: int | None,
    mode: str | None,
) -> dict[str, object]:
    return dict(
        execute_target(
            target,
            output_root=output_root,
            run_name=run_name,
            device=device,
            seed=seed,
            mode=mode,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        target = _load_cli_target(spec=args.spec)
    except ValueError as exc:
        parser.error(str(exc))

    result = _run_loaded_pipeline(
        target,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
