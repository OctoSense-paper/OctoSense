"""CLI for transform contract checks."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import sys
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Sequence

from octosense.transforms.core.base import BaseTransform
from octosense.transforms.ops.axis import AxisNormalize
from octosense.transforms.ops.einops import Rearrange, Reduce, Repeat

_TRANSFORM_FACTORIES: dict[str, Callable[[], BaseTransform]] = {
    "AxisNormalize": lambda: AxisNormalize(axis_name="time", method="l2"),
    "Rearrange": lambda: Rearrange("batch time subc -> batch (time subc)"),
    "Reduce": lambda: Reduce("batch time subc -> batch time", reduction="mean"),
    "Repeat": lambda: Repeat("batch time subc -> batch time subc rx", rx=1),
}


@lru_cache(maxsize=1)
def _bootstrap_transform_registry() -> tuple[str, ...]:
    import octosense.transforms as transforms_pkg

    import_errors: list[str] = []
    prefix = f"{transforms_pkg.__name__}."
    for module_info in pkgutil.walk_packages(transforms_pkg.__path__, prefix=prefix):
        module_name = module_info.name
        relative_name = module_name.removeprefix(prefix)
        parts = relative_name.split(".")
        if any(part.startswith("_") for part in parts):
            continue
        if parts[0] in {"backends", "core"}:
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive
            import_errors.append(f"{module_name}: {exc}")
    return tuple(import_errors)


def _has_required_init_args(transform_cls: type[BaseTransform]) -> bool:
    sig = inspect.signature(transform_cls)
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            return True
    return False


def _instantiate_transform(name: str, transform_cls: type[BaseTransform]) -> tuple[BaseTransform | None, str | None]:
    factory = _TRANSFORM_FACTORIES.get(name)
    if factory is not None:
        return factory(), None
    if _has_required_init_args(transform_cls):
        return None, f"Skipped (requires init args): {name}"
    try:
        return transform_cls(), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


def check_contracts() -> dict[str, Any]:
    from octosense.transforms.core.registry import (
        TRANSFORM_CLASS_REGISTRY,
        get_operator_info,
        list_operators,
    )

    results: dict[str, Any] = {"total_transforms": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": []}
    errors = results["errors"]
    assert isinstance(errors, list)

    for error in _bootstrap_transform_registry():
        results["failed"] = int(results["failed"]) + 1
        errors.append({"transform": "<bootstrap>", "error": error})

    operator_names = sorted(list_operators())
    results["total_transforms"] = len(operator_names)
    if not operator_names:
        results["failed"] = int(results["failed"]) + 1
        errors.append(
            {
                "transform": "<registry>",
                "error": "No transforms registered after bootstrap; contract discovery is broken.",
            }
        )
        return results

    for name in operator_names:
        obj = TRANSFORM_CLASS_REGISTRY.get(name)
        if not isinstance(obj, type):
            results["failed"] = int(results["failed"]) + 1
            errors.append({"transform": name, "error": "Registered transform class missing from class registry"})
            continue

        operator_info = get_operator_info(name)
        if operator_info.runtime_only:
            results["passed"] = int(results["passed"]) + 1
            continue

        if not issubclass(obj, BaseTransform) or obj is BaseTransform:
            results["failed"] = int(results["failed"]) + 1
            errors.append({"transform": name, "error": "Registered class is not a concrete BaseTransform"})
            continue
        instance, err = _instantiate_transform(name, obj)
        if instance is None:
            if err and err.startswith("Skipped"):
                results["skipped"] = int(results["skipped"]) + 1
                continue
            results["failed"] = int(results["failed"]) + 1
            errors.append({"transform": name, "error": err or "Failed to instantiate transform"})
            continue
        try:
            input_contract = instance.input_contract
            output_contract = instance.output_contract
            if input_contract and output_contract:
                results["passed"] = int(results["passed"]) + 1
            else:
                results["failed"] = int(results["failed"]) + 1
                errors.append({"transform": name, "error": "Missing input_contract or output_contract"})
        except Exception as exc:
            results["failed"] = int(results["failed"]) + 1
            errors.append({"transform": name, "error": str(exc)})
    return results


def check_schema_registry() -> dict[str, Any]:
    from octosense.io.semantics.registry import get_schema_version

    return {"schema_version": get_schema_version(), "errors": []}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate canonical transform contracts and IO schema registry")
    parser.add_argument(
        "--module",
        choices=["all", "transforms", "io"],
        default="all",
        help="Scope the contract check to one canonical owner module",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload: dict[str, Any] = {}
    contract_results: dict[str, Any] | None = None
    registry_results: dict[str, Any] | None = None

    if args.module in {"all", "transforms"}:
        contract_results = check_contracts()
        payload["contracts"] = contract_results
    if args.module in {"all", "io"}:
        registry_results = check_schema_registry()
        payload["registry"] = registry_results

    total_errors = 0
    if contract_results is not None:
        total_errors += int(contract_results["failed"])
    if registry_results is not None:
        total_errors += len(registry_results["errors"])

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if total_errors > 0 else 0

    print("=" * 60)
    print("OctoSense Contract Check")
    print("=" * 60)
    print()
    if contract_results is not None:
        print("Checking transform contracts...")
    if registry_results is not None:
        print("Checking IO schema registry...")
    if contract_results is not None:
        print("\nTransform Contracts:")
        print(f"  Total: {contract_results['total_transforms']}")
        print(f"  Passed: {contract_results['passed']}")
        print(f"  Skipped: {contract_results['skipped']}")
        print(f"  Failed: {contract_results['failed']}")
        errors = contract_results["errors"]
        assert isinstance(errors, list)
        if errors:
            print("\n  Errors:")
            for error in errors:
                print(f"    - {error['transform']}: {error['error']}")
    if registry_results is not None:
        print("\nSchema Registry:")
        print(f"  Schema version: {registry_results['schema_version']}")
        registry_errors = registry_results["errors"]
        assert isinstance(registry_errors, list)
        if registry_errors:
            print("\n  Errors:")
            for error in registry_errors:
                print(f"    - {error['family']}: {error['error']}")

    print()
    print("=" * 60)
    if total_errors > 0:
        print(f"FAILED: {total_errors} errors found")
        return 1
    print("PASSED: All checks successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
