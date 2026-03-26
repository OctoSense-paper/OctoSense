"""Canonical transform operator registry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pkgutil import walk_packages
from typing import Any, TypeVar

_T = TypeVar("_T", bound=type)


@dataclass(frozen=True)
class OperatorInfo:
    """Registration record for a transform operator."""

    name: str
    required_axes: tuple[str, ...] = ()
    required_meta: tuple[str, ...] = ()
    description: str = ""
    runtime_only: bool = False


TRANSFORM_REGISTRY: dict[str, OperatorInfo] = {}
TRANSFORM_CLASS_REGISTRY: dict[str, type[object]] = {}
_DISCOVERY_PACKAGE_NAMES = (
    "octosense.transforms.ops",
    "octosense.transforms.adapters",
    "octosense.transforms.modalities",
)
_BOOTSTRAP_COMPLETE = False
_BOOTSTRAP_IN_PROGRESS = False


def ensure_operator_registry_bootstrapped() -> None:
    """Import canonical operator owner modules exactly once for cold-start discovery."""

    global _BOOTSTRAP_COMPLETE, _BOOTSTRAP_IN_PROGRESS

    if _BOOTSTRAP_COMPLETE or _BOOTSTRAP_IN_PROGRESS:
        return

    _BOOTSTRAP_IN_PROGRESS = True
    try:
        for package_name in _DISCOVERY_PACKAGE_NAMES:
            package = import_module(package_name)
            package_paths = getattr(package, "__path__", None)
            if package_paths is None:
                continue
            for module_info in walk_packages(package_paths, prefix=f"{package_name}."):
                if module_info.name.rsplit(".", 1)[-1].startswith("_"):
                    continue
                import_module(module_info.name)
        _BOOTSTRAP_COMPLETE = True
    finally:
        _BOOTSTRAP_IN_PROGRESS = False


def register_operator(
    name: str,
    *,
    required_axes: list[str] | tuple[str, ...] = (),
    required_meta: list[str] | tuple[str, ...] = (),
    description: str = "",
) -> None:
    """Register an operator in the transforms-local registry."""
    if name in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Operator '{name}' already registered. "
            "Choose a different name or unregister the existing operator first."
        )
    TRANSFORM_REGISTRY[name] = OperatorInfo(
        name=name,
        required_axes=tuple(required_axes),
        required_meta=tuple(required_meta),
        description=description,
    )


def get_operator_info(name: str) -> OperatorInfo:
    """Get operator registration info by name."""
    ensure_operator_registry_bootstrapped()
    if name not in TRANSFORM_REGISTRY:
        raise KeyError(
            f"Operator '{name}' not registered. Available: {list_operators()}"
        )
    return TRANSFORM_REGISTRY[name]


def list_operators() -> list[str]:
    """List all registered operator names."""
    ensure_operator_registry_bootstrapped()
    return list(TRANSFORM_REGISTRY.keys())


def get_operator_class(name: str) -> type[object]:
    """Resolve a registered operator class by name."""
    ensure_operator_registry_bootstrapped()
    try:
        return TRANSFORM_CLASS_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Operator class '{name}' not registered. Available: {list_operators()}"
        ) from exc


def registered_operator(
    *,
    required_axes: list[str] | tuple[str, ...] = (),
    required_meta: list[str] | tuple[str, ...] = (),
    description: str = "",
) -> Any:
    """Class decorator to register a transform in the transforms-local registry."""

    def decorator(cls: _T) -> _T:
        register_operator(
            cls.__name__,
            required_axes=required_axes,
            required_meta=required_meta,
            description=description or cls.__doc__ or "",
        )
        TRANSFORM_CLASS_REGISTRY[cls.__name__] = cls
        return cls

    return decorator


def registered_runtime_operator(*, description: str = "") -> Any:
    """Class decorator for runtime-serializable operators without AxisContract surface."""

    def decorator(cls: _T) -> _T:
        TRANSFORM_REGISTRY[cls.__name__] = OperatorInfo(
            name=cls.__name__,
            description=description or cls.__doc__ or "",
            runtime_only=True,
        )
        TRANSFORM_CLASS_REGISTRY[cls.__name__] = cls
        return cls

    return decorator

__all__ = [
    "OperatorInfo",
    "TRANSFORM_REGISTRY",
    "TRANSFORM_CLASS_REGISTRY",
    "ensure_operator_registry_bootstrapped",
    "get_operator_class",
    "get_operator_info",
    "list_operators",
    "register_operator",
    "registered_operator",
    "registered_runtime_operator",
]
