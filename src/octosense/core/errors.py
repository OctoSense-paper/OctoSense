"""Cross-module canonical error types."""


class OctoSenseError(Exception):
    """Base error for canonical OctoSense runtime."""


class ContractError(OctoSenseError):
    """Raised when a declared semantic contract is violated."""


class DimensionError(ContractError, ValueError):
    """Raised when tensor dimensions don't match expected semantics."""

    def __init__(
        self,
        message: str,
        available_axes: list[str] | tuple[str, ...] | None = None,
        suggestion: str | None = None,
    ) -> None:
        full_message = message
        if available_axes:
            full_message += f"\n  Available axes: {list(available_axes)}"
        if suggestion:
            full_message += f"\n  Did you mean: '{suggestion}'?"
        super().__init__(full_message)


class MetadataError(ContractError, ValueError):
    """Raised when metadata is invalid or inconsistent."""


class SchemaValidationError(ContractError, ValueError):
    """Raised when a signal-like object fails structural schema validation."""


__all__ = [
    "ContractError",
    "DimensionError",
    "MetadataError",
    "OctoSenseError",
    "SchemaValidationError",
]
