"""Base transform class for RadioTensor processing."""

from abc import ABC, abstractmethod
from dataclasses import replace

import torch.nn as nn

from octosense.core.contracts import AxisContract
from octosense.core.errors import DimensionError
from octosense.io.tensor import RadioTensor, is_safety_mode
from octosense.transforms.core.validators import (
    compute_axis_contract_output_schema,
    validate_axis_contract_input,
)


class BaseTransform(nn.Module, ABC):
    """Base class for all RadioTensor transforms.

    All transforms are nn.Module subclasses and must define:
    - input_contract: Required input axes and metadata
    - output_contract: Expected output axes
    - forward(): Transform logic

    Transforms are:
    - GPU-native (run on GPU only)
    - Differentiable (support autograd)
    - Metadata-aware (preserve/update metadata correctly)
    """

    @property
    @abstractmethod
    def input_contract(self) -> AxisContract:
        """Define required input axes and metadata.

        Returns:
            AxisContract specifying required_axes and required_metadata

        Example:
            >>> return AxisContract(
            ...     required_axes=["time"],
            ...     dtype_constraint="complex",
            ... )
        """
        pass

    @property
    @abstractmethod
    def output_contract(self) -> AxisContract:
        """Define output axes after transformation.

        Returns:
            AxisContract specifying output axes

        Example:
            >>> return AxisContract(
            ...     output_axes=["freq"],  # Replaces input axis
            ... )
        """
        pass

    @abstractmethod
    def forward(self, x: RadioTensor) -> RadioTensor:
        """Transform RadioTensor.

        Args:
            x: Input RadioTensor

        Returns:
            Transformed RadioTensor with updated metadata

        Raises:
            DimensionError: If input doesn't satisfy input_contract
            MetadataError: If required metadata missing
        """
        pass

    def _validate_input(self, x: RadioTensor) -> None:
        """Validate input RadioTensor against input_contract.

        Args:
            x: Input RadioTensor to validate

        Raises:
            DimensionError: If validation fails
        """
        contract = self.input_contract
        if is_safety_mode():
            # Full checks in safety mode: axes + dtype + metadata.
            validate_axis_contract_input(
                contract,
                x.axis_schema,
                x.as_tensor(),
                x.metadata,
            )
            return

        # Performance mode fast path: keep lightweight axis checks per op.
        # Full boundary validation is expected at reader/model or pipeline boundaries.
        validate_axis_contract_input(contract, x.axis_schema, None, None)

    def _validate_output(self, input_rt: RadioTensor, out: RadioTensor) -> None:
        """Validate transform output against output contract and axis updates."""
        contract = self.output_contract
        if is_safety_mode():
            expected_schema = self.expected_output_schema(input_rt)
            if expected_schema.axes != out.axis_schema.axes:
                raise DimensionError(
                    f"{self.__class__.__name__} produced unexpected axes.\n"
                    f"Expected axes: {expected_schema.axes}\n"
                    f"Got axes: {out.axis_schema.axes}\n"
                    f"Fix: Update the transform output schema or its output_contract."
                )
            output_contract = replace(
                contract,
                required_axes=[],
                forbidden_axes=[],
                output_axes=None,
                add_axes=[],
                remove_axes=[],
            )
            validate_axis_contract_input(
                output_contract,
                out.axis_schema,
                out.as_tensor(),
                out.metadata,
            )

    def expected_output_schema(self, input_rt: RadioTensor):
        """Infer the output schema from the input schema and output contract."""
        contract = self.output_contract
        effective_contract = contract
        if contract.output_axes and not contract.required_axes:
            inferred_replacements = [
                axis_name
                for axis_name in self.input_contract.required_axes
                if axis_name in input_rt.axis_schema.axes
            ]
            if len(inferred_replacements) == len(contract.output_axes):
                effective_contract = replace(
                    contract,
                    required_axes=inferred_replacements,
                )
        return compute_axis_contract_output_schema(effective_contract, input_rt.axis_schema)

    @property
    def requires(self) -> dict[str, object]:
        """Explicit semantic preconditions for this operator."""
        return self.input_contract.to_requires_dict()

    @property
    def updates(self) -> dict[str, object]:
        """Explicit semantic postconditions for this operator."""
        return self.output_contract.to_updates_dict(
            source_axes=self.input_contract.required_axes,
        )

    def semantic_contract(self) -> dict[str, object]:
        """Return the full semantic contract for inspection."""
        return {
            "requires": self.requires,
            "updates": self.updates,
        }

    def on_keys(
        self,
        *keys: str,
        output_keys: tuple[str, ...] | None = None,
    ) -> nn.Module:
        """Adapt a RadioTensor transform into a dictionary-based operator."""
        from octosense.transforms.core.keyed import KeyedTransform

        return KeyedTransform(
            self,
            keys=tuple(keys),
            output_keys=output_keys,
        )

    def __call__(self, *args, **kwargs):
        """Run the transform and enforce output contracts."""
        input_rt = args[0] if args and isinstance(args[0], RadioTensor) else None
        out = super().__call__(*args, **kwargs)
        if not isinstance(out, RadioTensor) or input_rt is None:
            return out
        self._validate_output(input_rt, out)
        return out
