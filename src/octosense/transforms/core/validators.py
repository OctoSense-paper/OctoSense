"""Contract helpers owned by ``octosense.transforms.core``."""

from __future__ import annotations

import math
from collections.abc import Mapping

from octosense.core.contracts import AxisContract
from octosense.core.contracts.model import ModelInputContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor


def resolve_model_input_contract(model_or_contract: object) -> ModelInputContract:
    """Normalize a model object or a direct contract into ``ModelInputContract``."""

    if isinstance(model_or_contract, ModelInputContract):
        return model_or_contract

    provider = getattr(model_or_contract, "get_input_contract", None)
    if not callable(provider):
        raise TypeError(
            "AutoAlign expects a ModelInputContract or an object implementing "
            "get_input_contract()."
        )
    contract = provider()
    if not isinstance(contract, ModelInputContract):
        raise TypeError(
            f"{type(model_or_contract).__name__}.get_input_contract() must return "
            "ModelInputContract."
        )
    return contract


def schema_from_model_input_contract(
    contract: ModelInputContract,
    *,
    axis_metadata: Mapping[str, AxisMetadata] | None = None,
) -> AxisSchema:
    """Build an ``AxisSchema`` that mirrors a model entry contract."""

    metadata = dict(axis_metadata or {})
    for axis_name in contract.axes:
        metadata.setdefault(
            axis_name,
            AxisMetadata(
                name=axis_name,
                description=f"Model-entry axis '{axis_name}'",
            ),
        )
    return AxisSchema(axes=contract.axes, axis_metadata=metadata)


def validate_aligned_sample(
    sample: RadioTensor,
    contract: ModelInputContract,
) -> None:
    """Check that an aligned sample already matches the model-entry contract."""

    actual_axes = tuple(sample.axis_schema.axes)
    if actual_axes != contract.axes:
        raise DimensionError(
            "AutoAlign produced axes that do not match the model input contract.\n"
            f"Expected axes: {contract.axes}\n"
            f"Got axes: {actual_axes}\n"
            "Fix: update axis inference or pass an explicit axis_map to AutoAlign."
        )

    tensor = sample.as_tensor()
    if contract.dtype_kind == "real" and tensor.is_complex():
        raise DimensionError(
            "AutoAlign produced a complex tensor for a real-valued model contract.\n"
            f"Contract axes: {contract.axes}\n"
            f"Tensor dtype: {tensor.dtype}\n"
            "Fix: choose a real-valued projection such as value='magnitude' or value='real'."
        )
    if contract.dtype_kind == "complex" and not tensor.is_complex():
        raise DimensionError(
            "AutoAlign produced a real tensor for a complex-valued model contract.\n"
            f"Contract axes: {contract.axes}\n"
            f"Tensor dtype: {tensor.dtype}\n"
            "Fix: keep value='identity' so complex semantics survive until model entry."
        )

    for axis_name, expected_size in contract.fixed_sizes.items():
        axis_index = sample.axis_schema.index(axis_name)
        actual_size = int(tensor.shape[axis_index])
        if actual_size != expected_size:
            raise DimensionError(
                "AutoAlign produced a sample with the wrong fixed-size axis.\n"
                f"Axis: {axis_name}\n"
                f"Expected: {expected_size}\n"
                f"Got: {actual_size}\n"
                f"Contract axes: {contract.axes}"
            )


def validate_axis_contract_input(
    contract: AxisContract,
    schema: AxisSchema,
    tensor=None,
    metadata: object | None = None,
) -> None:
    """Run transform-owned runtime validation against an ``AxisContract``."""

    missing = [ax for ax in contract.required_axes if not schema.has_axis(ax)]
    if missing:
        raise DimensionError(
            f"Transform requires axes: {contract.required_axes}\n"
            f"Current axes: {list(schema.axes)}\n"
            f"Missing axes: {missing}\n"
            f"Fix: Ensure input tensor has axes {missing}. "
            f"Check your reader output or previous transforms."
        )

    present = [ax for ax in contract.forbidden_axes if schema.has_axis(ax)]
    if present:
        raise DimensionError(
            f"Transform forbids axes: {contract.forbidden_axes}\n"
            f"Current axes: {list(schema.axes)}\n"
            f"Present forbidden axes: {present}\n"
            f"Fix: Remove {present} axes before applying this transform. "
            f"Use appropriate transforms to eliminate these dimensions."
        )

    if contract.dtype_constraint and tensor is not None:
        is_complex = tensor.is_complex()
        expected_complex = contract.dtype_constraint == "complex"
        if is_complex != expected_complex:
            actual = "complex" if is_complex else "real"
            expected = contract.dtype_constraint
            fix_hint = (
                "Convert to complex using torch.view_as_complex()"
                if expected_complex
                else "Extract real/imag component using torch.real() or torch.abs()"
            )
            raise DimensionError(
                f"Transform requires dtype: {expected}\n"
                f"Current dtype: {tensor.dtype} ({actual})\n"
                f"Fix: {fix_hint}"
            )

    if metadata is not None and contract.required_metadata:
        for req in contract.required_metadata:
            if not req.required:
                continue

            if req.layer == "physical":
                value = getattr(metadata, req.field_name, None)
                if value is None or value == 0.0 or (
                    isinstance(value, float) and math.isnan(value)
                ):
                    raise DimensionError(
                        f"Transform requires metadata field: {req.field_name}\n"
                        f"Current value: {value}\n"
                        f"Layer: {req.layer} (Physical Constants)\n"
                        f"Fix: Ensure your reader sets metadata.{req.field_name}. "
                        f"This is typically set by the hardware reader."
                    )
            elif req.layer == "coords":
                coord = metadata.get_coord(req.field_name)
                if coord is None:
                    raise DimensionError(
                        f"Transform requires coordinate: {req.field_name}\n"
                        f"Current coords: {list(metadata.coords.keys())}\n"
                        f"Layer: {req.layer} (Coordinate Axes)\n"
                        f"Fix: Ensure metadata has coordinate axis '{req.field_name}'. "
                        f"This should be set by reader or previous transforms."
                    )
            elif req.layer == "provenance":
                value = getattr(metadata, req.field_name, None)
                if not value:
                    raise DimensionError(
                        f"Transform requires provenance field: {req.field_name}\n"
                        f"Current value: {value}\n"
                        f"Layer: {req.layer} (Processing Provenance)\n"
                        f"Fix: Ensure metadata.{req.field_name} is set. "
                        f"This tracks data origin and processing history."
                    )

    if metadata is not None and contract.required_coord_units:
        for axis_name, expected_unit in contract.required_coord_units.items():
            coord = metadata.get_coord(axis_name)
            if coord is None or coord.values is None:
                raise DimensionError(
                    f"Transform requires coordinate axis '{axis_name}' with explicit values.\n"
                    f"Current coords: {list(metadata.coords.keys())}\n"
                    f"Fix: Ensure metadata.set_coord('{axis_name}', values, unit) "
                    f"was populated by the reader or an earlier transform."
                )
            actual_unit = coord.unit or ""
            if actual_unit != expected_unit:
                raise DimensionError(
                    f"Transform requires coordinate axis '{axis_name}' in unit '{expected_unit}'.\n"
                    f"Current unit: '{actual_unit}'\n"
                    f"Fix: Ensure upstream semantics preserve the expected physical unit."
                )

    if metadata is not None and contract.required_extra_fields:
        for field_name in contract.required_extra_fields:
            if field_name not in metadata.extra:
                raise DimensionError(
                    f"Transform requires metadata.extra['{field_name}'].\n"
                    f"Current extra fields: {sorted(metadata.extra.keys())}\n"
                    f"Fix: Ensure the reader or earlier transforms persist this provenance."
                )


def compute_axis_contract_output_schema(
    contract: AxisContract,
    input_schema: AxisSchema,
) -> AxisSchema:
    """Apply transform-owned axis updates to derive the output schema."""

    new_axes = list(input_schema.axes)

    if contract.output_axes:
        for req_ax, out_ax in zip(contract.required_axes, contract.output_axes, strict=True):
            if req_ax in new_axes:
                idx = new_axes.index(req_ax)
                new_axes[idx] = out_ax

    for ax in contract.remove_axes:
        if ax in new_axes:
            new_axes.remove(ax)

    for ax in contract.add_axes:
        if ax not in new_axes:
            new_axes.append(ax)

    new_metadata = {ax: meta for ax, meta in input_schema.axis_metadata.items() if ax in new_axes}
    return AxisSchema(axes=tuple(new_axes), axis_metadata=new_metadata)


def infer_contract_family(contract: ModelInputContract) -> str:
    """Classify the contract into a transform-owned alignment family."""

    axes = tuple(contract.axes)
    if axes == ("channel", "height", "width"):
        return "image"
    if axes == ("time", "feature"):
        return "sequence"
    if len(axes) == 1:
        return "vector"
    return "generic"


__all__ = [
    "compute_axis_contract_output_schema",
    "infer_contract_family",
    "resolve_model_input_contract",
    "schema_from_model_input_contract",
    "validate_axis_contract_input",
    "validate_aligned_sample",
]
