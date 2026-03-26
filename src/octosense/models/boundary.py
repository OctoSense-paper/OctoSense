"""Internal model-boundary contracts and validators behind load dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn

from octosense.core.contracts.model import ModelInputContract, ModelOutputContract
from octosense.core.errors import DimensionError


@dataclass(frozen=True)
class ModelBoundarySpec:
    """Resolved boundary metadata for one model family plus entry overrides."""

    model_id: str
    input_contract: ModelInputContract
    output_contract: ModelOutputContract
    required_entry_overrides: tuple[str, ...] = ()
    optional_entry_overrides: tuple[str, ...] = ()
    unresolved_entry_overrides: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_contract"] = self.input_contract.to_dict()
        payload["output_contract"] = self.output_contract.to_dict()
        return payload


def _normalize_optional_int_tuple(
    value: object,
    *,
    key: str,
    expected_len: int | None = None,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{key} must be a sequence of ints, got {type(value)!r}")
    normalized = tuple(int(item) for item in value)
    if expected_len is not None and len(normalized) != expected_len:
        raise ValueError(f"{key} must contain exactly {expected_len} ints, got {normalized}")
    return normalized


def _build_resnet18_boundary(entry_overrides: Mapping[str, object]) -> ModelBoundarySpec:
    fixed_sizes: dict[str, int] = {}
    if entry_overrides.get("in_channels") is not None:
        fixed_sizes["channel"] = int(entry_overrides["in_channels"])
    return ModelBoundarySpec(
        model_id="resnet18",
        input_contract=ModelInputContract(
            axes=("channel", "height", "width"),
            dtype_kind="real",
            fixed_sizes=fixed_sizes,
            layout="BCHW",
            channel_semantics=(
                "Feature-map channels are stacked along the channel axis after "
                "an explicit model-boundary bridge."
            ),
            canonicalization_assumption=(
                "Input must already be converted into a real-valued "
                "channel-first representation before model entry."
            ),
            notes="Height and width remain task-specific; only the channel count is fixed when known.",
        ),
        output_contract=ModelOutputContract(
            kind="classification_logits",
            notes="Classification models emit logits shaped (batch, class).",
        ),
        required_entry_overrides=("num_classes",),
        optional_entry_overrides=("in_channels",),
        unresolved_entry_overrides=(
            ()
            if entry_overrides.get("in_channels") is not None
            else ("in_channels",)
        ),
    )


def _build_rfnet_boundary(entry_overrides: Mapping[str, object]) -> ModelBoundarySpec:
    fixed_sizes: dict[str, int] = {}
    input_shape = entry_overrides.get("input_shape")
    if input_shape is not None:
        time_steps, feature_dim = _normalize_optional_int_tuple(
            input_shape,
            key="input_shape",
            expected_len=2,
        ) or (0, 0)
        fixed_sizes = {"time": int(time_steps), "feature": int(feature_dim)}
    return ModelBoundarySpec(
        model_id="rfnet",
        input_contract=ModelInputContract(
            axes=("time", "feature"),
            dtype_kind="real",
            fixed_sizes=fixed_sizes,
            layout="BTF",
            channel_semantics="Sequence-style feature tensors arranged as time-major trajectories.",
            canonicalization_assumption=(
                "RFNet expects a real-valued (time, feature) tensor at model entry; "
                "time-frequency fusion happens inside the model."
            ),
            notes="Time and feature sizes become fixed once input_shape is resolved.",
        ),
        output_contract=ModelOutputContract(
            kind="classification_logits",
            notes="RFNet emits per-class logits after time-frequency fusion.",
        ),
        required_entry_overrides=("num_classes",),
        optional_entry_overrides=("input_shape", "hidden_dim", "dropout"),
        unresolved_entry_overrides=(
            ()
            if entry_overrides.get("input_shape") is not None
            else ("input_shape",)
        ),
    )


def _build_mlp_boundary(entry_overrides: Mapping[str, object]) -> ModelBoundarySpec:
    adaptive_shape = entry_overrides.get("adaptive_shape")
    if adaptive_shape is not None:
        adaptive_shape = _normalize_optional_int_tuple(adaptive_shape, key="adaptive_shape")
    input_dim = entry_overrides.get("input_dim")
    fixed_sizes: dict[str, int] = {}
    if adaptive_shape is None and input_dim is not None:
        fixed_sizes["feature"] = int(input_dim)
    if adaptive_shape is None:
        axes = ("feature",)
    else:
        axes = tuple(f"feature_dim_{index + 1}" for index in range(len(adaptive_shape)))
    return ModelBoundarySpec(
        model_id="mlp",
        input_contract=ModelInputContract(
            axes=axes,
            dtype_kind="real",
            fixed_sizes=fixed_sizes,
            layout="B*",
            channel_semantics=(
                "Generic real-valued feature tensors flattened after optional adaptive pooling."
            ),
            canonicalization_assumption=(
                "Input should already be a real-valued tensor representation suitable for "
                "direct flattening at model entry."
            ),
            notes=(
                "When adaptive_shape is provided, the contract fixes feature rank but leaves "
                "per-axis sizes dynamic because pooling canonicalizes them inside the model."
            ),
        ),
        output_contract=ModelOutputContract(
            kind="classification_logits",
            notes="MLPClassifier emits per-class logits after flattening pooled features.",
        ),
        required_entry_overrides=("num_classes",),
        optional_entry_overrides=("input_dim", "hidden_dims", "dropout", "adaptive_shape"),
    )


def _build_cnn1d_boundary(entry_overrides: Mapping[str, object]) -> ModelBoundarySpec:
    fixed_sizes: dict[str, int] = {}
    if entry_overrides.get("in_channels") is not None:
        fixed_sizes["feature"] = int(entry_overrides["in_channels"])
    return ModelBoundarySpec(
        model_id="cnn1d",
        input_contract=ModelInputContract(
            axes=("time", "feature"),
            dtype_kind="real",
            fixed_sizes=fixed_sizes,
            layout="BTF",
            channel_semantics="Feature axis for one canonicalized per-step vector.",
            canonicalization_assumption=(
                "Sequence models must receive real-valued features laid out as "
                "(time, feature) before batch stacking."
            ),
            notes=(
                "SimpleCNN1D consumes sequence-style feature representations and "
                "internally transposes to Conv1d layout."
            ),
        ),
        output_contract=ModelOutputContract(
            kind="classification_logits",
            notes="SimpleCNN1D emits per-class logits.",
        ),
        required_entry_overrides=("num_classes",),
        optional_entry_overrides=("in_channels", "hidden_dim", "num_layers", "kernel_size", "dropout"),
        unresolved_entry_overrides=(
            ()
            if entry_overrides.get("in_channels") is not None
            else ("in_channels",)
        ),
    )


def _build_resnet18_pose_boundary(entry_overrides: Mapping[str, object]) -> ModelBoundarySpec:
    fixed_sizes: dict[str, int] = {}
    if entry_overrides.get("in_channels") is not None:
        fixed_sizes["channel"] = int(entry_overrides["in_channels"])
    predict_bbox = entry_overrides.get("predict_bbox")
    if predict_bbox is None:
        required_keys: tuple[str, ...] = ()
    else:
        required_keys = ("joints", "bbox") if bool(predict_bbox) else ("joints",)
    unresolved_entry_overrides = tuple(
        key
        for key in ("in_channels", "num_joints", "joint_dims", "predict_bbox")
        if entry_overrides.get(key) is None
    )
    return ModelBoundarySpec(
        model_id="resnet18_pose",
        input_contract=ModelInputContract(
            axes=("channel", "height", "width"),
            dtype_kind="real",
            fixed_sizes=fixed_sizes,
            layout="BCHW",
            channel_semantics=(
                "Feature-map channels are stacked along the channel axis before structured "
                "prediction; target structure remains outside the input tensor."
            ),
            canonicalization_assumption=(
                "The pose model expects a real-valued, channel-first feature map "
                "representation before model entry."
            ),
            notes=(
                "Pose targets are schema-defined structured dictionaries. "
                "Output keys are finalized once the caller provides the target schema."
            ),
        ),
        output_contract=ModelOutputContract(
            kind="structured_dict",
            required_keys=required_keys,
            notes="Pose models emit structured prediction dictionaries keyed by target name.",
        ),
        required_entry_overrides=("num_joints", "joint_dims", "predict_bbox"),
        optional_entry_overrides=("in_channels",),
        unresolved_entry_overrides=unresolved_entry_overrides,
    )


BoundaryBuilder = Callable[[Mapping[str, object]], ModelBoundarySpec]

BOUNDARY_BUILDERS: dict[str, BoundaryBuilder] = {
    "cnn1d": _build_cnn1d_boundary,
    "mlp": _build_mlp_boundary,
    "resnet18": _build_resnet18_boundary,
    "resnet18_pose": _build_resnet18_pose_boundary,
    "rfnet": _build_rfnet_boundary,
}


class BaseContractModel(nn.Module):
    """Base class for custom models that own explicit contracts at initialization time."""

    def __init__(
        self,
        *,
        input_contract: ModelInputContract,
        output_contract: ModelOutputContract,
    ) -> None:
        super().__init__()
        if not isinstance(input_contract, ModelInputContract):
            raise TypeError(
                "BaseContractModel expects input_contract to be a ModelInputContract, "
                f"got {type(input_contract)!r}"
            )
        if not isinstance(output_contract, ModelOutputContract):
            raise TypeError(
                "BaseContractModel expects output_contract to be a ModelOutputContract, "
                f"got {type(output_contract)!r}"
            )
        self._input_contract = input_contract
        self._output_contract = output_contract

    @property
    def input_contract(self) -> ModelInputContract:
        return self._input_contract

    @property
    def output_contract(self) -> ModelOutputContract:
        return self._output_contract

    def get_input_contract(self) -> ModelInputContract:
        return self._input_contract

    def get_output_contract(self) -> ModelOutputContract:
        return self._output_contract


class BoundaryBackedModel(BaseContractModel):
    """Base class for registered model families resolved through canonical boundaries."""

    boundary_model_id: str

    def __init__(
        self,
        *,
        boundary_model_id: str | None = None,
        entry_overrides: Mapping[str, object] | None = None,
    ) -> None:
        resolved_model_id = str(boundary_model_id or getattr(self, "boundary_model_id", "")).strip()
        if not resolved_model_id:
            raise ValueError(
                f"{self.__class__.__name__} must provide boundary_model_id during initialization."
            )
        resolved_entry_overrides = (
            {}
            if entry_overrides is None
            else {str(key): value for key, value in entry_overrides.items()}
        )
        boundary = describe_model_boundary(
            resolved_model_id,
            entry_overrides=resolved_entry_overrides,
        )
        super().__init__(
            input_contract=boundary.input_contract,
            output_contract=boundary.output_contract,
        )
        self.boundary_model_id = resolved_model_id
        self._boundary_entry_overrides = resolved_entry_overrides
        self._boundary_spec = boundary

    @property
    def boundary_spec(self) -> ModelBoundarySpec:
        return self._boundary_spec

    def boundary_entry_overrides(self) -> Mapping[str, object]:
        return dict(self._boundary_entry_overrides)


def _resolve_model_boundary(
    model_id: str,
    *,
    entry_overrides: Mapping[str, object] | None = None,
    **kwargs: object,
) -> tuple[dict[str, object], ModelBoundarySpec]:
    from octosense.models.registry import (
        _get_model_registration,
        _normalize_entry_overrides,
    )

    normalized = _normalize_entry_overrides(
        model_id,
        entry_overrides=entry_overrides,
        **kwargs,
    )
    registration = _get_model_registration(model_id)
    try:
        builder = BOUNDARY_BUILDERS[registration.boundary_id]
    except KeyError as exc:
        raise ValueError(
            f"Model '{registration.model_id}' declares unknown boundary_id "
            f"'{registration.boundary_id}'."
        ) from exc
    return normalized, builder(normalized)


def describe_model_boundary(
    model_id: str,
    *,
    entry_overrides: Mapping[str, object] | None = None,
    **kwargs: object,
) -> ModelBoundarySpec:
    _, boundary = _resolve_model_boundary(
        model_id,
        entry_overrides=entry_overrides,
        **kwargs,
    )
    return boundary


def can_materialize_model(
    model_id: str,
    *,
    entry_overrides: Mapping[str, object] | None = None,
    **kwargs: object,
) -> bool:
    boundary = describe_model_boundary(
        model_id,
        entry_overrides=entry_overrides,
        **kwargs,
    )
    return not boundary.unresolved_entry_overrides


def resolve_materialized_entry_overrides(
    model_id: str,
    *,
    entry_overrides: Mapping[str, object] | None = None,
    **kwargs: object,
) -> dict[str, object]:
    normalized, boundary = _resolve_model_boundary(
        model_id,
        entry_overrides=entry_overrides,
        **kwargs,
    )
    if boundary.unresolved_entry_overrides:
        missing_keys = ", ".join(boundary.unresolved_entry_overrides)
        raise ValueError(
            f"Cannot materialize model '{boundary.model_id}' without entry overrides: {missing_keys}"
        )
    return normalized


def infer_entry_overrides_from_sample(
    model_id: str,
    sample_tensor: torch.Tensor | None,
    *,
    entry_overrides: Mapping[str, object] | None = None,
    num_classes: int | None = None,
) -> dict[str, object]:
    from octosense.models.registry import _get_model_registration, _normalize_entry_overrides

    resolved = _normalize_entry_overrides(model_id, entry_overrides=entry_overrides)
    normalized_model_id = _get_model_registration(model_id).model_id

    if num_classes is not None and "num_classes" not in resolved:
        resolved["num_classes"] = int(num_classes)

    if sample_tensor is None:
        return _normalize_entry_overrides(normalized_model_id, entry_overrides=resolved)

    if normalized_model_id == "resnet18":
        if sample_tensor.ndim != 3:
            raise ValueError(
                f"resnet18 expects one sample shaped (channel, height, width), got {tuple(sample_tensor.shape)}"
            )
        resolved.setdefault("in_channels", int(sample_tensor.shape[0]))
    elif normalized_model_id == "rfnet":
        if sample_tensor.ndim != 2:
            raise ValueError(
                f"rfnet expects one sample shaped (time, feature), got {tuple(sample_tensor.shape)}"
            )
        resolved.setdefault(
            "input_shape",
            (int(sample_tensor.shape[0]), int(sample_tensor.shape[1])),
        )
    elif normalized_model_id == "mlp":
        if sample_tensor.ndim < 1:
            raise ValueError(
                f"mlp expects one unbatched sample with at least one feature axis, got {tuple(sample_tensor.shape)}"
            )
        if "input_dim" not in resolved and "adaptive_shape" not in resolved:
            resolved["adaptive_shape"] = tuple(int(dim) for dim in sample_tensor.shape)
    elif normalized_model_id == "cnn1d":
        if sample_tensor.ndim != 2:
            raise ValueError(
                f"cnn1d expects one sample shaped (time, feature), got {tuple(sample_tensor.shape)}"
            )
        resolved.setdefault("in_channels", int(sample_tensor.shape[-1]))
    elif normalized_model_id == "resnet18_pose":
        if sample_tensor.ndim != 3:
            raise ValueError(
                "resnet18_pose expects one sample shaped (channel, height, width), "
                f"got {tuple(sample_tensor.shape)}"
            )
        resolved.setdefault("in_channels", int(sample_tensor.shape[0]))

    return _normalize_entry_overrides(normalized_model_id, entry_overrides=resolved)


def get_model_input_contract(model: torch.nn.Module | object) -> ModelInputContract:
    provider = getattr(model, "get_input_contract", None)
    if callable(provider):
        contract = provider()
        if not isinstance(contract, ModelInputContract):
            raise TypeError(
                f"{model.__class__.__name__}.get_input_contract() must return ModelInputContract"
            )
        return contract
    raise ValueError(
        f"Model {model.__class__.__name__} does not declare a model-entry contract. "
        "Implement get_input_contract()."
    )


def get_model_output_contract(model: torch.nn.Module | object) -> ModelOutputContract:
    provider = getattr(model, "get_output_contract", None)
    if callable(provider):
        contract = provider()
        if not isinstance(contract, ModelOutputContract):
            raise TypeError(
                f"{model.__class__.__name__}.get_output_contract() must return ModelOutputContract"
            )
        return contract
    raise ValueError(
        f"Model {model.__class__.__name__} does not declare an output contract. "
        "Implement get_output_contract()."
    )


def validate_model_input(
    tensor: torch.Tensor,
    contract: ModelInputContract,
    *,
    batched: bool = False,
) -> None:
    expected_axes = contract.batched_axes if batched else contract.axes
    if tensor.ndim != len(expected_axes):
        raise DimensionError(
            f"Model input rank mismatch.\n"
            f"Expected axes: {expected_axes}\n"
            f"Got shape: {tuple(tensor.shape)}\n"
            f"Fix: Ensure the model-entry bridge produces the expected layout."
        )

    if contract.dtype_kind == "real" and tensor.is_complex():
        raise DimensionError(
            f"Model input dtype mismatch.\n"
            f"Expected: real tensor for axes {expected_axes}\n"
            f"Got: {tensor.dtype}\n"
            f"Fix: Convert complex RF tensors to a real-valued model representation "
            f"before model entry."
        )
    if contract.dtype_kind == "complex" and not tensor.is_complex():
        raise DimensionError(
            f"Model input dtype mismatch.\n"
            f"Expected: complex tensor for axes {expected_axes}\n"
            f"Got: {tensor.dtype}\n"
            f"Fix: Preserve complex semantics until model entry for this model family."
        )

    for axis_name, expected_size in contract.fixed_sizes.items():
        axis_idx = expected_axes.index(axis_name)
        actual = int(tensor.shape[axis_idx])
        if actual != expected_size:
            raise DimensionError(
                f"Model input axis '{axis_name}' mismatch.\n"
                f"Expected {axis_name}={expected_size} under axes {expected_axes}\n"
                f"Got shape: {tuple(tensor.shape)}\n"
                f"Fix: Ensure the representation bridge uses the same canonicalization "
                f"assumed by the target model."
            )


def validate_model_entry(
    model: torch.nn.Module | object,
    sample_tensor: torch.Tensor,
) -> ModelInputContract:
    contract = get_model_input_contract(model)
    validate_model_input(sample_tensor, contract, batched=False)
    validate_model_input(sample_tensor.unsqueeze(0), contract, batched=True)
    return contract


def validate_model_output(
    model: torch.nn.Module | object,
    outputs: object,
    *,
    batch_size: int | None = None,
    target_schema: Mapping[str, object] | None = None,
) -> ModelOutputContract:
    contract = get_model_output_contract(model)
    if contract.kind == "classification_logits":
        if not torch.is_tensor(outputs):
            raise TypeError(
                f"Classification model {model.__class__.__name__} must return a Tensor, "
                f"got {type(outputs)!r}"
            )
        if outputs.ndim != 2:
            raise DimensionError(
                f"Classification outputs must have shape (batch, class), got {tuple(outputs.shape)}"
            )
        if batch_size is not None and int(outputs.shape[0]) != int(batch_size):
            raise DimensionError(
                f"Classification outputs batch mismatch: expected {batch_size}, got {int(outputs.shape[0])}"
            )
        return contract

    if not isinstance(outputs, dict):
        raise TypeError(
            f"Structured-output model {model.__class__.__name__} must return dict[str, Tensor], "
            f"got {type(outputs)!r}"
        )
    for key in contract.required_keys:
        if key not in outputs:
            raise KeyError(
                f"Structured-output model {model.__class__.__name__} is missing required key '{key}'"
            )
    if batch_size is not None:
        for key in contract.required_keys:
            value = outputs[key]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Structured-output model key '{key}' must be a Tensor, got {type(value)!r}"
                )
            if value.ndim < 1 or int(value.shape[0]) != int(batch_size):
                raise DimensionError(
                    f"Structured-output model key '{key}' must start with batch={batch_size}, "
                    f"got shape {tuple(value.shape)}"
                )
    expected_shapes = _normalize_output_shapes(target_schema)
    for key, expected_shape in expected_shapes.items():
        value = outputs.get(key)
        if not torch.is_tensor(value):
            continue
        if tuple(value.shape[1:]) != expected_shape:
            raise DimensionError(
                f"Structured-output model key '{key}' shape mismatch after batch axis. "
                f"Expected {expected_shape}, got {tuple(value.shape[1:])}"
            )
    return contract


def _normalize_output_shapes(target_schema: Mapping[str, object] | None) -> dict[str, tuple[int, ...]]:
    if target_schema is None:
        return {}
    if not isinstance(target_schema, Mapping):
        raise TypeError(f"Unsupported target_schema type: {type(target_schema)!r}")

    normalized: dict[str, tuple[int, ...]] = {}
    for key, shape in target_schema.items():
        normalized_shape = _normalize_output_shape(shape)
        if normalized_shape is not None:
            normalized[str(key)] = normalized_shape
    return normalized


def _normalize_output_shape(shape: object) -> tuple[int, ...] | None:
    if isinstance(shape, torch.Size):
        dims = tuple(int(dim) for dim in shape)
    elif isinstance(shape, (tuple, list)):
        dims = tuple(shape)
    else:
        return None

    normalized: list[int] = []
    for dim in dims:
        if isinstance(dim, int):
            normalized.append(int(dim))
            continue
        text = str(dim).strip()
        if not text:
            continue
        if text.isdigit():
            normalized.append(int(text))
            continue
        return None
    return tuple(normalized) if normalized else None
