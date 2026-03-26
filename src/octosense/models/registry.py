"""Internal model registry and construction helpers behind ``models.load(...)``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from octosense.models.families.image_like.resnet import ResNet18
from octosense.models.families.pose.resnet_pose import ResNet18Pose
from octosense.models.families.reference.cnn1d import SimpleCNN1D
from octosense.models.families.reference.mlp import MLPClassifier
from octosense.models.families.sequence.rfnet import RFNetClassifier
from octosense.models.weights.loaders import get_default_weights_id

_ModelFactory = Callable[..., object]


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalize_resnet18_entry_overrides(entry_overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in entry_overrides.items():
        if key == "in_channels":
            normalized[key] = int(value)
        elif key == "num_classes":
            normalized[key] = int(value)
        else:
            raise ValueError(f"resnet18 does not accept entry override '{key}'")
    return normalized


def _normalize_rfnet_entry_overrides(entry_overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in entry_overrides.items():
        if key == "input_shape":
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{key} must be a sequence of ints, got {type(value)!r}")
            input_shape = tuple(int(item) for item in value)
            if len(input_shape) != 2:
                raise ValueError(f"{key} must contain exactly 2 ints, got {input_shape}")
            if input_shape is None:
                continue
            normalized[key] = input_shape
        elif key == "num_classes":
            normalized[key] = int(value)
        elif key == "hidden_dim":
            normalized[key] = int(value)
        elif key == "dropout":
            normalized[key] = float(value)
        else:
            raise ValueError(f"rfnet does not accept entry override '{key}'")
    return normalized


def _normalize_mlp_entry_overrides(entry_overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in entry_overrides.items():
        if key == "input_dim":
            normalized[key] = int(value)
        elif key == "num_classes":
            normalized[key] = int(value)
        elif key == "hidden_dims":
            if value is None:
                hidden_dims = None
            elif not isinstance(value, (list, tuple)):
                raise TypeError(f"{key} must be a sequence of ints, got {type(value)!r}")
            else:
                hidden_dims = tuple(int(item) for item in value)
            normalized[key] = () if hidden_dims is None else hidden_dims
        elif key == "dropout":
            normalized[key] = float(value)
        elif key == "adaptive_shape":
            if value is None:
                adaptive_shape = None
            elif not isinstance(value, (list, tuple)):
                raise TypeError(f"{key} must be a sequence of ints, got {type(value)!r}")
            else:
                adaptive_shape = tuple(int(item) for item in value)
            normalized[key] = adaptive_shape
        else:
            raise ValueError(f"mlp does not accept entry override '{key}'")
    return normalized


def _normalize_cnn1d_entry_overrides(entry_overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in entry_overrides.items():
        if key in {"in_channels", "num_classes", "hidden_dim", "num_layers", "kernel_size"}:
            normalized[key] = int(value)
        elif key == "dropout":
            normalized[key] = float(value)
        else:
            raise ValueError(f"cnn1d does not accept entry override '{key}'")
    return normalized


def _normalize_resnet18_pose_entry_overrides(entry_overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in entry_overrides.items():
        if key in {"in_channels", "num_joints", "joint_dims"}:
            normalized[key] = int(value)
        elif key == "predict_bbox":
            normalized[key] = _normalize_bool(value)
        else:
            raise ValueError(f"resnet18_pose does not accept entry override '{key}'")
    return normalized


@dataclass(frozen=True)
class ModelRegistration:
    model_id: str
    family: str
    task: str
    factory: _ModelFactory
    boundary_id: str
    entry_override_normalizer: Callable[[Mapping[str, Any]], dict[str, Any]]
    description: str
    default_weights_id: str | None = None


_MODEL_REGISTRY: dict[str, ModelRegistration] = {
    "resnet18": ModelRegistration(
        model_id="resnet18",
        family="image_like",
        task="classification",
        factory=ResNet18,
        boundary_id="resnet18",
        entry_override_normalizer=_normalize_resnet18_entry_overrides,
        description="Reference image-like ResNet18 baseline with configurable channels.",
        default_weights_id=get_default_weights_id("resnet18"),
    ),
    "rfnet": ModelRegistration(
        model_id="rfnet",
        family="sequence",
        task="classification",
        factory=RFNetClassifier,
        boundary_id="rfnet",
        entry_override_normalizer=_normalize_rfnet_entry_overrides,
        description="Sequence RFNet baseline with time-frequency fusion.",
    ),
    "mlp": ModelRegistration(
        model_id="mlp",
        family="reference",
        task="classification",
        factory=MLPClassifier,
        boundary_id="mlp",
        entry_override_normalizer=_normalize_mlp_entry_overrides,
        description="Reference MLP baseline for flattened real-valued features.",
    ),
    "cnn1d": ModelRegistration(
        model_id="cnn1d",
        family="reference",
        task="classification",
        factory=SimpleCNN1D,
        boundary_id="cnn1d",
        entry_override_normalizer=_normalize_cnn1d_entry_overrides,
        description="Reference Conv1D baseline for sequence-style feature tensors.",
    ),
    "resnet18_pose": ModelRegistration(
        model_id="resnet18_pose",
        family="pose",
        task="pose",
        factory=ResNet18Pose,
        boundary_id="resnet18_pose",
        entry_override_normalizer=_normalize_resnet18_pose_entry_overrides,
        description="Pose-oriented ResNet18 baseline with caller-defined target schema.",
        default_weights_id=get_default_weights_id("resnet18_pose"),
    ),
}

def _list_registered_model_ids() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def _get_model_registration(model_id: str) -> ModelRegistration:
    normalized = model_id.strip().lower()
    try:
        return _MODEL_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(_list_registered_model_ids())
        raise ValueError(f"Unsupported model '{model_id}'. Supported: {supported}") from exc


def _normalize_entry_overrides(
    model_id: str,
    *,
    entry_overrides: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if entry_overrides is not None:
        merged.update(dict(entry_overrides))
    duplicate_keys = set(merged).intersection(kwargs)
    if duplicate_keys:
        duplicates = ", ".join(sorted(duplicate_keys))
        raise ValueError(f"Duplicate entry override keys provided twice: {duplicates}")
    merged.update(kwargs)
    merged = {key: value for key, value in merged.items() if value is not None}
    registration = _get_model_registration(model_id)
    return registration.entry_override_normalizer(merged)


def _build_model(
    model_id: str,
    *,
    weights_id: str | None = None,
    entry_overrides: Mapping[str, Any] | None = None,
    **kwargs: Any,
):
    from octosense.models.boundary import resolve_materialized_entry_overrides

    normalized_model_id = _get_model_registration(model_id).model_id
    registration = _get_model_registration(normalized_model_id)
    normalized_entry_overrides = resolve_materialized_entry_overrides(
        normalized_model_id,
        entry_overrides=entry_overrides,
        **kwargs,
    )
    factory_kwargs = dict(normalized_entry_overrides)
    if weights_id is not None:
        factory_kwargs["weights_id"] = weights_id
    return registration.factory(**factory_kwargs)
