"""Private lowering helpers for pipeline execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from octosense.specs.schemas.benchmark import BenchmarkSpec

if TYPE_CHECKING:
    from octosense.specs.schemas.runtime import RuntimeSpec
    from octosense.transforms import Sequential


def _bootstrap_operator_registry() -> None:
    from octosense.transforms.core.registry import ensure_operator_registry_bootstrapped

    ensure_operator_registry_bootstrapped()


def transform_to_step_dict(transform: object) -> dict[str, object]:
    """Serialize one transform instance into an internal runtime payload."""

    custom_serializer = getattr(transform, "to_dict", None)
    if callable(custom_serializer):
        payload = custom_serializer()
        if not isinstance(payload, dict):
            raise TypeError(
                f"{transform.__class__.__name__}.to_dict() must return a mapping for runtime specs."
            )
        operator_id = payload.get("operator_id")
        params = payload.get("params", {})
        if not isinstance(operator_id, str) or not operator_id:
            raise ValueError(
                f"{transform.__class__.__name__}.to_dict() must include a non-empty operator_id."
            )
        if not isinstance(params, dict):
            raise TypeError(
                f"{transform.__class__.__name__}.to_dict() params must be a mapping for runtime specs."
            )
        return {
            "operator_id": operator_id,
            "params": dict(params),
        }

    signature = inspect.signature(transform.__init__)
    valid_params = set(signature.parameters) - {"self"}

    params: dict[str, object] = {}
    for key, value in vars(transform).items():
        if key.startswith("_") or "contract" in key or key not in valid_params:
            continue
        if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
            params[key] = value

    return {
        "operator_id": transform.__class__.__name__,
        "params": params,
    }


def _transform_from_step_dict(payload: dict[str, object]) -> object:
    from octosense.transforms.core.registry import get_operator_class

    operator_id = payload.get("operator_id")
    if not isinstance(operator_id, str) or not operator_id:
        raise ValueError("Runtime transform payload requires a non-empty operator_id.")
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Runtime transform payload params must be a mapping.")

    _bootstrap_operator_registry()
    try:
        transform_class = get_operator_class(operator_id)
    except KeyError as exc:
        raise ValueError(f"Transform operator {operator_id!r} is not registered.") from exc
    return transform_class(**params)


def transform_from_step_dict(payload: dict[str, object]) -> object:
    """Deserialize one runtime step payload into a transform instance."""

    return _transform_from_step_dict(payload)


class PipelineExecutionHandle(ABC):
    """Internal handle contract shared by pipeline builder and execution runner."""

    @abstractmethod
    def _resolve_execution_payload(
        self,
        *,
        mode: str | None,
        runtime: "RuntimeSpec | Mapping[str, Any] | None" = None,
    ) -> Mapping[str, object]:
        """Return one lowered execution payload for runner-owned internal reconstruction."""

    @abstractmethod
    def _build_execution_components(
        self,
        *,
        seed: int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> tuple[Any, Any]:
        """Build model and dataloaders for train/eval execution."""

    @abstractmethod
    def _prepare_inference_inputs(self, inputs: Any) -> Any:
        """Project caller inputs into runtime-ready tensors."""

    @abstractmethod
    def get_execution_dataset_source(self) -> Any:
        """Expose the canonical dataset source bound to this handle."""

    def get_benchmark_spec(self) -> BenchmarkSpec | None:
        """Expose the frozen canonical BenchmarkSpec bound to this handle when available."""

        return None


@dataclass(slots=True)
class RuntimeTransformStep:
    """Private execution-lowering step used only inside ``octosense.pipelines``."""

    operator_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "RuntimeTransformStep":
        if payload is None:
            return cls(operator_id="")
        params = payload.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("RuntimeTransformStep.params must be a mapping")
        return cls(
            operator_id=str(payload.get("operator_id", "") or ""),
            params=dict(params),
        )

    @classmethod
    def from_transform(cls, transform: object) -> "RuntimeTransformStep":
        return cls.from_dict(transform_to_step_dict(transform))


@dataclass(slots=True)
class RuntimePipelineSpec:
    """Private execution payload used by pipeline builder/runner internals."""

    transforms: list[RuntimeTransformStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "transforms": [step.to_dict() for step in self.transforms],
            "metadata": dict(self.metadata),
            "task": dict(self.task),
            "dataset": dict(self.dataset),
            "model": dict(self.model),
            "training": dict(self.training),
            "runtime": dict(self.runtime),
            "protocol": dict(self.protocol),
            "seed": self.seed,
        }

    def build_transform(self) -> "Sequential":
        from octosense.transforms import compose

        return compose(_transform_from_step_dict(step.to_dict()) for step in self.transforms)

    def validate(self) -> None:
        if not isinstance(self.metadata, dict):
            raise TypeError("RuntimePipelineSpec.metadata must be a mapping")
        if not isinstance(self.task, dict):
            raise TypeError("RuntimePipelineSpec.task must be a mapping")
        if not isinstance(self.dataset, dict):
            raise TypeError("RuntimePipelineSpec.dataset must be a mapping")
        if not isinstance(self.model, dict):
            raise TypeError("RuntimePipelineSpec.model must be a mapping")
        if not isinstance(self.training, dict):
            raise TypeError("RuntimePipelineSpec.training must be a mapping")
        if not isinstance(self.runtime, dict):
            raise TypeError("RuntimePipelineSpec.runtime must be a mapping")
        if not isinstance(self.protocol, dict):
            raise TypeError("RuntimePipelineSpec.protocol must be a mapping")
        if not isinstance(self.transforms, list):
            raise TypeError("RuntimePipelineSpec.transforms must be a list")

    def replace_transforms(
        self,
        transforms: Iterable[object],
    ) -> "RuntimePipelineSpec":
        payload = self.to_dict()
        payload["transforms"] = [
            RuntimeTransformStep.from_transform(transform).to_dict()
            for transform in transforms
        ]
        return type(self).from_dict(cast_runtime_payload(payload))

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "RuntimePipelineSpec":
        if payload is None:
            return cls()
        transforms_payload = payload.get("transforms", [])
        if not isinstance(transforms_payload, list):
            raise TypeError("RuntimePipelineSpec.transforms must be a list")
        return cls(
            transforms=[RuntimeTransformStep.from_dict(step) for step in transforms_payload],
            metadata=_mapping(payload.get("metadata"), "metadata"),
            task=_mapping(payload.get("task"), "task"),
            dataset=_mapping(payload.get("dataset"), "dataset"),
            model=_mapping(payload.get("model"), "model"),
            training=_mapping(payload.get("training"), "training"),
            runtime=_mapping(payload.get("runtime"), "runtime"),
            protocol=_mapping(payload.get("protocol"), "protocol"),
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
        )


def _build_execution_spec(spec: BenchmarkSpec) -> RuntimePipelineSpec:
    """Lower one already-resolved ``BenchmarkSpec`` into the execution payload form."""

    dataset_payload = spec.dataset.to_dict()
    runtime_payload = spec.runtime.to_dict()
    model_payload = spec.model.to_dict()
    task_payload = spec.task.to_dict()
    return RuntimePipelineSpec(
        transforms=[
            RuntimeTransformStep(
                operator_id=step.operator_id,
                params=copy.deepcopy(step.params),
            )
            for step in spec.transform.steps
        ],
        metadata={},
        task={
            "task_id": str(task_payload.get("task_id", "") or ""),
            "task_binding": str(task_payload.get("task_binding", "") or ""),
        },
        dataset={
            "dataset_id": str(dataset_payload.get("dataset_id", "") or ""),
            "variant": dataset_payload.get("variant"),
            "split_scheme": dataset_payload.get("split_scheme"),
            "path": dataset_payload.get("dataset_root"),
            "modalities": list(dataset_payload.get("modalities", []) or []),
            "input_selection": copy.deepcopy(dataset_payload.get("input_selection")),
        },
        model={
            "model_id": str(model_payload.get("model_id", "") or ""),
            "weights_id": model_payload.get("weights_id"),
            "entry_overrides": copy.deepcopy(model_payload.get("entry_overrides", {})),
        },
        runtime=dict(runtime_payload),
        protocol=copy.deepcopy(spec.protocol),
    )


def cast_runtime_payload(payload: dict[str, object]) -> dict[str, object]:
    return dict(payload)


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"RuntimePipelineSpec.{field_name} must be a mapping")
    return dict(value)
