"""Internal weight manifests for model loading."""

from __future__ import annotations

from dataclasses import dataclass

from torchvision.models import ResNet18_Weights


@dataclass(frozen=True)
class WeightManifest:
    weights_id: str
    model_id: str
    factory_weights: object


WEIGHT_MANIFESTS: dict[str, WeightManifest] = {
    "resnet18/torchvision-default": WeightManifest(
        weights_id="resnet18/torchvision-default",
        model_id="resnet18",
        factory_weights=ResNet18_Weights.DEFAULT,
    ),
    "resnet18_pose/torchvision-default": WeightManifest(
        weights_id="resnet18_pose/torchvision-default",
        model_id="resnet18_pose",
        factory_weights=ResNet18_Weights.DEFAULT,
    ),
}

DEFAULT_WEIGHTS_BY_MODEL: dict[str, str] = {
    "resnet18": "resnet18/torchvision-default",
    "resnet18_pose": "resnet18_pose/torchvision-default",
}
