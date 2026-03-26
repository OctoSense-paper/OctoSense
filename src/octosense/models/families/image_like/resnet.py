"""ResNet18 image-like family implementation."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchvision.models import resnet18

from octosense.models.boundary import BoundaryBackedModel
from octosense.models.weights.loaders import resolve_factory_weights


class ResNet18(BoundaryBackedModel):
    """ResNet18 family with configurable input channels and classes."""

    boundary_model_id = "resnet18"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        weights_id: str | None = None,
    ) -> None:
        in_channels = int(in_channels)
        num_classes = int(num_classes)
        super().__init__(
            boundary_model_id=self.boundary_model_id,
            entry_overrides={
                "in_channels": in_channels,
                "num_classes": num_classes,
            },
        )
        weights = resolve_factory_weights("resnet18", weights_id=weights_id)
        model = resnet18(weights=weights)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.weights_id = weights_id

        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=False,
            )

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))
