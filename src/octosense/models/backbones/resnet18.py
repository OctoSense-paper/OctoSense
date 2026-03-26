"""ResNet18 backbone wrapper."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    """ResNet18 backbone with configurable input channels and classes."""

    def __init__(self, in_channels: int, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

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
