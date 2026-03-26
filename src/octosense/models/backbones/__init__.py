"""Backbone models for signal classification."""

from torchradio.models.backbones.cnn1d import SimpleCNN1D
from torchradio.models.backbones.resnet18 import ResNet18

__all__ = [
    "SimpleCNN1D",
    "ResNet18",
]
