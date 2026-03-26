"""Generic MLP classifier for flattened tensor inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from octosense.models.boundary import BoundaryBackedModel


class MLPClassifier(BoundaryBackedModel):
    """Flatten arbitrary non-batch dimensions and apply a configurable MLP."""

    boundary_model_id = "mlp"

    def __init__(
        self,
        *,
        input_dim: int | None = None,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        adaptive_shape: tuple[int, ...] | None = None,
    ) -> None:
        inferred_input_dim = None if input_dim is None else int(input_dim)
        num_classes = int(num_classes)
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        dropout = float(dropout)
        adaptive_shape = (
            None
            if adaptive_shape is None
            else tuple(int(dim) for dim in adaptive_shape)
        )
        if inferred_input_dim is None and adaptive_shape is not None:
            flattened_dim = 1
            for dim in adaptive_shape:
                flattened_dim *= int(dim)
            inferred_input_dim = int(flattened_dim)
        super().__init__(
            boundary_model_id=self.boundary_model_id,
            entry_overrides={
                "num_classes": num_classes,
                "input_dim": inferred_input_dim,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "adaptive_shape": adaptive_shape,
            },
        )
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.adaptive_shape = adaptive_shape
        self.input_dim = inferred_input_dim
        self.network: nn.Sequential | None = None
        if self.input_dim is not None:
            self.network = self._build_network(self.input_dim)

    def _feature_axes(self) -> tuple[str, ...]:
        if self.adaptive_shape is not None:
            return tuple(f"feature_dim_{index + 1}" for index in range(len(self.adaptive_shape)))
        return ("feature",)

    def _build_network(self, input_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(previous, int(hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            previous = int(hidden_dim)
        layers.append(nn.Linear(previous, self.num_classes))
        return nn.Sequential(*layers)

    def _ensure_network(self, input_dim: int) -> nn.Sequential:
        if self.network is None:
            self.input_dim = int(input_dim)
            self.network = self._build_network(self.input_dim)
        return self.network

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_shape is None:
            return x
        feature_dims = x.ndim - 1
        if len(self.adaptive_shape) != feature_dims:
            raise ValueError(
                "adaptive_shape must match the number of non-batch feature dimensions, "
                f"got adaptive_shape={self.adaptive_shape} for shape={tuple(x.shape)}"
            )
        if feature_dims == 1:
            return F.adaptive_avg_pool1d(x.unsqueeze(1), self.adaptive_shape[0]).squeeze(1)
        if feature_dims == 2:
            return F.adaptive_avg_pool2d(x.unsqueeze(1), self.adaptive_shape).squeeze(1)
        if feature_dims == 3:
            return F.adaptive_avg_pool3d(x.unsqueeze(1), self.adaptive_shape).squeeze(1)
        raise ValueError(
            "MLPClassifier adaptive pooling currently supports up to 3 feature dimensions, "
            f"got shape={tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                "MLPClassifier expects a batched tensor with at least one feature dimension, "
                f"got shape={tuple(x.shape)}"
            )
        pooled = self._pool_features(x)
        flattened = pooled.reshape(int(pooled.shape[0]), -1)
        network = self._ensure_network(int(flattened.shape[1]))
        return network(flattened)

    def initialize_for_sample(self, sample_tensor: torch.Tensor) -> MLPClassifier:
        if sample_tensor.ndim < 1:
            raise ValueError(
                "MLPClassifier.initialize_for_sample expects an unbatched sample tensor with "
                f"at least one feature dimension, got shape={tuple(sample_tensor.shape)}"
            )
        if self.network is not None:
            return self
        with torch.no_grad():
            pooled = self._pool_features(sample_tensor.unsqueeze(0))
            flattened_dim = int(pooled.reshape(1, -1).shape[1])
        self.input_dim = int(flattened_dim)
        self.network = self._build_network(self.input_dim)
        return self
