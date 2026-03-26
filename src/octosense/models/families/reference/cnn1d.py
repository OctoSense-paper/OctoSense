"""Simple 1D CNN baseline for sequence-style feature tensors."""

import torch
import torch.nn as nn

from octosense.models.boundary import BoundaryBackedModel


class SimpleCNN1D(BoundaryBackedModel):
    """Simple 1D CNN for generic sequence feature extraction.

    Architecture:
    - Conv1D layers for temporal feature extraction
    - Batch normalization for training stability
    - Dropout for regularization
    - Global average pooling
    - Fully connected classification head

    Example:
        >>> model = SimpleCNN1D(in_channels=30, num_classes=5)
        >>> x = torch.randn(16, 100, 30)  # (batch, time, feature)
        >>> output = model(x)  # (16, 5)
    """

    boundary_model_id = "cnn1d"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        """Initialize SimpleCNN1D.

        Args:
            in_channels: Feature width of each time step
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for conv layers
            num_layers: Number of conv layers (default: 3)
            kernel_size: Kernel size for conv layers (default: 5)
            dropout: Dropout probability (default: 0.3)
        """
        in_channels = int(in_channels)
        num_classes = int(num_classes)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        kernel_size = int(kernel_size)
        dropout = float(dropout)
        super().__init__(
            boundary_model_id=self.boundary_model_id,
            entry_overrides={
                "in_channels": in_channels,
                "num_classes": num_classes,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "kernel_size": kernel_size,
                "dropout": dropout,
            },
        )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Build conv layers
        layers: list[nn.Module] = []
        current_channels = self.in_channels

        for _ in range(self.num_layers):
            # Conv1D expects input shaped (batch, channels, time).
            layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            current_channels = self.hidden_dim

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling (adaptive to any sequence length)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, time, feature)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Input: (batch, time, feature)
        # Conv1D expects: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, feature, time)

        # Conv layers
        x = self.conv_layers(x)  # (batch, hidden_dim, time)

        # Global average pooling
        x = self.global_pool(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)

        # Classification
        x = self.classifier(x)  # (batch, num_classes)

        return x
