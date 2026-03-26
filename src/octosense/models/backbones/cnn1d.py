"""Simple 1D CNN for WiFi CSI gesture recognition."""

import torch
import torch.nn as nn


class SimpleCNN1D(nn.Module):
    """Simple 1D CNN for CSI feature extraction.

    Architecture:
    - Conv1D layers for temporal feature extraction
    - Batch normalization for training stability
    - Dropout for regularization
    - Global average pooling
    - Fully connected classification head

    Example:
        >>> model = SimpleCNN1D(in_channels=30, num_classes=5)
        >>> x = torch.randn(16, 100, 30)  # (batch, time, channels)
        >>> output = model(x)  # (16, 5)
    """

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
            in_channels: Number of input channels (e.g., 30 subcarriers)
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for conv layers
            num_layers: Number of conv layers (default: 3)
            kernel_size: Kernel size for conv layers (default: 5)
            dropout: Dropout probability (default: 0.3)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Build conv layers
        layers: list[nn.Module] = []
        current_channels = in_channels

        for _ in range(num_layers):
            # Conv1D (expects input: (batch, channels, time))
            layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            current_channels = hidden_dim

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling (adaptive to any sequence length)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, time, channels)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Input: (batch, time, channels)
        # Conv1D expects: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, channels, time)

        # Conv layers
        x = self.conv_layers(x)  # (batch, hidden_dim, time)

        # Global average pooling
        x = self.global_pool(x)  # (batch, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch, hidden_dim)

        # Classification
        x = self.classifier(x)  # (batch, num_classes)

        return x
