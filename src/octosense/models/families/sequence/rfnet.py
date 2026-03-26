"""RFNet-style sequence classifier adapted to OctoSense contracts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchvision.models import resnet18

from octosense.models.boundary import BoundaryBackedModel


def _build_feature_extractor() -> nn.Module:
    backbone = resnet18(weights=None)
    return nn.Sequential(*list(backbone.children())[:-1])


class _FCNet(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        activate: str | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        activation = (activate or "").lower()
        if activation == "relu":
            self.activation: nn.Module = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.linear(x)
        return self.activation(x)


class _BiAttention(nn.Module):
    def __init__(
        self,
        *,
        time_features: int,
        freq_features: int,
        mid_features: int,
        glimpses: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_aug = 3
        expanded_features = int(mid_features * hidden_aug)
        self.glimpses = glimpses
        self.lin_time = _FCNet(
            time_features,
            expanded_features,
            activate="relu",
            drop=drop / 2.5,
        )
        self.lin_freq = _FCNet(
            freq_features,
            expanded_features,
            activate="relu",
            drop=drop / 2.5,
        )
        self.h_weight = nn.Parameter(
            torch.empty(1, glimpses, 1, expanded_features).normal_()
        )
        self.h_bias = nn.Parameter(torch.empty(1, glimpses, 1, 1).normal_())
        self.drop = nn.Dropout(drop)

    def forward(self, time: torch.Tensor, freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_num = int(time.shape[1])
        freq_num = int(freq.shape[1])

        time_proj = self.drop(self.lin_time(time)).unsqueeze(1)
        freq_proj = self.lin_freq(freq).unsqueeze(1)
        weighted = time_proj * self.h_weight
        logits = torch.matmul(weighted, freq_proj.transpose(2, 3)) + self.h_bias
        attention = F.softmax(logits.reshape(-1, self.glimpses, time_num * freq_num), dim=2)
        return attention.view(-1, self.glimpses, time_num, freq_num), logits


class _ApplySingleAttention(nn.Module):
    def __init__(
        self,
        *,
        time_features: int,
        freq_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin_time = _FCNet(time_features, time_features, activate="relu", drop=drop)
        self.lin_freq = _FCNet(freq_features, freq_features, activate="relu", drop=drop)

    def forward(
        self,
        time: torch.Tensor,
        freq: torch.Tensor,
        attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended_time = self.lin_time((time.transpose(1, 2) @ attention).transpose(1, 2))
        attended_freq = self.lin_freq((freq.transpose(1, 2) @ attention).transpose(1, 2))
        return attended_time, attended_freq


class _ApplyAttention(nn.Module):
    def __init__(
        self,
        *,
        time_features: int,
        freq_features: int,
        mid_features: int,
        glimpses: int,
        num_obj: int,
        drop: float = 0.0,
    ) -> None:
        del mid_features, num_obj
        super().__init__()
        self.glimpse_layers = nn.ModuleList(
            [
                _ApplySingleAttention(
                    time_features=time_features,
                    freq_features=freq_features,
                    drop=drop,
                )
                for _ in range(glimpses)
            ]
        )

    def forward(
        self,
        time: torch.Tensor,
        freq: torch.Tensor,
        attention: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del logits
        for glimpse_index, layer in enumerate(self.glimpse_layers):
            attended_time, attended_freq = layer(time, freq, attention[:, glimpse_index, :, :])
            time = time + attended_time
            freq = freq + attended_freq
        return time, freq


class RFNetClassifier(BoundaryBackedModel):
    """Canonical RFNet-style sequence classifier.

    The implementation keeps the RFNet family architecture dataset-agnostic:
    a time-domain LSTM branch, an FFT-magnitude LSTM branch, a bi-attention
    fusion block, and a ResNet18 feature extractor over fused time-frequency
    heat maps.
    """

    boundary_model_id = "rfnet"

    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ) -> None:
        if len(input_shape) != 2:
            raise ValueError(f"RFNetClassifier expects (time, features), got {input_shape}")
        time_steps, feature_dim = input_shape
        if time_steps <= 0 or feature_dim <= 0:
            raise ValueError(f"RFNetClassifier input dims must be > 0, got {input_shape}")
        if hidden_dim <= 0 or hidden_dim % 2 != 0:
            raise ValueError(f"RFNetClassifier hidden_dim must be a positive even number, got {hidden_dim}")

        time_steps = int(time_steps)
        feature_dim = int(feature_dim)
        hidden_dim = int(hidden_dim)
        num_classes = int(num_classes)
        dropout = float(dropout)
        super().__init__(
            boundary_model_id=self.boundary_model_id,
            entry_overrides={
                "input_shape": (time_steps, feature_dim),
                "num_classes": num_classes,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
            },
        )

        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.glimpses = 1

        self.pre_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        recurrent_hidden = self.hidden_dim // 2
        self.lstm_time = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=recurrent_hidden,
            batch_first=True,
        )
        self.lstm_freq = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=recurrent_hidden,
            batch_first=True,
        )

        self.cnn_in = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
        self.cnn_backbone = _build_feature_extractor()
        self.cnn_proj = (
            nn.Identity()
            if self.hidden_dim == 512
            else nn.Linear(512, self.hidden_dim)
        )

        self.attention = weight_norm(
            _BiAttention(
                time_features=recurrent_hidden,
                freq_features=recurrent_hidden,
                mid_features=self.hidden_dim,
                glimpses=self.glimpses,
                drop=0.5,
            ),
            name="h_weight",
            dim=None,
        )
        self.apply_attention = _ApplyAttention(
            time_features=recurrent_hidden,
            freq_features=recurrent_hidden,
            mid_features=recurrent_hidden,
            glimpses=self.glimpses,
            num_obj=512,
            drop=0.2,
        )
        self.time_head = _FCNet(
            recurrent_hidden,
            recurrent_hidden,
            activate="relu",
            drop=self.dropout,
        )
        self.freq_head = _FCNet(
            recurrent_hidden,
            recurrent_hidden,
            activate="relu",
            drop=self.dropout,
        )
        self.fusion_head = _FCNet(self.hidden_dim, self.hidden_dim, drop=self.dropout)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "RFNetClassifier expects input shaped (batch, time, features), "
                f"got {tuple(x.shape)}"
            )
        _, time_steps, feature_dim = x.shape
        if time_steps != self.time_steps or feature_dim != self.feature_dim:
            raise ValueError(
                "RFNetClassifier input shape mismatch: "
                f"expected (*, {self.time_steps}, {self.feature_dim}), got {tuple(x.shape)}"
            )

        x = x.float()
        x_freq = torch.fft.fft(x, dim=1).abs().float()
        combined = self.pre_fusion(torch.cat([x, x_freq], dim=-1))

        heatmap = combined.view(x.shape[0], self.time_steps, self.hidden_dim // 2, 2).permute(0, 3, 2, 1)
        cnn_features = self.cnn_backbone(self.cnn_in(heatmap)).flatten(1)
        cnn_features = self.cnn_proj(cnn_features)

        time_features, _ = self.lstm_time(x)
        freq_features, _ = self.lstm_freq(x_freq)
        attention, logits = self.attention(time_features, freq_features)
        time_features, freq_features = self.apply_attention(
            time_features,
            freq_features,
            attention,
            logits,
        )

        time_summary = self.time_head(time_features[:, -1, :])
        freq_summary = self.freq_head(freq_features[:, -1, :])
        fused = self.fusion_head(torch.cat([time_summary, freq_summary], dim=-1)) + cnn_features
        return self.classifier(fused)
