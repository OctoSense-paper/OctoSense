"""ResNet18-based pose head for generic structured pose regression."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

from octosense.models.boundary import BoundaryBackedModel
from octosense.models.weights.loaders import resolve_factory_weights


class ResNet18Pose(BoundaryBackedModel):
    """ResNet18 backbone with joint and optional bbox regression heads."""

    boundary_model_id = "resnet18_pose"

    def __init__(
        self,
        in_channels: int,
        num_joints: int,
        *,
        joint_dims: int,
        predict_bbox: bool,
        weights_id: str | None = None,
    ) -> None:
        in_channels = int(in_channels)
        num_joints = int(num_joints)
        joint_dims = int(joint_dims)
        predict_bbox = bool(predict_bbox)
        super().__init__(
            boundary_model_id=self.boundary_model_id,
            entry_overrides={
                "in_channels": in_channels,
                "num_joints": num_joints,
                "joint_dims": joint_dims,
                "predict_bbox": predict_bbox,
            },
        )
        weights = resolve_factory_weights("resnet18_pose", weights_id=weights_id)
        backbone = resnet18(weights=weights)
        self.in_channels = in_channels
        self.weights_id = weights_id
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=False,
            )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.num_joints = num_joints
        self.joint_dims = joint_dims
        self.predict_bbox = predict_bbox
        self.joint_head = nn.Linear(feature_dim, num_joints * joint_dims)
        self.bbox_head = nn.Linear(feature_dim, 4) if predict_bbox else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        joints = self.joint_head(features).reshape(x.shape[0], self.num_joints, self.joint_dims)
        out = {"joints": joints}
        if self.bbox_head is not None:
            out["bbox"] = self.bbox_head(features)
        return out
