"""Faster R-CNN detector with a frozen backbone for head-only federated learning.

Builds ``fasterrcnn_mobilenet_v3_large_fpn``, swaps the box predictor for the PPE
class count, and freezes the backbone (which includes the FPN). Only the RPN head
and ROI heads remain trainable -- that head is what FedAvg aggregates. Head
parameters are serialized in a stable ``named_parameters`` order so aggregation is
consistent across heterogeneous clients (cpu/cuda).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.utils.detection_config import NUM_DETECTION_CLASSES


def build_detection_model(
    num_classes: int = NUM_DETECTION_CLASSES,
    *,
    pretrained: bool = True,
    seed: int = 2026,
) -> nn.Module:
    """Build a Faster R-CNN MobileNetV3-FPN with a frozen backbone.

    ``pretrained=False`` skips all weight downloads (random init) for fast tests.
    """

    torch.manual_seed(seed)
    weights = "DEFAULT" if pretrained else None
    weights_backbone = "DEFAULT" if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    _freeze_backbone(model)
    return model


def _freeze_backbone(model: nn.Module) -> None:
    for param in model.backbone.parameters():
        param.requires_grad_(False)


def detection_trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def get_detection_head_parameters(model: nn.Module) -> list[np.ndarray]:
    return [
        param.detach().cpu().numpy().copy()
        for param in model.parameters()
        if param.requires_grad
    ]


def set_detection_head_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    trainable = [param for param in model.parameters() if param.requires_grad]
    if len(parameters) != len(trainable):
        raise ValueError(
            f"Expected {len(trainable)} head parameter arrays, got {len(parameters)}"
        )
    with torch.no_grad():
        for param, array in zip(trainable, parameters, strict=True):
            param.copy_(torch.as_tensor(array, dtype=param.dtype, device=param.device))


def resolve_device(name: str = "auto") -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
