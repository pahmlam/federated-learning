"""Configuration for the PPE object-detection federated track.

Mirrors the philosophy of ``DemoConfig`` (frozen dataclass, ``from_run_config``)
but for detection: the trainable part is the detector head, FedAvg aggregates
only that head, and the primary metric is mAP. Class order is the single source
of truth shared by the dataset (label remap) and the model (head size).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.manifest_generator import DEFAULT_CORE_PPE

# Canonical class order: index 0 is background, PPE classes are 1..8.
PPE_CORE_CLASSES: tuple[str, ...] = DEFAULT_CORE_PPE
NUM_DETECTION_CLASSES: int = len(PPE_CORE_CLASSES) + 1  # + background

DEFAULT_DET_EXP_ID = "EXP-011"
DEFAULT_DET_OUTPUT_DIR = "outputs/EXP-011"
DEFAULT_DET_MANIFEST_PATH = "configs/datasets/ppe_detection_exp011_manifest.csv"
DEFAULT_DET_ROOT_DIR = "data/ppe"
VALID_DEVICES = {"auto", "cpu", "cuda"}


def ppe_label_to_index() -> dict[str, int]:
    """Map PPE class name (lowercase) -> detection label (1-based; 0 = background)."""

    return {name.strip().lower(): index + 1 for index, name in enumerate(PPE_CORE_CLASSES)}


@dataclass(frozen=True)
class DetectionConfig:
    exp_id: str = DEFAULT_DET_EXP_ID
    output_dir: str = DEFAULT_DET_OUTPUT_DIR
    manifest_path: str = DEFAULT_DET_MANIFEST_PATH
    root_dir: str = DEFAULT_DET_ROOT_DIR
    num_clients: int = 3
    num_classes: int = NUM_DETECTION_CLASSES
    image_size: int = 512
    batch_size: int = 2
    local_epochs: int = 1
    centralized_epochs: int = 1
    num_rounds: int = 1
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_workers: int = 0
    score_threshold: float = 0.05
    device: str = "auto"
    pretrained: bool = True
    seed: int = 2026

    def normalized(self) -> "DetectionConfig":
        _validate_detection_config(self)
        return self

    @classmethod
    def from_run_config(cls, values: dict[str, Any]) -> "DetectionConfig":
        normalized = {key.replace("-", "_"): value for key, value in values.items()}
        field_names = set(cls.__dataclass_fields__)
        init_values = {key: value for key, value in normalized.items() if key in field_names}
        return cls(**init_values).normalized()


def _validate_detection_config(config: DetectionConfig) -> None:
    if config.num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if config.num_classes < 2:
        raise ValueError("num_classes must be >= 2 (background + at least one class)")
    if config.image_size < 64:
        raise ValueError("image_size must be >= 64")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.local_epochs < 1:
        raise ValueError("local_epochs must be >= 1")
    if config.centralized_epochs < 1:
        raise ValueError("centralized_epochs must be >= 1")
    if config.num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")
    if config.lr <= 0:
        raise ValueError("lr must be > 0")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if not 0.0 <= config.score_threshold < 1.0:
        raise ValueError("score_threshold must be in [0, 1)")
    if config.device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {sorted(VALID_DEVICES)}")
