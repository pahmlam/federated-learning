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
from src.utils.env import fl_env_values

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


def ppe_index_to_label() -> dict[int, str]:
    """Map detection label index (1-based) -> canonical PPE class name (0 = background)."""

    return {index + 1: name for index, name in enumerate(PPE_CORE_CLASSES)}


@dataclass(frozen=True)
class DetectionConfig:
    exp_id: str = DEFAULT_DET_EXP_ID
    output_dir: str = DEFAULT_DET_OUTPUT_DIR
    manifest_path: str = DEFAULT_DET_MANIFEST_PATH
    root_dir: str = DEFAULT_DET_ROOT_DIR
    client_id: str | None = None
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
    # Straggler/dropout robustness knobs. Unset (None) keeps the strict baseline:
    # the ServerApp then requires all ``num_clients`` for train/evaluate/availability.
    # Set explicitly (e.g. 1) to let a round proceed with a Colab node disconnected.
    min_train_nodes: int | None = None
    min_evaluate_nodes: int | None = None
    min_available_nodes: int | None = None

    def normalized(self) -> "DetectionConfig":
        _validate_detection_config(self)
        return self

    @property
    def effective_min_train_nodes(self) -> int:
        """Min clients required to train a round (defaults to strict ``num_clients``)."""

        return self.num_clients if self.min_train_nodes is None else self.min_train_nodes

    @property
    def effective_min_evaluate_nodes(self) -> int:
        """Min clients required to evaluate a round (defaults to strict ``num_clients``)."""

        return (
            self.num_clients
            if self.min_evaluate_nodes is None
            else self.min_evaluate_nodes
        )

    @property
    def effective_min_available_nodes(self) -> int:
        """Min clients that must be connected before a round (defaults to ``num_clients``)."""

        return (
            self.num_clients
            if self.min_available_nodes is None
            else self.min_available_nodes
        )

    @classmethod
    def from_run_config(cls, values: dict[str, Any]) -> "DetectionConfig":
        normalized = {key.replace("-", "_"): value for key, value in values.items()}
        field_names = set(cls.__dataclass_fields__)
        init_values = {key: value for key, value in normalized.items() if key in field_names}
        return cls(**init_values).normalized()

    @classmethod
    def from_env_and_overrides(
        cls,
        overrides: dict[str, Any] | None = None,
        *,
        env_path: str | None = None,
        env_overrides: bool = False,
    ) -> "DetectionConfig":
        """Build config from defaults, ``.env`` values, and explicit overrides.

        Default precedence is ``defaults < env < overrides``. Flower deployment
        can set ``env_overrides=True`` to let env replace run_config values that
        still equal dataclass defaults while preserving explicit run_config
        overrides such as ``num-clients=2``.
        """

        raw_overrides = overrides or {}
        normalized_overrides = {
            key.replace("-", "_"): value for key, value in raw_overrides.items()
        }
        field_names = set(cls.__dataclass_fields__)
        init_values = {
            key: value for key, value in normalized_overrides.items() if key in field_names
        }
        env_values = fl_env_values(cls, path=env_path)
        if env_overrides:
            default_config = cls()
            merged = {**init_values, **env_values}
            for key, value in init_values.items():
                if getattr(default_config, key) != value:
                    merged[key] = value
        else:
            merged = {**env_values, **init_values}
        return cls(**merged).normalized()


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
    for field_name in ("min_train_nodes", "min_evaluate_nodes", "min_available_nodes"):
        value = getattr(config, field_name)
        if value is not None and value < 1:
            raise ValueError(f"{field_name} must be >= 1 when set")
        if value is not None and value > config.num_clients:
            raise ValueError(f"{field_name} must be <= num_clients when set")
    if config.effective_min_available_nodes < config.effective_min_train_nodes:
        raise ValueError("min_available_nodes must be >= min_train_nodes")
    if config.effective_min_available_nodes < config.effective_min_evaluate_nodes:
        raise ValueError("min_available_nodes must be >= min_evaluate_nodes")
