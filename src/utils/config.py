"""Configuration defaults for the synthetic FL demo."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class DemoConfig:
    exp_id: str = "EXP-001"
    output_dir: str = "outputs/EXP-001"
    num_clients: int = 5
    num_classes: int = 3
    input_dim: int = 16
    embedding_dim: int = 8
    samples_per_client: int = 120
    val_fraction: float = 0.25
    partition: str = "non_iid"
    dirichlet_alpha: float = 0.3
    batch_size: int = 16
    local_epochs: int = 2
    centralized_epochs: int = 4
    lr: float = 0.05
    num_rounds: int = 3
    seed: int = 2026
    quick: bool = False

    def normalized(self) -> "DemoConfig":
        config = self
        if self.quick:
            config = replace(
                config,
                samples_per_client=min(self.samples_per_client, 40),
                batch_size=min(self.batch_size, 8),
                local_epochs=1,
                centralized_epochs=2,
                num_rounds=2,
            )
        return config

    @classmethod
    def from_run_config(cls, values: dict[str, Any]) -> "DemoConfig":
        normalized = {_python_key(key): value for key, value in values.items()}
        mapping = {
            "num_server_rounds": "num_rounds",
        }
        init_values: dict[str, Any] = {}
        field_names = set(cls.__dataclass_fields__)
        for key, value in normalized.items():
            target = mapping.get(key, key)
            if target in field_names:
                init_values[target] = value
        return cls(**init_values).normalized()


def build_config(
    *,
    partition: str = "non_iid",
    quick: bool = False,
    output_dir: str = "outputs/EXP-001",
    seed: int = 2026,
) -> DemoConfig:
    return DemoConfig(
        partition=partition,
        quick=quick,
        output_dir=output_dir,
        seed=seed,
    ).normalized()


def _python_key(key: str) -> str:
    return key.replace("-", "_")
