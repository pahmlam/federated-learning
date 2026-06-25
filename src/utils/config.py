"""Configuration defaults for the synthetic FL demo."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


DEFAULT_EXP_ID = "EXP-001"
DEFAULT_OUTPUT_DIR = "outputs/EXP-001"
OOM_SAFE_EXP_ID = "EXP-002"
OOM_SAFE_OUTPUT_DIR = "outputs/EXP-002"
VALID_PROFILES = {"default", "quick", "oom-safe"}


@dataclass(frozen=True)
class DemoConfig:
    exp_id: str = DEFAULT_EXP_ID
    output_dir: str = DEFAULT_OUTPUT_DIR
    profile: str = "default"
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
    weight_decay: float = 0.0
    normalize_embedding: bool = False
    head_hidden_dim: int | None = None
    num_rounds: int = 3
    num_workers: int = 0
    client_num_cpus: float = 1.0
    ray_num_cpus: int | None = None
    seed: int = 2026
    quick: bool = False

    def normalized(self) -> "DemoConfig":
        profile = _normalize_profile(self.profile)
        if self.quick:
            if profile == "oom-safe":
                raise ValueError("Do not combine quick=True with profile='oom-safe'")
            profile = "quick"

        config = replace(self, profile=profile, quick=(profile == "quick"))
        if profile == "quick":
            config = replace(
                config,
                samples_per_client=min(self.samples_per_client, 40),
                batch_size=min(self.batch_size, 8),
                local_epochs=1,
                centralized_epochs=2,
                num_rounds=2,
                num_workers=0,
            )
        elif profile == "oom-safe":
            exp_id = OOM_SAFE_EXP_ID if config.exp_id == DEFAULT_EXP_ID else config.exp_id
            output_dir = (
                OOM_SAFE_OUTPUT_DIR
                if config.output_dir == DEFAULT_OUTPUT_DIR
                else config.output_dir
            )
            config = replace(
                config,
                exp_id=exp_id,
                output_dir=output_dir,
                num_clients=min(config.num_clients, 3),
                samples_per_client=min(config.samples_per_client, 40),
                batch_size=min(config.batch_size, 4),
                local_epochs=1,
                centralized_epochs=1,
                num_rounds=1,
                num_workers=0,
                client_num_cpus=max(config.client_num_cpus, 1.0),
                ray_num_cpus=1 if config.ray_num_cpus is None else config.ray_num_cpus,
            )
        _validate_config(config)
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
    profile: str = "default",
    partition: str = "non_iid",
    quick: bool = False,
    output_dir: str | None = None,
    seed: int = 2026,
) -> DemoConfig:
    values: dict[str, Any] = {
        "profile": profile,
        "partition": partition,
        "quick": quick,
        "seed": seed,
    }
    if output_dir is not None:
        values["output_dir"] = output_dir
    return DemoConfig(
        **values,
    ).normalized()


def _normalize_profile(profile: str) -> str:
    normalized = profile.strip().lower().replace("_", "-")
    if normalized not in VALID_PROFILES:
        choices = ", ".join(sorted(VALID_PROFILES))
        raise ValueError(f"Unsupported profile '{profile}'. Expected one of: {choices}")
    return normalized


def _validate_config(config: DemoConfig) -> None:
    if config.num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if config.samples_per_client < 2:
        raise ValueError("samples_per_client must be >= 2")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.local_epochs < 1:
        raise ValueError("local_epochs must be >= 1")
    if config.centralized_epochs < 1:
        raise ValueError("centralized_epochs must be >= 1")
    if config.num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if config.head_hidden_dim is not None and config.head_hidden_dim < 1:
        raise ValueError("head_hidden_dim must be >= 1 when set")
    if config.client_num_cpus <= 0:
        raise ValueError("client_num_cpus must be > 0")
    if config.ray_num_cpus is not None and config.ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be >= 1 when set")


def _python_key(key: str) -> str:
    return key.replace("-", "_")
