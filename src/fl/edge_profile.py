"""Code-level edge-device emulation for FL clients.

``EdgeProfile`` is intentionally independent from any concrete vision model. It
describes lightweight, deterministic knobs that can make a client behave like a
fast, slow, unreliable, or low-bandwidth edge device while keeping experiments
reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any


VALID_DROPOUT_STAGES = {"none", "train", "evaluate", "both"}
REASON_NONE = 0
REASON_UNAVAILABLE = 1
REASON_DROPPED = 2


@dataclass(frozen=True)
class EdgeProfile:
    tier: str = "custom"
    image_size: int | None = None
    batch_size: int | None = None
    max_train_samples: int | None = None
    num_workers: int | None = None
    availability_prob: float = 1.0
    dropout_prob: float = 0.0
    dropout_stage: str = "none"
    artificial_train_delay_sec: float = 0.0
    latency_ms: float | None = None
    bandwidth_mbps: float | None = None

    def normalized(self) -> "EdgeProfile":
        profile = replace(
            self,
            dropout_stage=self.dropout_stage.strip().lower(),
            tier=str(self.tier).strip() or "custom",
        )
        _validate_edge_profile(profile)
        return profile


@dataclass(frozen=True)
class EdgeDecision:
    available: bool
    dropped: bool
    reason_code: int
    reason: str

    @property
    def should_run(self) -> bool:
        return self.available and not self.dropped


BUILTIN_EDGE_PROFILES: dict[str, EdgeProfile] = {
    "fast": EdgeProfile(tier="fast-gpu"),
    "slow": EdgeProfile(tier="slow-edge", batch_size=1, artificial_train_delay_sec=0.1),
    "unreliable": EdgeProfile(
        tier="unreliable",
        availability_prob=0.8,
        dropout_prob=0.5,
        dropout_stage="both",
    ),
    "low-bandwidth": EdgeProfile(
        tier="low-bandwidth",
        batch_size=1,
        latency_ms=150.0,
        bandwidth_mbps=3.0,
    ),
}


def parse_edge_profile(value: str | dict[str, Any] | EdgeProfile | None) -> EdgeProfile | None:
    """Parse a profile name, JSON object string, mapping, or ``EdgeProfile``."""

    if value is None:
        return None
    if isinstance(value, EdgeProfile):
        return value.normalized()
    if isinstance(value, dict):
        return _profile_from_mapping(value).normalized()

    text = str(value).strip()
    if not text:
        return None
    key = text.lower()
    if key in BUILTIN_EDGE_PROFILES:
        return BUILTIN_EDGE_PROFILES[key].normalized()
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        choices = ", ".join(sorted(BUILTIN_EDGE_PROFILES))
        raise ValueError(
            f"Unknown edge profile {text!r}. Use one of: {choices}, or a JSON object."
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError("edge profile JSON must be an object")
    return _profile_from_mapping(loaded).normalized()


def resolve_edge_profile(
    *,
    edge_profile: str | None,
    edge_profiles: str | None,
    client_label: str,
    client_id: int,
) -> EdgeProfile | None:
    """Resolve per-client profile mapping, falling back to the global profile."""

    if edge_profiles:
        try:
            mapping = json.loads(edge_profiles)
        except json.JSONDecodeError as exc:
            raise ValueError("edge_profiles must be a JSON object mapping clients to profiles") from exc
        if not isinstance(mapping, dict):
            raise ValueError("edge_profiles must be a JSON object mapping clients to profiles")
        for key in (client_label, str(client_id), "default"):
            if key in mapping:
                return parse_edge_profile(mapping[key])
    return parse_edge_profile(edge_profile)


def edge_decision(
    profile: EdgeProfile | None,
    *,
    seed: int,
    client_id: int,
    round_number: int,
    stage: str,
) -> EdgeDecision:
    """Return a deterministic availability/dropout decision for a stage."""

    if profile is None:
        return EdgeDecision(True, False, REASON_NONE, "none")

    availability_draw = _unit_interval(seed, client_id, round_number, stage, "available")
    if availability_draw >= profile.availability_prob:
        return EdgeDecision(False, False, REASON_UNAVAILABLE, "unavailable")

    if _stage_matches(profile.dropout_stage, stage):
        dropout_draw = _unit_interval(seed, client_id, round_number, stage, "dropout")
        if dropout_draw < profile.dropout_prob:
            return EdgeDecision(True, True, REASON_DROPPED, "dropped")

    return EdgeDecision(True, False, REASON_NONE, "none")


def communication_time_seconds(
    update_size_bytes: int,
    profile: EdgeProfile | None,
    *,
    transfers: int = 1,
) -> float:
    """Estimate transfer time from bytes, bandwidth, and latency."""

    if profile is None:
        return 0.0
    latency = 0.0 if profile.latency_ms is None else profile.latency_ms / 1000.0
    if profile.bandwidth_mbps is None:
        return float(transfers) * latency
    transfer = (float(update_size_bytes) * 8.0) / (profile.bandwidth_mbps * 1_000_000.0)
    return float(transfers) * (latency + transfer)


def profile_metrics(
    profile: EdgeProfile | None,
    decision: EdgeDecision,
    *,
    update_size_bytes: int = 0,
) -> dict[str, float]:
    """MetricRecord-safe numeric profile metadata."""

    if profile is None:
        return {}
    expected_transfer = communication_time_seconds(update_size_bytes, profile)
    return {
        "edge_profile_enabled": 1.0,
        "edge_availability_decision": 1.0 if decision.available else 0.0,
        "edge_dropout_decision": 1.0 if decision.dropped else 0.0,
        "edge_dropout_reason_code": float(decision.reason_code),
        "expected_transfer_time": expected_transfer,
        "simulated_network_time": expected_transfer,
    }


def apply_profile_overrides(config: Any, profile: EdgeProfile | None) -> Any:
    """Return config with safe per-client resource overrides applied."""

    if profile is None:
        return config
    updates: dict[str, Any] = {}
    for field in ("image_size", "batch_size", "num_workers"):
        value = getattr(profile, field)
        if value is not None:
            updates[field] = value
    if not updates:
        return config
    return replace(config, **updates).normalized()


def _profile_from_mapping(values: dict[str, Any]) -> EdgeProfile:
    normalized = {str(key).replace("-", "_"): value for key, value in values.items()}
    fields = set(EdgeProfile.__dataclass_fields__)
    unknown = sorted(set(normalized) - fields)
    if unknown:
        raise ValueError(f"Unknown edge profile fields: {unknown}")
    return EdgeProfile(**normalized)


def _validate_edge_profile(profile: EdgeProfile) -> None:
    for field in ("image_size", "batch_size", "max_train_samples"):
        value = getattr(profile, field)
        if value is not None and value < 1:
            raise ValueError(f"{field} must be >= 1")
    if profile.image_size is not None and profile.image_size < 64:
        raise ValueError("image_size must be >= 64")
    if profile.num_workers is not None and profile.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if not 0.0 <= profile.availability_prob <= 1.0:
        raise ValueError("availability_prob must be in [0, 1]")
    if not 0.0 <= profile.dropout_prob <= 1.0:
        raise ValueError("dropout_prob must be in [0, 1]")
    if profile.dropout_stage not in VALID_DROPOUT_STAGES:
        raise ValueError(f"dropout_stage must be one of {sorted(VALID_DROPOUT_STAGES)}")
    if profile.artificial_train_delay_sec < 0:
        raise ValueError("artificial_train_delay_sec must be >= 0")
    if profile.latency_ms is not None and profile.latency_ms < 0:
        raise ValueError("latency_ms must be >= 0")
    if profile.bandwidth_mbps is not None and profile.bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be > 0")


def _stage_matches(dropout_stage: str, stage: str) -> bool:
    normalized_stage = stage.strip().lower()
    return dropout_stage == "both" or dropout_stage == normalized_stage


def _unit_interval(
    seed: int,
    client_id: int,
    round_number: int,
    stage: str,
    salt: str,
) -> float:
    payload = f"{seed}:{client_id}:{round_number}:{stage}:{salt}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    integer = int.from_bytes(digest[:8], "big")
    return integer / float(2**64)
