import json

import pytest

from src.fl.edge_profile import (
    REASON_DROPPED,
    REASON_UNAVAILABLE,
    EdgeProfile,
    communication_time_seconds,
    edge_decision,
    parse_edge_profile,
    resolve_edge_profile,
)


def test_builtin_profiles_parse():
    assert parse_edge_profile("fast").tier == "fast-gpu"
    assert parse_edge_profile("slow").batch_size == 1
    assert parse_edge_profile("low-bandwidth").bandwidth_mbps == 3.0


def test_json_profile_parses_and_validates():
    profile = parse_edge_profile(
        json.dumps(
            {
                "tier": "tiny",
                "batch-size": 1,
                "max_train_samples": 3,
                "availability_prob": 0.75,
                "latency_ms": 25,
            }
        )
    )
    assert profile.tier == "tiny"
    assert profile.batch_size == 1
    assert profile.max_train_samples == 3
    assert profile.availability_prob == 0.75
    assert profile.latency_ms == 25


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"availability_prob": 1.1}, "availability_prob"),
        ({"dropout_prob": -0.1}, "dropout_prob"),
        ({"artificial_train_delay_sec": -1}, "artificial_train_delay_sec"),
        ({"bandwidth_mbps": 0}, "bandwidth_mbps"),
        ({"unknown": 1}, "Unknown edge profile fields"),
    ],
)
def test_invalid_profiles_raise(payload, match):
    with pytest.raises(ValueError, match=match):
        parse_edge_profile(json.dumps(payload))


def test_resolve_edge_profile_mapping_by_label_id_and_default():
    mapping = json.dumps(
        {
            "site-b": "slow",
            "2": {"tier": "id-profile", "batch_size": 1},
            "default": "fast",
        }
    )

    assert (
        resolve_edge_profile(
            edge_profile=None,
            edge_profiles=mapping,
            client_label="site-b",
            client_id=1,
        ).tier
        == "slow-edge"
    )
    assert (
        resolve_edge_profile(
            edge_profile=None,
            edge_profiles=mapping,
            client_label="site-c",
            client_id=2,
        ).tier
        == "id-profile"
    )
    assert (
        resolve_edge_profile(
            edge_profile=None,
            edge_profiles=mapping,
            client_label="site-z",
            client_id=9,
        ).tier
        == "fast-gpu"
    )


def test_deterministic_decisions_are_stable():
    profile = EdgeProfile(availability_prob=0.5, dropout_prob=0.5, dropout_stage="both")
    first = edge_decision(
        profile, seed=2026, client_id=1, round_number=2, stage="train"
    )
    second = edge_decision(
        profile, seed=2026, client_id=1, round_number=2, stage="train"
    )
    assert first == second


def test_availability_and_dropout_can_be_forced():
    unavailable = edge_decision(
        EdgeProfile(availability_prob=0.0),
        seed=1,
        client_id=1,
        round_number=1,
        stage="train",
    )
    assert unavailable.reason_code == REASON_UNAVAILABLE
    assert unavailable.should_run is False

    dropped = edge_decision(
        EdgeProfile(dropout_prob=1.0, dropout_stage="train"),
        seed=1,
        client_id=1,
        round_number=1,
        stage="train",
    )
    assert dropped.reason_code == REASON_DROPPED
    assert dropped.should_run is False


def test_communication_time_estimate():
    profile = EdgeProfile(latency_ms=100, bandwidth_mbps=10)
    assert communication_time_seconds(1_000_000, profile) == pytest.approx(0.9)
    assert communication_time_seconds(1_000_000, None) == 0.0
