import pytest

from src.utils.config import DemoConfig, build_config


def test_quick_profile_matches_legacy_quick_alias():
    from_profile = build_config(profile="quick")
    from_alias = build_config(quick=True)

    assert from_profile.profile == "quick"
    assert from_alias.profile == "quick"
    assert from_profile.quick is True
    assert from_alias.quick is True
    assert from_profile.samples_per_client == from_alias.samples_per_client
    assert from_profile.batch_size == from_alias.batch_size
    assert from_profile.num_rounds == from_alias.num_rounds


def test_oom_safe_profile_uses_small_single_machine_defaults():
    config = build_config(profile="oom-safe")

    assert config.exp_id == "EXP-002"
    assert config.output_dir == "outputs/EXP-002"
    assert config.profile == "oom-safe"
    assert config.num_clients == 3
    assert config.samples_per_client == 40
    assert config.batch_size == 4
    assert config.local_epochs == 1
    assert config.centralized_epochs == 1
    assert config.num_rounds == 1
    assert config.num_workers == 0
    assert config.ray_num_cpus == 1


def test_oom_safe_profile_respects_output_override():
    config = build_config(profile="oom-safe", output_dir="outputs/CUSTOM")

    assert config.exp_id == "EXP-002"
    assert config.output_dir == "outputs/CUSTOM"


def test_invalid_profile_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported profile"):
        build_config(profile="large")


def test_weight_decay_defaults_to_zero():
    assert DemoConfig().weight_decay == 0.0
    assert DemoConfig().normalized().weight_decay == 0.0


def test_weight_decay_is_preserved_under_oom_safe_profile():
    config = DemoConfig(profile="oom-safe", weight_decay=0.01).normalized()
    assert config.weight_decay == 0.01


def test_negative_weight_decay_raises_value_error():
    with pytest.raises(ValueError, match="weight_decay must be >= 0"):
        DemoConfig(weight_decay=-0.1).normalized()


def test_capacity_knobs_default_off():
    config = DemoConfig()
    assert config.normalize_embedding is False
    assert config.head_hidden_dim is None
    normalized = config.normalized()
    assert normalized.normalize_embedding is False
    assert normalized.head_hidden_dim is None


def test_capacity_knobs_preserved_under_oom_safe_profile():
    config = DemoConfig(
        profile="oom-safe",
        normalize_embedding=True,
        head_hidden_dim=64,
    ).normalized()
    assert config.normalize_embedding is True
    assert config.head_hidden_dim == 64


def test_zero_head_hidden_dim_raises_value_error():
    with pytest.raises(ValueError, match="head_hidden_dim must be >= 1"):
        DemoConfig(head_hidden_dim=0).normalized()
