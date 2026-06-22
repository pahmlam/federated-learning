import pytest

from src.utils.config import build_config


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
