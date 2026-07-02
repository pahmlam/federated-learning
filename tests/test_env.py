import pytest

from src.utils.detection_config import DetectionConfig
from src.utils.env import detection_env_values, fl_env_values


def _supported_env_names():
    base = (
        "RUN_ID",
        "OUTPUT_DIR",
        "MANIFEST_PATH",
        "DATA_ROOT",
        "CLIENT_ID",
        "NUM_CLIENTS",
        "NUM_CLASSES",
        "IMAGE_SIZE",
        "BATCH_SIZE",
        "LOCAL_EPOCHS",
        "CENTRALIZED_EPOCHS",
        "NUM_ROUNDS",
        "LR",
        "MOMENTUM",
        "WEIGHT_DECAY",
        "NUM_WORKERS",
        "SCORE_THRESHOLD",
        "DEVICE",
        "PRETRAINED",
        "SEED",
        "MIN_TRAIN_NODES",
        "MIN_EVALUATE_NODES",
        "MIN_AVAILABLE_NODES",
        "EDGE_PROFILE",
        "EDGE_PROFILES",
    )
    legacy = {
        "RUN_ID": "EXP_ID",
        "DATA_ROOT": "ROOT_DIR",
    }
    names = []
    for suffix in base:
        names.append(f"FL_{suffix}")
        names.append(f"FL_DET_{legacy.get(suffix, suffix)}")
    return names


@pytest.fixture(autouse=True)
def _clear_supported_env(monkeypatch):
    for name in _supported_env_names():
        monkeypatch.delenv(name, raising=False)


def test_fl_env_values_parse_supported_types(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FL_RUN_ID=run-env",
                "FL_NUM_CLIENTS=7",
                "FL_LR=0.01",
                "FL_PRETRAINED=false",
                "FL_UNKNOWN=ignored",
                "FL_OUTPUT_DIR=",
            ]
        ),
        encoding="utf-8",
    )
    values = fl_env_values(DetectionConfig, path=env_file)

    assert values["exp_id"] == "run-env"
    assert values["num_clients"] == 7
    assert values["lr"] == 0.01
    assert values["pretrained"] is False
    assert "unknown" not in values
    assert "output_dir" not in values


def test_fl_env_values_parse_min_node_overrides_as_int(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FL_MIN_TRAIN_NODES=1",
                "FL_MIN_EVALUATE_NODES=1",
                "FL_MIN_AVAILABLE_NODES=2",
            ]
        ),
        encoding="utf-8",
    )
    values = fl_env_values(DetectionConfig, path=env_file)

    # int | None fields must parse as int, not fall back to str.
    assert values["min_train_nodes"] == 1
    assert values["min_evaluate_nodes"] == 1
    assert values["min_available_nodes"] == 2
    assert all(isinstance(values[key], int) for key in values)


def test_fl_env_values_parse_edge_profile_strings(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'FL_EDGE_PROFILE=slow\nFL_EDGE_PROFILES={"site-b":"low-bandwidth"}\n',
        encoding="utf-8",
    )
    values = fl_env_values(DetectionConfig, path=env_file)

    assert values["edge_profile"] == "slow"
    assert values["edge_profiles"] == '{"site-b":"low-bandwidth"}'


def test_legacy_detection_env_alias_still_works(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FL_DET_EXP_ID=legacy-run\nFL_DET_ROOT_DIR=data/legacy\n",
        encoding="utf-8",
    )
    values = detection_env_values(DetectionConfig, path=env_file)

    assert values["exp_id"] == "legacy-run"
    assert values["root_dir"] == "data/legacy"


def test_system_env_wins_over_legacy_alias(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FL_RUN_ID=system-run",
                "FL_DET_EXP_ID=legacy-run",
                "FL_DATA_ROOT=data/system",
                "FL_DET_ROOT_DIR=data/legacy",
            ]
        ),
        encoding="utf-8",
    )
    values = fl_env_values(DetectionConfig, path=env_file)

    assert values["exp_id"] == "system-run"
    assert values["root_dir"] == "data/system"


def test_fl_env_values_raise_for_invalid_bool(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FL_PRETRAINED=maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid boolean"):
        fl_env_values(DetectionConfig, path=env_file)


def test_fl_env_values_raise_for_invalid_number(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FL_NUM_ROUNDS=lots\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid value for FL_NUM_ROUNDS"):
        fl_env_values(DetectionConfig, path=env_file)


def test_detection_config_merges_defaults_env_and_overrides(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FL_BATCH_SIZE=4\nFL_DEVICE=cpu\n",
        encoding="utf-8",
    )
    config = DetectionConfig.from_env_and_overrides(
        {"batch-size": 8},
        env_path=env_file,
    )

    assert config.batch_size == 8
    assert config.device == "cpu"


def test_detection_config_env_overrides_default_run_config_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FL_DATA_ROOT=data/from-env\n", encoding="utf-8")
    config = DetectionConfig.from_env_and_overrides(
        {"root-dir": DetectionConfig().root_dir},
        env_path=env_file,
        env_overrides=True,
    )

    assert config.root_dir == "data/from-env"


def test_detection_config_keeps_explicit_run_config_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FL_NUM_CLIENTS=3\n", encoding="utf-8")
    config = DetectionConfig.from_env_and_overrides(
        {"num-clients": 2},
        env_path=env_file,
        env_overrides=True,
    )

    assert config.num_clients == 2


def test_detection_config_env_can_override_pyproject_default_min_nodes(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text("FL_MIN_TRAIN_NODES=1\n", encoding="utf-8")
    config = DetectionConfig.from_env_and_overrides(
        {
            "num-clients": 2,
            "min-train-nodes": 2,
            "min-evaluate-nodes": 2,
            "min-available-nodes": 2,
        },
        env_path=env_file,
        env_overrides=True,
    )

    assert config.num_clients == 2
    assert config.min_train_nodes == 1
    assert config.effective_min_evaluate_nodes == 2
    assert config.effective_min_available_nodes == 2


def test_detection_config_env_can_override_blank_pyproject_edge_profile(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FL_EDGE_PROFILE=slow\n", encoding="utf-8")
    config = DetectionConfig.from_env_and_overrides(
        {"edge-profile": ""},
        env_path=env_file,
        env_overrides=True,
    )

    assert config.edge_profile == "slow"


def test_detection_config_keeps_explicit_run_config_min_node_override(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text("FL_MIN_TRAIN_NODES=2\n", encoding="utf-8")
    config = DetectionConfig.from_env_and_overrides(
        {
            "num-clients": 2,
            "min-train-nodes": 1,
            "min-evaluate-nodes": 1,
            "min-available-nodes": 1,
        },
        env_path=env_file,
        env_overrides=True,
    )

    assert config.min_train_nodes == 1
    assert config.min_evaluate_nodes == 1
    assert config.min_available_nodes == 1
