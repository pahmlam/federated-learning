import pytest

from src.utils.detection_config import (
    NUM_DETECTION_CLASSES,
    PPE_CORE_CLASSES,
    DetectionConfig,
    ppe_index_to_label,
    ppe_label_to_index,
)


def test_num_detection_classes_is_eight_ppe_plus_background():
    assert len(PPE_CORE_CLASSES) == 8
    assert NUM_DETECTION_CLASSES == 9
    assert DetectionConfig().num_classes == 9


def test_ppe_label_to_index_is_one_based_with_background_reserved():
    label_map = ppe_label_to_index()
    assert label_map["helmet"] == 1
    assert label_map["face-guard"] == 8
    assert 0 not in label_map.values()  # 0 reserved for background
    assert len(label_map) == 8


def test_ppe_index_to_label_is_symmetric_with_label_map():
    label_map = ppe_label_to_index()
    index_map = ppe_index_to_label()
    assert index_map[label_map["helmet"]] == "helmet"
    assert index_map[label_map["face-guard"]] == "face-guard"
    assert 0 not in index_map


def test_from_run_config_maps_dashed_keys_and_ignores_unknown():
    config = DetectionConfig.from_run_config(
        {"num-rounds": 5, "image-size": 320, "unknown-key": 99}
    )
    assert config.num_rounds == 5
    assert config.image_size == 320


def test_from_run_config_accepts_client_id():
    config = DetectionConfig.from_run_config({"client-id": "site-b"})
    assert config.client_id == "site-b"


def test_from_run_config_accepts_edge_profile_strings():
    config = DetectionConfig.from_run_config(
        {
            "edge-profile": "slow",
            "edge-profiles": '{"site-b":"low-bandwidth"}',
        }
    )
    assert config.edge_profile == "slow"
    assert config.edge_profiles == '{"site-b":"low-bandwidth"}'


def test_blank_edge_profile_strings_normalize_to_none():
    config = DetectionConfig.from_run_config({"edge-profile": "", "edge-profiles": ""})
    assert config.edge_profile is None
    assert config.edge_profiles is None


def test_normalized_rejects_too_small_image_size():
    with pytest.raises(ValueError, match="image_size must be >= 64"):
        DetectionConfig(image_size=32).normalized()


def test_normalized_rejects_bad_device():
    with pytest.raises(ValueError, match="device must be one of"):
        DetectionConfig(device="tpu").normalized()


def test_normalized_rejects_out_of_range_score_threshold():
    with pytest.raises(ValueError, match="score_threshold"):
        DetectionConfig(score_threshold=1.0).normalized()


# --- min-node robustness knobs --------------------------------------------


def test_min_nodes_default_to_strict_num_clients():
    config = DetectionConfig(num_clients=2)
    assert config.min_train_nodes is None
    assert config.min_evaluate_nodes is None
    assert config.min_available_nodes is None
    assert config.effective_min_train_nodes == 2
    assert config.effective_min_evaluate_nodes == 2
    assert config.effective_min_available_nodes == 2


def test_min_nodes_explicit_overrides_are_used():
    config = DetectionConfig(
        num_clients=2,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1,
    )
    assert config.effective_min_train_nodes == 1
    assert config.effective_min_evaluate_nodes == 1
    assert config.effective_min_available_nodes == 1


def test_from_run_config_maps_dashed_min_node_keys():
    config = DetectionConfig.from_run_config(
        {
            "num-clients": 2,
            "min-train-nodes": 1,
            "min-evaluate-nodes": 1,
            "min-available-nodes": 1,
        }
    )
    assert config.min_train_nodes == 1
    assert config.min_evaluate_nodes == 1
    assert config.min_available_nodes == 1
    assert config.effective_min_train_nodes == 1


def test_normalized_rejects_min_nodes_below_one():
    with pytest.raises(ValueError, match="min_train_nodes must be >= 1"):
        DetectionConfig(min_train_nodes=0).normalized()


def test_normalized_rejects_min_nodes_above_num_clients():
    with pytest.raises(ValueError, match="min_train_nodes must be <= num_clients"):
        DetectionConfig(num_clients=2, min_train_nodes=3).normalized()


def test_normalized_rejects_available_nodes_below_train_or_eval_minimum():
    with pytest.raises(ValueError, match="min_available_nodes must be >= min_train_nodes"):
        DetectionConfig(
            num_clients=2,
            min_train_nodes=2,
            min_available_nodes=1,
        ).normalized()
    with pytest.raises(ValueError, match="min_available_nodes must be >= min_evaluate_nodes"):
        DetectionConfig(
            num_clients=2,
            min_train_nodes=1,
            min_evaluate_nodes=2,
            min_available_nodes=1,
        ).normalized()
