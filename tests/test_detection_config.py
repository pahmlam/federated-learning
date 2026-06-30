import pytest

from src.utils.detection_config import (
    NUM_DETECTION_CLASSES,
    PPE_CORE_CLASSES,
    DetectionConfig,
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


def test_from_run_config_maps_dashed_keys_and_ignores_unknown():
    config = DetectionConfig.from_run_config(
        {"num-rounds": 5, "image-size": 320, "unknown-key": 99}
    )
    assert config.num_rounds == 5
    assert config.image_size == 320


def test_from_run_config_accepts_client_id():
    config = DetectionConfig.from_run_config({"client-id": "site-b"})
    assert config.client_id == "site-b"


def test_normalized_rejects_too_small_image_size():
    with pytest.raises(ValueError, match="image_size must be >= 64"):
        DetectionConfig(image_size=32).normalized()


def test_normalized_rejects_bad_device():
    with pytest.raises(ValueError, match="device must be one of"):
        DetectionConfig(device="tpu").normalized()


def test_normalized_rejects_out_of_range_score_threshold():
    with pytest.raises(ValueError, match="score_threshold"):
        DetectionConfig(score_threshold=1.0).normalized()
