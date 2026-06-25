import numpy as np
import pytest
import torch

from src.models.detection_model import (
    build_detection_model,
    detection_trainable_parameter_names,
    get_detection_head_parameters,
    resolve_device,
    set_detection_head_parameters,
)


@pytest.fixture(scope="module")
def model():
    # pretrained=False avoids any weight download and keeps the test fast.
    return build_detection_model(num_classes=9, pretrained=False, seed=2026)


def test_box_predictor_matches_requested_class_count(model):
    assert model.roi_heads.box_predictor.cls_score.out_features == 9


def test_backbone_is_frozen(model):
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_trainable_head_is_rpn_and_roi_only(model):
    names = detection_trainable_parameter_names(model)
    assert names, "expected a non-empty trainable head"
    assert all(not name.startswith("backbone") for name in names)
    assert any(name.startswith("rpn") for name in names)
    assert any(name.startswith("roi_heads") for name in names)


def test_head_parameter_roundtrip_preserves_values(model):
    original = get_detection_head_parameters(model)
    set_detection_head_parameters(model, original)
    restored = get_detection_head_parameters(model)
    assert len(original) == len(restored)
    for before, after in zip(original, restored):
        assert np.allclose(before, after)


def test_set_head_parameters_changes_weights(model):
    original = get_detection_head_parameters(model)
    zeros = [np.zeros_like(arr) for arr in original]
    set_detection_head_parameters(model, zeros)
    after = get_detection_head_parameters(model)
    assert any(not np.allclose(arr, 0) for arr in original)  # sanity: not already zero
    assert all(np.allclose(arr, 0) for arr in after)
    set_detection_head_parameters(model, original)  # restore for other tests


def test_set_head_parameters_rejects_wrong_count(model):
    with pytest.raises(ValueError, match="Expected .* head parameter arrays"):
        set_detection_head_parameters(model, get_detection_head_parameters(model)[:-1])


def test_resolve_device_explicit_and_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
