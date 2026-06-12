import numpy as np

from src.models.head_model import (
    create_model,
    get_head_parameters,
    set_head_parameters,
    trainable_parameter_names,
)
from src.utils.config import DemoConfig


def test_model_freezes_backbone_and_trains_only_head():
    model = create_model(DemoConfig())

    assert trainable_parameter_names(model) == ["head.weight", "head.bias"]
    assert all(not param.requires_grad for param in model.backbone.parameters())
    assert all(param.requires_grad for param in model.head.parameters())


def test_head_parameter_round_trip():
    model = create_model(DemoConfig())
    params = get_head_parameters(model)
    changed = [param + 1.0 for param in params]

    set_head_parameters(model, changed)
    round_tripped = get_head_parameters(model)

    assert len(round_tripped) == len(changed)
    for actual, expected in zip(round_tripped, changed):
        np.testing.assert_allclose(actual, expected)
