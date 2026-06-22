import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import parameter_bytes
from src.training.trainer import evaluate_model


class _FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.logits[: inputs.shape[0]]


def test_parameter_bytes_counts_numpy_array_payloads():
    params = [
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
    ]

    assert parameter_bytes(params) == (3 * 4 + 3) * 4


def test_evaluate_model_reports_macro_f1_and_unsafe_metrics():
    model = _FixedLogits(
        [
            [3.0, 1.0],
            [2.0, 1.0],
            [1.0, 3.0],
            [1.0, 2.0],
        ]
    )
    features = torch.zeros((4, 2), dtype=torch.float32)
    labels = torch.tensor([0, 1, 1, 0])

    metrics = evaluate_model(model, features, labels, positive_class_id=1)

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["unsafe_recall"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
