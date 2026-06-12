import numpy as np

from src.evaluation.metrics import parameter_bytes


def test_parameter_bytes_counts_numpy_array_payloads():
    params = [
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
    ]

    assert parameter_bytes(params) == (3 * 4 + 3) * 4
