"""Metrics and model update accounting."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def accuracy_from_counts(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(correct / total)


def parameter_bytes(parameters: Iterable[np.ndarray]) -> int:
    return int(sum(param.nbytes for param in parameters))


def weighted_average_metric(records: list[dict[str, float | int]]) -> dict[str, float]:
    total_examples = sum(int(record["num_examples"]) for record in records)
    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    loss = sum(float(record["loss"]) * int(record["num_examples"]) for record in records)
    accuracy = sum(
        float(record["accuracy"]) * int(record["num_examples"]) for record in records
    )
    return {
        "loss": float(loss / total_examples),
        "accuracy": float(accuracy / total_examples),
    }
