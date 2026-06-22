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

    metric_names = sorted(
        {
            key
            for record in records
            for key in record
            if key not in {"client_id", "num_examples"}
            and isinstance(record[key], (float, int))
        }
    )
    return {
        metric_name: float(
            sum(
                float(record[metric_name]) * int(record["num_examples"])
                for record in records
                if metric_name in record
            )
            / total_examples
        )
        for metric_name in metric_names
    }
