"""Metrics and model update accounting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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


def aggregate_confusion_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    matrices = [
        record["confusion_matrix"]
        for record in records
        if "confusion_matrix" in record
    ]
    if not matrices:
        return {}

    num_classes = len(matrices[0])
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for matrix in matrices:
        if len(matrix) != num_classes:
            raise ValueError("All confusion matrices must have the same class count")
        for row_index, row in enumerate(matrix):
            if len(row) != num_classes:
                raise ValueError("Confusion matrix must be square")
            for col_index, value in enumerate(row):
                confusion[row_index][col_index] += int(value)

    per_class = {
        str(class_id): _metrics_for_class(confusion, class_id)
        for class_id in range(num_classes)
    }
    return {
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": float(
            sum(metrics["f1"] for metrics in per_class.values()) / num_classes
        ),
    }


def _metrics_for_class(
    confusion: list[list[int]],
    class_id: int,
) -> dict[str, float]:
    tp = confusion[class_id][class_id]
    fp = sum(row[class_id] for index, row in enumerate(confusion) if index != class_id)
    fn = sum(value for index, value in enumerate(confusion[class_id]) if index != class_id)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
