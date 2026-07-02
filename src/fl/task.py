"""Minimal task abstraction for the federated learning system.

The Flower ClientApp/ServerApp orchestration should not hard-code a single
workload. ``FederatedTask`` captures exactly what that orchestration needs from a
workload -- build/load per-site data, build a model, get/set the global (trainable)
arrays that FedAvg aggregates, and run one client train/evaluate round -- so a new
workload (e.g. face recognition) can plug into the same machinery by implementing
this interface. ``DetectionTask`` (``src/fl/detection_task.py``) is the first, and
currently only, implementation; it wraps the existing detection functions.

This is deliberately a plain ``Protocol`` (structural typing) with one concrete
implementation -- no plugin registry or dynamic import layer until one is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class RoundOutput:
    """Result of one client train/evaluate round: a sample count + scalar metrics.

    ``num_examples`` is the FedAvg weighting count; ``metrics`` are JSON/float-ready
    scalars the ClientApp packs into its reply ``MetricRecord`` (alongside
    ``num-examples``).
    """

    num_examples: int
    metrics: dict[str, float]


@runtime_checkable
class FederatedTask(Protocol):
    """What the FL orchestration needs from a workload, and nothing more."""

    def load_client_context(self, context: Any) -> Any:
        """Resolve config + per-site data for this node into a task context."""

    def build_model(self, config: Any, *, num_classes: int) -> Any:
        """Build the model. ``num_classes`` is explicit so the client can pass the
        data-derived count and the server the configured count."""

    def get_global_arrays(self, model: Any) -> list[np.ndarray]:
        """Return the trainable/global arrays FedAvg aggregates."""

    def set_global_arrays(self, model: Any, arrays: list[np.ndarray]) -> None:
        """Load aggregated global arrays back into the model."""

    def train_round(self, model: Any, ctx: Any) -> RoundOutput:
        """Train one local round; return sample count + metrics."""

    def evaluate_round(self, model: Any, ctx: Any) -> RoundOutput:
        """Evaluate one local round; return sample count + metrics."""
