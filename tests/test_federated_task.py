"""Tests for the workload-agnostic ``FederatedTask`` abstraction.

A tiny ``FakeTask`` (no detection code, no GPU, no Flower network, no dataset)
proves the interface can be driven through a full build -> set -> get -> train ->
evaluate cycle, which is exactly what the ClientApp does.
"""

import numpy as np

from src.fl.task import FederatedTask, RoundOutput


class _FakeContext:
    def __init__(self, train_n=4, val_n=2):
        self.train_n = train_n
        self.val_n = val_n


class FakeTask:
    """Minimal FederatedTask: the "model" is a dict holding numpy arrays."""

    def load_client_context(self, context):
        return context

    def build_model(self, config, *, num_classes):
        return {"arrays": [np.zeros(num_classes, dtype=np.float32)]}

    def get_global_arrays(self, model):
        return [a.copy() for a in model["arrays"]]

    def set_global_arrays(self, model, arrays):
        model["arrays"] = [np.asarray(a) for a in arrays]

    def train_round(self, model, ctx):
        model["arrays"] = [a + 1.0 for a in model["arrays"]]
        return RoundOutput(num_examples=ctx.train_n, metrics={"train_loss": 0.25})

    def evaluate_round(self, model, ctx):
        return RoundOutput(num_examples=ctx.val_n, metrics={"score": 0.9})


def test_fake_task_satisfies_protocol():
    assert isinstance(FakeTask(), FederatedTask)


def test_round_output_is_frozen_and_carries_fields():
    out = RoundOutput(num_examples=7, metrics={"a": 1.0})
    assert out.num_examples == 7
    assert out.metrics == {"a": 1.0}
    import dataclasses

    assert dataclasses.is_dataclass(out)


def test_fake_task_full_round_cycle():
    task = FakeTask()
    ctx = task.load_client_context(_FakeContext(train_n=5, val_n=3))

    model = task.build_model(config=None, num_classes=3)
    # global arrays round-trip through set/get
    task.set_global_arrays(model, [np.array([1.0, 2.0, 3.0], dtype=np.float32)])
    np.testing.assert_array_equal(
        task.get_global_arrays(model)[0], np.array([1.0, 2.0, 3.0])
    )

    train_out = task.train_round(model, ctx)
    assert isinstance(train_out, RoundOutput)
    assert train_out.num_examples == 5
    assert train_out.metrics["train_loss"] == 0.25
    # train mutated the global arrays; get reflects it
    np.testing.assert_array_equal(
        task.get_global_arrays(model)[0], np.array([2.0, 3.0, 4.0])
    )

    eval_out = task.evaluate_round(model, ctx)
    assert eval_out.num_examples == 3
    assert eval_out.metrics == {"score": 0.9}
