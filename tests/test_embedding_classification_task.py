"""Tests for ``EmbeddingClassificationTask`` -- the second FederatedTask workload.

Everything runs on small in-memory embedding tensors: no images, GPU, Flower
network, or artifact files. Proves the task satisfies ``FederatedTask`` and can be
driven through a full build -> set -> get -> train -> evaluate cycle.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.fl.embedding_classification_task as ect
from src.data.embedding import EmbeddingClientDataset, EmbeddingDatasetBundle
from src.fl.embedding_classification_task import (
    EmbeddingClassificationTask,
    EmbeddingClientContext,
)
from src.fl.task import FederatedTask, RoundOutput
from src.utils.config import DemoConfig


def _bundle(*, embedding_dim=8, num_classes=3, n_train=12, n_val=6, seed=0):
    rng = np.random.default_rng(seed)
    client = EmbeddingClientDataset(
        client_id=0,
        client_label="site-a",
        train_x=torch.tensor(rng.standard_normal((n_train, embedding_dim)), dtype=torch.float32),
        train_y=torch.tensor(rng.integers(0, num_classes, n_train), dtype=torch.long),
        val_x=torch.tensor(rng.standard_normal((n_val, embedding_dim)), dtype=torch.float32),
        val_y=torch.tensor(rng.integers(0, num_classes, n_val), dtype=torch.long),
        label_histogram={},
    )
    pooled = SimpleNamespace(
        train_x=client.train_x, train_y=client.train_y, val_x=client.val_x, val_y=client.val_y
    )
    return EmbeddingDatasetBundle(
        clients=[client],
        pooled=pooled,
        label_mapping={str(i): i for i in range(num_classes)},
        artifact_path="",
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )


def _ctx(**overrides):
    bundle = _bundle(**{k: v for k, v in overrides.items() if k in {"embedding_dim", "num_classes"}})
    config = DemoConfig(
        embedding_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
        local_epochs=2,
        batch_size=4,
        lr=0.05,
        seed=7,
    )
    return EmbeddingClientContext(config=config, bundle=bundle, client=bundle.clients[0])


def test_satisfies_federated_task_protocol():
    assert isinstance(EmbeddingClassificationTask(), FederatedTask)


def test_build_model_respects_num_classes_and_arrays_round_trip():
    task = EmbeddingClassificationTask()
    ctx = _ctx()
    model = task.build_model(ctx.config, num_classes=ctx.bundle.num_classes)

    # linear head: [weight, bias]; out_features == num_classes
    arrays = task.get_global_arrays(model)
    assert len(arrays) == 2
    assert arrays[0].shape == (ctx.bundle.num_classes, ctx.config.embedding_dim)

    perturbed = [a + 1.0 for a in arrays]
    task.set_global_arrays(model, perturbed)
    reloaded = task.get_global_arrays(model)
    for expected, actual in zip(perturbed, reloaded):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_build_model_hidden_dim_makes_mlp_head():
    task = EmbeddingClassificationTask()
    config = DemoConfig(embedding_dim=8, num_classes=3, head_hidden_dim=5, seed=1)
    model = task.build_model(config, num_classes=3)
    # 2-layer MLP head -> 4 param tensors (w,b,w,b) instead of 2
    assert len(task.get_global_arrays(model)) == 4


def test_train_round_returns_expected_metrics_and_count():
    task = EmbeddingClassificationTask()
    ctx = _ctx()
    model = task.build_model(ctx.config, num_classes=ctx.bundle.num_classes)

    out = task.train_round(model, ctx)

    assert isinstance(out, RoundOutput)
    assert out.num_examples == int(ctx.client.train_y.numel())
    assert set(out.metrics) == {"train_loss", "train_accuracy", "train_macro_f1"}
    assert all(isinstance(v, float) for v in out.metrics.values())


def test_evaluate_round_returns_expected_metrics_and_count():
    task = EmbeddingClassificationTask()
    ctx = _ctx()
    model = task.build_model(ctx.config, num_classes=ctx.bundle.num_classes)

    out = task.evaluate_round(model, ctx)

    assert out.num_examples == int(ctx.client.val_y.numel())
    assert set(out.metrics) == {"loss", "accuracy", "macro_f1"}
    assert 0.0 <= out.metrics["accuracy"] <= 1.0


# --- load_client_context (no artifact file) --------------------------------


def test_load_client_context_composes_config_bundle_client(monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(ect, "load_embedding_dataset_bundle", lambda path: bundle)
    context = SimpleNamespace(
        run_config={"embedding_dim": 8, "num_classes": 3},
        node_config={"embedding-artifact-path": "ignored.npz"},
        node_id=0,
    )

    ctx = EmbeddingClassificationTask().load_client_context(context)

    assert isinstance(ctx, EmbeddingClientContext)
    assert ctx.bundle is bundle
    assert ctx.client is bundle.clients[0]


def test_load_client_context_missing_artifact_path_raises():
    context = SimpleNamespace(run_config={}, node_config={}, node_id=0)
    with pytest.raises(ValueError, match="artifact path not found"):
        EmbeddingClassificationTask().load_client_context(context)


def test_select_embedding_client_uses_loaded_bundle_size_not_configured_num_clients():
    bundle = _bundle()
    client_b = EmbeddingClientDataset(
        client_id=1,
        client_label="site-b",
        train_x=bundle.clients[0].train_x,
        train_y=bundle.clients[0].train_y,
        val_x=bundle.clients[0].val_x,
        val_y=bundle.clients[0].val_y,
        label_histogram={},
    )
    bundle.clients.append(client_b)
    context = SimpleNamespace(node_config={"partition-id": 3}, node_id=3)

    selected = ect._select_embedding_client(context, bundle, num_clients=5)

    assert selected.client_label == "site-b"
