"""Tests for ``DetectionTask`` -- the detection ``FederatedTask`` implementation.

These prove the abstraction preserves the previous inline ClientApp behavior:
the exact config -> trainer kwarg mapping, the head get/set path, and the metric
keys reported back. Heavy training is faked (monkeypatched) so no GPU, Flower
network, or real dataset is needed; ``pretrained=False`` avoids weight downloads.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import src.fl.detection_task as dt
from src.fl.edge_profile import EdgeProfile
from src.fl.detection_task import DetectionTask, DetectionTaskContext
from src.fl.task import RoundOutput
from src.models.detection_model import resolve_device
from src.utils.detection_config import DetectionConfig, NUM_DETECTION_CLASSES


def _config(**overrides):
    base = dict(pretrained=False, image_size=64, device="cpu")
    base.update(overrides)
    return DetectionConfig(**base)


# --- build_model + get/set global arrays round-trip (real model, no training) --


def test_build_model_and_global_arrays_round_trip():
    task = DetectionTask()
    config = _config(seed=7)
    model = task.build_model(config, num_classes=NUM_DETECTION_CLASSES)

    arrays = task.get_global_arrays(model)
    assert arrays and all(isinstance(a, np.ndarray) for a in arrays)

    # set a perturbed copy, then get returns the same values with stable shapes.
    perturbed = [a + 1.0 for a in arrays]
    task.set_global_arrays(model, perturbed)
    reloaded = task.get_global_arrays(model)
    assert [a.shape for a in reloaded] == [a.shape for a in arrays]
    for expected, actual in zip(perturbed, reloaded):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_build_model_uses_requested_num_classes():
    task = DetectionTask()
    model = task.build_model(_config(), num_classes=NUM_DETECTION_CLASSES)
    # Faster R-CNN box predictor cls_score has one logit per class.
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_DETECTION_CLASSES


# --- train_round wiring: exact config -> trainer kwargs --------------------


def test_train_round_maps_config_to_trainer_kwargs(monkeypatch):
    captured = {}

    def _fake_train(model, dataset, **kwargs):
        captured["model"] = model
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return {"train_loss": 0.5}

    monkeypatch.setattr(dt, "train_detection_head", _fake_train)

    config = _config(
        local_epochs=3,
        batch_size=4,
        lr=0.01,
        momentum=0.8,
        weight_decay=1e-3,
        num_workers=2,
        device="cpu",
        seed=100,
    )
    client = SimpleNamespace(client_id=2, train=list(range(9)), val=list(range(4)))
    ctx = DetectionTaskContext(config=config, bundle=SimpleNamespace(num_classes=9), client=client)

    out = DetectionTask().train_round(model="MODEL", ctx=ctx)

    assert captured["model"] == "MODEL"
    assert captured["dataset"] is client.train
    kw = captured["kwargs"]
    assert kw["epochs"] == 3
    assert kw["batch_size"] == 4
    assert kw["lr"] == 0.01
    assert kw["momentum"] == 0.8
    assert kw["weight_decay"] == 1e-3
    assert kw["num_workers"] == 2
    assert kw["device"] == resolve_device("cpu")
    assert kw["seed"] == config.seed + client.client_id  # per-client seed offset

    assert out == RoundOutput(num_examples=9, metrics={"train_loss": 0.5})


def test_train_round_applies_edge_profile_sample_limit_and_delay(monkeypatch):
    captured = {}
    sleeps = []

    def _fake_train(model, dataset, **kwargs):
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return {"train_loss": 0.25}

    monkeypatch.setattr(dt, "train_detection_head", _fake_train)
    monkeypatch.setattr(dt.time, "sleep", lambda seconds: sleeps.append(seconds))

    config = _config(batch_size=1, num_workers=0, seed=3)
    client = SimpleNamespace(
        client_id=2,
        client_label="site-b",
        train=list(range(9)),
        val=list(range(4)),
    )
    ctx = DetectionTaskContext(
        config=config,
        bundle=SimpleNamespace(num_classes=9),
        client=client,
        edge_profile=EdgeProfile(
            tier="slow-edge",
            max_train_samples=3,
            artificial_train_delay_sec=0.5,
            bandwidth_mbps=10,
            latency_ms=100,
        ),
        update_size_bytes=1_000_000,
    )

    out = DetectionTask().train_round(model="MODEL", ctx=ctx)

    assert len(captured["dataset"]) == 3
    assert captured["kwargs"]["batch_size"] == 1
    assert sleeps == [0.5]
    assert out.num_examples == 3
    assert out.metrics["train_loss"] == 0.25
    assert out.metrics["edge_profile_enabled"] == 1.0
    assert out.metrics["expected_transfer_time"] == pytest.approx(0.9)
    assert "effective_train_time" in out.metrics


def test_train_round_edge_profile_can_block_client():
    config = _config()
    client = SimpleNamespace(
        client_id=0,
        client_label="site-a",
        train=[0],
        val=[0],
    )
    ctx = DetectionTaskContext(
        config=config,
        bundle=SimpleNamespace(num_classes=9),
        client=client,
        edge_profile=EdgeProfile(availability_prob=0.0),
    )

    with pytest.raises(RuntimeError, match="EdgeProfile blocked client site-a"):
        DetectionTask().train_round(model="MODEL", ctx=ctx)


# --- evaluate_round wiring: metric keys + fallback -------------------------


def test_evaluate_round_reports_map_keys_and_drops_extras(monkeypatch):
    def _fake_eval(model, dataset, **kwargs):
        _fake_eval.kwargs = kwargs
        # includes an extra key that must be dropped from the reply
        return {"map": 0.4, "map_50": 0.6, "map_75": 0.2, "mar_100": 0.9}

    monkeypatch.setattr(dt, "evaluate_detection", _fake_eval)

    config = _config(batch_size=2, num_workers=1, score_threshold=0.3, device="cpu")
    client = SimpleNamespace(client_id=1, train=list(range(5)), val=list(range(3)))
    ctx = DetectionTaskContext(config=config, bundle=SimpleNamespace(num_classes=9), client=client)

    out = DetectionTask().evaluate_round(model="MODEL", ctx=ctx)

    assert _fake_eval.kwargs["batch_size"] == 2
    assert _fake_eval.kwargs["num_workers"] == 1
    assert _fake_eval.kwargs["score_threshold"] == 0.3
    assert _fake_eval.kwargs["device"] == resolve_device("cpu")
    assert out.num_examples == 3
    assert out.metrics == {"map": 0.4, "map_50": 0.6, "map_75": 0.2}


def test_evaluate_round_uses_minus_one_for_missing_metrics(monkeypatch):
    monkeypatch.setattr(dt, "evaluate_detection", lambda *a, **k: {"map": 0.5})
    config = _config()
    client = SimpleNamespace(client_id=0, train=[0], val=[0, 1])
    ctx = DetectionTaskContext(config=config, bundle=SimpleNamespace(num_classes=9), client=client)

    out = DetectionTask().evaluate_round(model="M", ctx=ctx)
    assert out.metrics == {"map": 0.5, "map_50": -1.0, "map_75": -1.0}


# --- load_client_context composition (no real dataset) --------------------


def test_load_client_context_composes_config_bundle_client(monkeypatch):
    config = _config()
    bundle = SimpleNamespace(num_classes=9, clients=["c0"], image_size=64)
    selected = SimpleNamespace(
        client_id=0,
        client_label="site-a",
        train=list(range(2)),
        val=list(range(1)),
    )
    monkeypatch.setattr(dt, "detection_config_from_context", lambda context: config)
    monkeypatch.setattr(dt, "load_detection_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(dt, "select_detection_client", lambda context, b, **k: selected)

    ctx = DetectionTask().load_client_context(context=object())

    assert isinstance(ctx, DetectionTaskContext)
    assert ctx.config is config
    assert ctx.bundle is bundle
    assert ctx.client is selected
