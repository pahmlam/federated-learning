"""Tests for post-FL site-side final-head evaluation.

These exercise npz loading by parameter-name order, client selection, report
assembly, the orchestration seam, and the CLI script -- all without a GPU, a
Flower network, or a real dataset (heavy dependencies are faked/monkeypatched).
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.evaluation.final_head_eval as fhe
from src.evaluation.final_head_eval import (
    build_eval_report,
    evaluate_final_head,
    load_head_arrays_for_model,
    select_eval_client,
)
from src.utils.detection_config import DetectionConfig

ROOT = Path(__file__).resolve().parents[1]


def _client(client_id=0, label="site-a", val_len=3):
    return SimpleNamespace(
        client_id=client_id,
        client_label=label,
        val=list(range(val_len)),
    )


# --- npz loading -----------------------------------------------------------


def test_load_head_arrays_by_name_order(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones(4, dtype=np.float32)
    # Store in a different order than requested to prove name-keyed ordering.
    path = tmp_path / "final_head.npz"
    np.savez(path, **{"fc.bias": b, "fc.weight": a})

    arrays = load_head_arrays_for_model(path, ["fc.weight", "fc.bias"])

    assert [arr.shape for arr in arrays] == [(2, 3), (4,)]
    np.testing.assert_array_equal(arrays[0], a)
    np.testing.assert_array_equal(arrays[1], b)


def test_load_head_arrays_missing_name_raises(tmp_path):
    path = tmp_path / "final_head.npz"
    np.savez(path, **{"fc.weight": np.ones(2, dtype=np.float32)})

    with pytest.raises(ValueError, match="missing required head parameter"):
        load_head_arrays_for_model(path, ["fc.weight", "fc.bias"])


def test_load_head_arrays_positional_fallback(tmp_path):
    path = tmp_path / "head.npz"
    np.savez(path, np.zeros(2, dtype=np.float32), np.ones(3, dtype=np.float32))

    arrays = load_head_arrays_for_model(path, ["fc.weight", "fc.bias"])

    assert [arr.shape for arr in arrays] == [(2,), (3,)]


# --- client selection ------------------------------------------------------


def test_select_eval_client_single_client_ignores_id():
    bundle = SimpleNamespace(clients=[_client(label="only")])
    assert select_eval_client(bundle, client_id=None).client_label == "only"


def test_select_eval_client_requires_id_when_multiple():
    bundle = SimpleNamespace(clients=[_client(0, "site-a"), _client(1, "site-b")])
    with pytest.raises(ValueError, match="multiple clients"):
        select_eval_client(bundle, client_id=None)


def test_select_eval_client_matches_label_or_numeric_id():
    bundle = SimpleNamespace(clients=[_client(0, "site-a"), _client(1, "site-b")])
    assert select_eval_client(bundle, "site-b").client_label == "site-b"
    assert select_eval_client(bundle, "0").client_label == "site-a"


def test_select_eval_client_unknown_id_raises():
    bundle = SimpleNamespace(clients=[_client(0, "site-a"), _client(1, "site-b")])
    with pytest.raises(ValueError, match="Unknown detection client_id"):
        select_eval_client(bundle, "site-z")


# --- report assembly -------------------------------------------------------


def test_build_eval_report_contains_expected_keys():
    config = DetectionConfig(pretrained=False)
    metrics = {
        "map": 0.5,
        "map_50": 0.7,
        "map_75": 0.3,
        "map_per_class": [0.4, 0.6],
        "classes": [1, 2],
        "mar_100": 0.9,  # extra key should be ignored
    }
    report = build_eval_report(
        config=config,
        client=_client(1, "site-b"),
        num_examples=12,
        metrics=metrics,
        head_path="outputs/EXP-012/final_head.npz",
        manifest="m.csv",
        root_dir="data/ppe",
    )

    assert report["client_id"] == 1
    assert report["client_label"] == "site-b"
    assert report["num_examples"] == 12
    assert report["map"] == pytest.approx(0.5)
    assert report["map_50"] == pytest.approx(0.7)
    assert report["map_75"] == pytest.approx(0.3)
    assert report["map_per_class"] == [0.4, 0.6]
    assert report["classes"] == [1, 2]
    assert "mar_100" not in report
    assert report["head_path"].endswith("final_head.npz")
    for key in ("image_size", "batch_size", "device", "score_threshold", "seed"):
        assert key in report["config"]


# --- orchestration seam (heavy deps faked) ---------------------------------


def test_evaluate_final_head_orchestration(tmp_path, monkeypatch):
    head_path = tmp_path / "final_head.npz"
    np.savez(head_path, **{"a": np.ones(2, dtype=np.float32), "b": np.zeros(3, dtype=np.float32)})

    bundle = SimpleNamespace(clients=[_client(0, "site-a", val_len=5)], num_classes=9)
    captured = {}

    monkeypatch.setattr(fhe, "load_detection_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(fhe, "resolve_device", lambda name: "cpu")
    monkeypatch.setattr(fhe, "build_detection_model", lambda **k: "MODEL")
    monkeypatch.setattr(
        fhe, "detection_trainable_parameter_names", lambda model: ["a", "b"]
    )

    def _fake_set(model, arrays):
        captured["arrays"] = arrays

    monkeypatch.setattr(fhe, "set_detection_head_parameters", _fake_set)
    monkeypatch.setattr(
        fhe,
        "evaluate_detection",
        lambda model, val, **k: {"map": 0.42, "map_50": 0.6, "map_75": 0.2},
    )

    config = DetectionConfig(
        manifest_path=str(tmp_path / "m.csv"),
        root_dir=str(tmp_path),
        pretrained=False,
    )
    report = evaluate_final_head(config, head_path)

    assert report["client_label"] == "site-a"
    assert report["num_examples"] == 5
    assert report["map"] == pytest.approx(0.42)
    # head arrays were loaded by name order and handed to the model
    assert [arr.shape for arr in captured["arrays"]] == [(2,), (3,)]


# --- CLI script ------------------------------------------------------------


def _load_script_module():
    path = ROOT / "scripts" / "evaluate_final_detection_head.py"
    spec = importlib.util.spec_from_file_location("evaluate_final_detection_head", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_metrics_json(tmp_path, monkeypatch):
    module = _load_script_module()
    output = tmp_path / "final_head_site_a_metrics.json"
    fake_report = {
        "mode": "post-fl-final-head-eval",
        "client_id": 0,
        "client_label": "site-a",
        "num_examples": 4,
        "map": 0.33,
        "map_50": 0.5,
        "map_75": 0.1,
        "head_path": "final_head.npz",
    }

    monkeypatch.setattr(module, "evaluate_final_head", lambda *a, **k: fake_report)

    module.main(
        [
            "--head-path",
            str(tmp_path / "final_head.npz"),
            "--manifest",
            str(tmp_path / "m.csv"),
            "--root-dir",
            str(tmp_path),
            "--client-id",
            "site-a",
            "--output",
            str(output),
            "--no-pretrained",
        ]
    )

    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["client_label"] == "site-a"
    assert data["map"] == pytest.approx(0.33)
    assert data["num_examples"] == 4
