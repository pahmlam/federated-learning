import json
import numpy as np
import pytest
import subprocess
import sys

from src.data.embedding import embedding_artifact_to_bundle
from src.data.real_data import EmbeddingArtifact, save_embedding_artifact
import torch

from src.models.embedding_head import (
    build_embedding_model,
    create_embedding_head_model,
    embedding_trainable_parameter_names,
    get_embedding_head_parameters,
    set_embedding_head_parameters,
)
from src.training.embedding_baselines import (
    run_embedding_centralized,
    run_embedding_local_only,
)
from src.training.trainer import train_head
from src.utils.config import DemoConfig


def _head_weight_norm(model) -> float:
    return float(model.head.weight.detach().norm().item())


def _tiny_artifact():
    return EmbeddingArtifact(
        features=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ],
            dtype=np.float32,
        ),
        labels=np.asarray(["safe", "safe", "unsafe", "unsafe", "safe", "unsafe"]),
        client_ids=np.asarray(["site-a", "site-a", "site-a", "site-a", "site-b", "site-b"]),
        splits=np.asarray(["train", "val", "train", "val", "train", "val"]),
        sample_ids=np.asarray(["s1", "s2", "s3", "s4", "s5", "s6"]),
    )


def test_embedding_artifact_to_bundle_creates_client_tensors():
    bundle = embedding_artifact_to_bundle(_tiny_artifact(), artifact_path="tiny.npz")

    assert bundle.artifact_path == "tiny.npz"
    assert bundle.embedding_dim == 3
    assert bundle.num_classes == 2
    assert bundle.label_mapping == {"safe": 0, "unsafe": 1}
    assert [client.client_label for client in bundle.clients] == ["site-a", "site-b"]
    assert bundle.clients[0].train_x.shape == (2, 3)
    assert bundle.clients[0].val_y.tolist() == [0, 1]
    assert bundle.pooled.train_x.shape == (3, 3)
    assert bundle.pooled.val_x.shape == (3, 3)


def test_embedding_artifact_to_bundle_rejects_client_missing_val_split():
    artifact = EmbeddingArtifact(
        features=np.ones((4, 3), dtype=np.float32),
        labels=np.asarray(["safe", "unsafe", "safe", "unsafe"]),
        client_ids=np.asarray(["site-a", "site-a", "site-b", "site-b"]),
        splits=np.asarray(["train", "val", "train", "train"]),
        sample_ids=np.asarray(["s1", "s2", "s3", "s4"]),
    )

    with pytest.raises(
        ValueError,
        match="Client site-b is missing required baseline split.*val",
    ):
        embedding_artifact_to_bundle(artifact, artifact_path="missing-val.npz")


def test_embedding_artifact_to_bundle_rejects_client_missing_train_split():
    artifact = EmbeddingArtifact(
        features=np.ones((4, 3), dtype=np.float32),
        labels=np.asarray(["safe", "unsafe", "safe", "unsafe"]),
        client_ids=np.asarray(["site-a", "site-a", "site-b", "site-b"]),
        splits=np.asarray(["train", "val", "val", "val"]),
        sample_ids=np.asarray(["s1", "s2", "s3", "s4"]),
    )

    with pytest.raises(
        ValueError,
        match="Client site-b is missing required baseline split.*train",
    ):
        embedding_artifact_to_bundle(artifact, artifact_path="missing-train.npz")


def test_embedding_head_model_trains_only_head():
    model = create_embedding_head_model(embedding_dim=3, num_classes=2, seed=2026)

    assert embedding_trainable_parameter_names(model) == ["head.weight", "head.bias"]


def test_embedding_head_l2_normalizes_input_when_enabled():
    model = create_embedding_head_model(
        embedding_dim=3, num_classes=2, seed=2026, normalize_input=True
    )
    # Same direction, different magnitude -> identical logits after L2-normalize.
    scaled = model(torch.tensor([[3.0, 0.0, 0.0]]))
    unit = model(torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(scaled, unit, atol=1e-6)


def test_embedding_head_does_not_normalize_by_default():
    model = create_embedding_head_model(embedding_dim=3, num_classes=2, seed=2026)
    scaled = model(torch.tensor([[3.0, 0.0, 0.0]]))
    unit = model(torch.tensor([[1.0, 0.0, 0.0]]))
    assert not torch.allclose(scaled, unit, atol=1e-6)


def test_embedding_mlp_head_exposes_all_layer_parameters():
    model = create_embedding_head_model(
        embedding_dim=3, num_classes=2, seed=2026, hidden_dim=8
    )
    params = get_embedding_head_parameters(model)
    # Two Linear layers -> 4 arrays (weight + bias each).
    assert len(params) == 4
    assert len(embedding_trainable_parameter_names(model)) == 4
    # Round-trip through FedAvg-style serialization must succeed.
    set_embedding_head_parameters(model, params)


def test_set_embedding_head_parameters_rejects_wrong_count():
    model = create_embedding_head_model(
        embedding_dim=3, num_classes=2, seed=2026, hidden_dim=8
    )
    with pytest.raises(ValueError, match="Expected 4 head parameter arrays"):
        set_embedding_head_parameters(model, get_embedding_head_parameters(model)[:2])


def test_build_embedding_model_reads_capacity_from_config():
    bundle = embedding_artifact_to_bundle(_tiny_artifact(), artifact_path="tiny.npz")
    config = DemoConfig(
        normalize_embedding=True,
        head_hidden_dim=8,
    )
    model = build_embedding_model(config, bundle)
    assert model.normalize_input is True
    assert len(get_embedding_head_parameters(model)) == 4


def test_train_head_weight_decay_shrinks_head_weight_norm():
    bundle = embedding_artifact_to_bundle(_tiny_artifact(), artifact_path="tiny.npz")
    train_x, train_y = bundle.pooled.train_x, bundle.pooled.train_y

    kwargs = dict(epochs=20, batch_size=2, lr=0.1, seed=2026)
    no_decay = create_embedding_head_model(embedding_dim=3, num_classes=2, seed=2026)
    train_head(model=no_decay, train_x=train_x, train_y=train_y, weight_decay=0.0, **kwargs)

    decayed = create_embedding_head_model(embedding_dim=3, num_classes=2, seed=2026)
    train_head(model=decayed, train_x=train_x, train_y=train_y, weight_decay=0.5, **kwargs)

    assert _head_weight_norm(decayed) < _head_weight_norm(no_decay)


def test_train_head_defaults_to_no_weight_decay():
    bundle = embedding_artifact_to_bundle(_tiny_artifact(), artifact_path="tiny.npz")
    model = create_embedding_head_model(embedding_dim=3, num_classes=2, seed=2026)
    # Must run without passing weight_decay (default 0.0).
    metrics = train_head(
        model=model,
        train_x=bundle.pooled.train_x,
        train_y=bundle.pooled.train_y,
        epochs=1,
        batch_size=2,
        lr=0.1,
        seed=2026,
    )
    assert "loss" in metrics


def test_embedding_centralized_and_local_only_baselines_run():
    bundle = embedding_artifact_to_bundle(_tiny_artifact(), artifact_path="tiny.npz")
    config = DemoConfig(
        profile="oom-safe",
        num_clients=len(bundle.clients),
        num_classes=bundle.num_classes,
        input_dim=bundle.embedding_dim,
        embedding_dim=bundle.embedding_dim,
    ).normalized()

    centralized = run_embedding_centralized(config, bundle)
    local_only = run_embedding_local_only(config, bundle)

    assert centralized["data_source"] == "embedding"
    assert local_only["data_source"] == "embedding"
    assert centralized["num_clients"] == 2
    assert local_only["num_clients"] == 2
    assert len(centralized["per_client"]) == 2
    assert len(local_only["per_client"]) == 2
    assert "macro_f1" in centralized["global"]
    assert "unsafe_recall" in centralized["global"]
    assert "false_negative_rate" in centralized["global"]


def test_run_embedding_demo_infers_exp_id_from_output_dir(tmp_path):
    artifact_path = tmp_path / "artifact.npz"
    output_dir = tmp_path / "EXP-004"
    save_embedding_artifact(artifact_path, _tiny_artifact())

    subprocess.run(
        [
            sys.executable,
            "scripts/run_embedding_demo.py",
            "--mode",
            "centralized",
            "--artifact",
            str(artifact_path),
            "--profile",
            "oom-safe",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "EXP-004"
    assert "macro_f1" in summary["modes"]["centralized"]
    assert "unsafe_recall" in summary["modes"]["centralized"]


def test_run_embedding_demo_exp_id_override_wins(tmp_path):
    artifact_path = tmp_path / "artifact.npz"
    output_dir = tmp_path / "not-an-exp"
    save_embedding_artifact(artifact_path, _tiny_artifact())

    subprocess.run(
        [
            sys.executable,
            "scripts/run_embedding_demo.py",
            "--mode",
            "centralized",
            "--artifact",
            str(artifact_path),
            "--profile",
            "oom-safe",
            "--output-dir",
            str(output_dir),
            "--exp-id",
            "CUSTOM",
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "CUSTOM"


def test_run_embedding_demo_accepts_training_overrides(tmp_path):
    artifact_path = tmp_path / "artifact.npz"
    output_dir = tmp_path / "EXP-005"
    save_embedding_artifact(artifact_path, _tiny_artifact())

    subprocess.run(
        [
            sys.executable,
            "scripts/run_embedding_demo.py",
            "--mode",
            "centralized",
            "--artifact",
            str(artifact_path),
            "--profile",
            "oom-safe",
            "--output-dir",
            str(output_dir),
            "--local-epochs",
            "3",
            "--centralized-epochs",
            "3",
            "--num-rounds",
            "3",
            "--batch-size",
            "4",
            "--lr",
            "0.01",
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "EXP-005"
    assert summary["config"]["local_epochs"] == 3
    assert summary["config"]["centralized_epochs"] == 3
    assert summary["config"]["num_rounds"] == 3
    assert summary["config"]["batch_size"] == 4


def test_run_embedding_demo_accepts_weight_decay(tmp_path):
    artifact_path = tmp_path / "artifact.npz"
    output_dir = tmp_path / "EXP-008"
    save_embedding_artifact(artifact_path, _tiny_artifact())

    subprocess.run(
        [
            sys.executable,
            "scripts/run_embedding_demo.py",
            "--mode",
            "centralized",
            "--artifact",
            str(artifact_path),
            "--profile",
            "oom-safe",
            "--output-dir",
            str(output_dir),
            "--weight-decay",
            "0.01",
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["weight_decay"] == 0.01


def test_run_embedding_demo_accepts_capacity_flags(tmp_path):
    artifact_path = tmp_path / "artifact.npz"
    output_dir = tmp_path / "EXP-010"
    save_embedding_artifact(artifact_path, _tiny_artifact())

    subprocess.run(
        [
            sys.executable,
            "scripts/run_embedding_demo.py",
            "--mode",
            "centralized",
            "--artifact",
            str(artifact_path),
            "--profile",
            "oom-safe",
            "--output-dir",
            str(output_dir),
            "--normalize-embedding",
            "--head-hidden-dim",
            "8",
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["normalize_embedding"] is True
    assert summary["config"]["head_hidden_dim"] == 8
