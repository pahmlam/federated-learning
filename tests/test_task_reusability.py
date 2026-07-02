"""Cross-workload proof that ``FederatedTask`` is genuinely reusable.

One workload-agnostic driver (``drive_task_round``) exercises the full
build -> get/set arrays -> train -> evaluate cycle purely through the
``FederatedTask`` interface, then the SAME driver is run against two very
different real implementations -- detection (tiny Faster R-CNN) and embedding
classification (in-memory tensors) -- with no GPU, Flower network, or real
pretrained weights.
"""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.data.detection_data import load_detection_bundle
from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    write_detection_manifest,
)
from src.data.embedding import EmbeddingClientDataset, EmbeddingDatasetBundle
from src.fl.detection_task import DetectionTask, DetectionTaskContext
from src.fl.embedding_classification_task import (
    EmbeddingClassificationTask,
    EmbeddingClientContext,
)
from src.fl.task import FederatedTask, RoundOutput
from src.utils.config import DemoConfig
from src.utils.detection_config import DetectionConfig


def drive_task_round(task: FederatedTask, *, config, ctx, num_classes):
    """Workload-agnostic: drive any FederatedTask through one local round.

    Mirrors what the ClientApp does, but with freshly built initial arrays so it
    needs no Flower message.
    """

    model = task.build_model(config, num_classes=num_classes)
    arrays = task.get_global_arrays(model)
    task.set_global_arrays(model, arrays)  # round-trip the global arrays
    train_out = task.train_round(model, ctx)
    eval_out = task.evaluate_round(model, ctx)
    return train_out, eval_out


def _assert_valid_round(train_out, eval_out):
    for out in (train_out, eval_out):
        assert isinstance(out, RoundOutput)
        assert out.num_examples > 0
        assert out.metrics  # non-empty scalar metrics
        assert all(isinstance(v, float) for v in out.metrics.values())


# --- workload A: embedding classification (in-memory) ----------------------


def _embedding_ctx(embedding_dim=8, num_classes=3, n_train=12, n_val=6):
    rng = np.random.default_rng(0)
    client = EmbeddingClientDataset(
        client_id=0,
        client_label="site-a",
        train_x=torch.tensor(rng.standard_normal((n_train, embedding_dim)), dtype=torch.float32),
        train_y=torch.tensor(rng.integers(0, num_classes, n_train), dtype=torch.long),
        val_x=torch.tensor(rng.standard_normal((n_val, embedding_dim)), dtype=torch.float32),
        val_y=torch.tensor(rng.integers(0, num_classes, n_val), dtype=torch.long),
        label_histogram={},
    )
    bundle = EmbeddingDatasetBundle(
        clients=[client],
        pooled=SimpleNamespace(
            train_x=client.train_x, train_y=client.train_y,
            val_x=client.val_x, val_y=client.val_y,
        ),
        label_mapping={str(i): i for i in range(num_classes)},
        artifact_path="",
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )
    config = DemoConfig(
        embedding_dim=embedding_dim, num_classes=num_classes,
        local_epochs=1, batch_size=4, lr=0.05, seed=7,
    )
    return config, EmbeddingClientContext(config=config, bundle=bundle, client=client)


# --- workload B: detection (tiny real bundle) ------------------------------


def _detection_ctx(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "voc_labels").mkdir()
    spec = {f"h{i}": "helmet" for i in range(4)}
    for stem, name in spec.items():
        Image.new("RGB", (48, 48), color=(20, 50, 70)).save(tmp_path / "images" / f"{stem}.png")
        (tmp_path / "voc_labels" / f"{stem}.xml").write_text(
            f"<annotation><object><name>{name}</name><bndbox>"
            "<xmin>2</xmin><ymin>2</ymin><xmax>20</xmax><ymax>20</ymax>"
            "</bndbox></object></annotation>",
            encoding="utf-8",
        )
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")
    rows = generate_detection_manifest_rows(
        samples, sites=["site-a"], per_site=4, val_fraction=0.5, seed=2026
    )
    manifest = tmp_path / "manifest.csv"
    write_detection_manifest(rows, manifest)
    bundle = load_detection_bundle(manifest, tmp_path, image_size=64)
    config = DetectionConfig(
        pretrained=False, image_size=64, device="cpu", local_epochs=1, batch_size=1
    )
    ctx = DetectionTaskContext(config=config, bundle=bundle, client=bundle.clients[0])
    return config, ctx


# --- the shared-driver proof ------------------------------------------------


def test_same_driver_runs_embedding_task():
    config, ctx = _embedding_ctx()
    train_out, eval_out = drive_task_round(
        EmbeddingClassificationTask(), config=config, ctx=ctx, num_classes=ctx.bundle.num_classes
    )
    _assert_valid_round(train_out, eval_out)
    assert "train_loss" in train_out.metrics
    assert "accuracy" in eval_out.metrics


def test_same_driver_runs_detection_task(tmp_path):
    config, ctx = _detection_ctx(tmp_path)
    train_out, eval_out = drive_task_round(
        DetectionTask(), config=config, ctx=ctx, num_classes=ctx.bundle.num_classes
    )
    _assert_valid_round(train_out, eval_out)
    assert "train_loss" in train_out.metrics
    assert {"map", "map_50", "map_75"} <= set(eval_out.metrics)


def test_both_tasks_are_federated_tasks():
    assert isinstance(EmbeddingClassificationTask(), FederatedTask)
    assert isinstance(DetectionTask(), FederatedTask)
