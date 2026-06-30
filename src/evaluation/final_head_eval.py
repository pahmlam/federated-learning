"""Site-side evaluation of the final aggregated detection head.

After a Flower deployment the server writes ``final_head.npz`` (the aggregated
detection head, keyed by ``detection_trainable_parameter_names``). This module
loads that head back into a fresh detector and evaluates it on a single site's
local validation shard -- the post-FL handoff step. The server never holds raw
data, so this runs on the site that owns the shard.

The pieces are split so the heavy parts (dataset, model, torchmetrics) are
isolated from the pure logic (npz loading by name, report assembly), which keeps
them unit-testable without a GPU, a Flower network, or a real dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.detection_data import (
    DetectionClientData,
    DetectionDatasetBundle,
    load_detection_bundle,
)
from src.models.detection_model import (
    build_detection_model,
    detection_trainable_parameter_names,
    resolve_device,
    set_detection_head_parameters,
)
from src.training.detection_trainer import evaluate_detection
from src.utils.detection_config import DetectionConfig

_SCALAR_METRIC_KEYS = ("map", "map_50", "map_75")
_PER_CLASS_KEYS = ("map_per_class", "mar_100_per_class", "classes")


def load_head_arrays_for_model(
    head_path: str | Path,
    param_names: list[str],
) -> list[np.ndarray]:
    """Load head arrays from ``head_path`` in ``param_names`` order.

    ``final_head.npz`` is keyed by parameter name when the server could line the
    names up with the arrays; otherwise it falls back to positional ``arr_<i>``
    keys. Both layouts are supported here, but a name-keyed archive that is
    missing any required parameter is an error -- silently dropping head weights
    would produce a wrong-but-plausible evaluation.
    """

    with np.load(head_path) as archive:
        files = set(archive.files)
        if set(param_names).issubset(files):
            return [np.asarray(archive[name]) for name in param_names]

        positional = [f"arr_{index}" for index in range(len(param_names))]
        if files == set(positional):
            return [np.asarray(archive[key]) for key in positional]

        missing = [name for name in param_names if name not in files]
        raise ValueError(
            f"{head_path} is missing required head parameter(s): {missing}. "
            f"Archive keys: {sorted(files)}"
        )


def select_eval_client(
    bundle: DetectionDatasetBundle,
    client_id: str | None = None,
) -> DetectionClientData:
    """Select the site shard to evaluate.

    A single-client manifest selects its only client. With multiple clients a
    ``client_id`` (matching the client label or numeric id) is required so the
    caller is explicit about which site's validation shard is being used.
    """

    if len(bundle.clients) == 1:
        return bundle.clients[0]

    if client_id is None:
        labels = [client.client_label for client in bundle.clients]
        raise ValueError(
            f"Manifest has multiple clients {labels}; pass client_id to choose one"
        )

    wanted = str(client_id)
    for client in bundle.clients:
        if wanted in {client.client_label, str(client.client_id)}:
            return client
    raise ValueError(f"Unknown detection client_id: {wanted}")


def build_eval_report(
    *,
    config: DetectionConfig,
    client: DetectionClientData,
    num_examples: int,
    metrics: dict[str, Any],
    head_path: str,
    manifest: str,
    root_dir: str,
) -> dict[str, Any]:
    """Assemble the JSON-ready evaluation report (pure, no I/O)."""

    report: dict[str, Any] = {
        "mode": "post-fl-final-head-eval",
        "client_id": client.client_id,
        "client_label": client.client_label,
        "num_examples": int(num_examples),
        "head_path": head_path,
        "manifest": manifest,
        "root_dir": root_dir,
        "config": {
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "device": config.device,
            "num_workers": config.num_workers,
            "score_threshold": config.score_threshold,
            "pretrained": config.pretrained,
            "seed": config.seed,
        },
    }
    for key in _SCALAR_METRIC_KEYS:
        if key in metrics:
            report[key] = float(metrics[key])
    for key in _PER_CLASS_KEYS:
        if key in metrics:
            report[key] = metrics[key]
    return report


def evaluate_final_head(
    config: DetectionConfig,
    head_path: str | Path,
    *,
    client_id: str | None = None,
) -> dict[str, Any]:
    """Load the final head and evaluate it on a site's local validation shard.

    Orchestration only: loads the bundle, selects the client, builds the model,
    loads the head by name order, and evaluates. Returns the JSON-ready report.
    """

    bundle = load_detection_bundle(
        config.manifest_path,
        config.root_dir,
        image_size=config.image_size,
    )
    client = select_eval_client(bundle, client_id)
    device = resolve_device(config.device)

    model = build_detection_model(
        num_classes=bundle.num_classes,
        pretrained=config.pretrained,
        seed=config.seed,
    )
    param_names = detection_trainable_parameter_names(model)
    arrays = load_head_arrays_for_model(head_path, param_names)
    set_detection_head_parameters(model, arrays)

    metrics = evaluate_detection(
        model,
        client.val,
        batch_size=config.batch_size,
        device=device,
        num_workers=config.num_workers,
        score_threshold=config.score_threshold,
        log_prefix=f"[final-head {client.client_label}]",
    )

    return build_eval_report(
        config=config,
        client=client,
        num_examples=len(client.val),
        metrics=metrics,
        head_path=str(head_path),
        manifest=config.manifest_path,
        root_dir=config.root_dir,
    )
