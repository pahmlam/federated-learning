"""Flower ServerApp for PPE detection head-only FedAvg deployment."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.evaluation.metrics import parameter_bytes
from src.fl.deployment_artifacts import finalize_deployment_artifacts
from src.fl.detection_task import DetectionTask
from src.models.detection_model import detection_trainable_parameter_names
from src.utils.detection_config import DetectionConfig

app = ServerApp()

_TASK = DetectionTask()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = DetectionConfig.from_env_and_overrides(
        dict(context.run_config),
        env_overrides=True,
    )
    model = _TASK.build_model(config, num_classes=config.num_classes)
    head_arrays = _TASK.get_global_arrays(model)
    head_param_names = detection_trainable_parameter_names(model)
    update_size_bytes = parameter_bytes(head_arrays)

    strategy = _build_strategy(config)

    started_at = datetime.now(timezone.utc).isoformat()
    start_perf = time.time()
    result = None
    exception_raised = False
    try:
        result = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(head_arrays),
            num_rounds=config.num_rounds,
            train_config=_round_config(config),
            evaluate_config=_round_config(config),
        )
    except Exception:
        exception_raised = True
        raise
    finally:
        # Always persist artifacts, even on failure, so a run can be journaled
        # from disk instead of only from the Flower logstream. The status is
        # derived from the result (completed / partial / failed), not assumed.
        finalize_deployment_artifacts(
            config,
            result=result,
            update_size_bytes=update_size_bytes,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            runtime_seconds=time.time() - start_perf,
            exception_raised=exception_raised,
            head_param_names=head_param_names,
            run_id=getattr(context, "run_id", None),
        )


def _build_strategy(config: DetectionConfig) -> FedAvg:
    """Construct the FedAvg strategy with resolved min-node thresholds.

    ``fraction_train``/``fraction_evaluate`` stay at 1.0 (use all sampled nodes);
    the min-node values are strict (equal to ``num_clients``) unless the config
    explicitly overrides them for a straggler/dropout robustness smoke run.
    """

    return FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=config.effective_min_train_nodes,
        min_evaluate_nodes=config.effective_min_evaluate_nodes,
        min_available_nodes=config.effective_min_available_nodes,
        weighted_by_key="num-examples",
    )


def _round_config(config: DetectionConfig) -> ConfigRecord:
    return ConfigRecord(
        {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            "score_threshold": config.score_threshold,
        }
    )
