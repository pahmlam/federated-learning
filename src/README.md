# Source Layout

Source package for the reusable FL camera/vision system.

The main deliverable is the FL system: data partitioning, local training,
distributed evaluation, aggregation, deployment, and experiment artifacts. PPE
detection is the current workload used to validate that system, not the final
application target.

Default training assumption: use a pretrained backbone, freeze it by default,
and train only lightweight components such as a task head, embedding layer,
adapter, or LoRA module. Full-model training from scratch is out of scope for
the first implementation stage.

## Active Workload

PPE detection currently uses:

- `fasterrcnn_mobilenet_v3_large_fpn`
- frozen backbone + FPN
- trainable RPN head + ROI box predictor
- FedAvg over detection-head parameters only
- distributed mAP evaluation per client/site

The archived PPE classification path is still kept for reference, but EXP-001
to EXP-010 should be treated as historical baseline work.

## Modules

- `data/`: dataset parsing, manifest generation, client partitioning, and
  detection dataset bundle loading.
- `models/`: model builders and trainable-parameter serialization helpers,
  including Faster R-CNN detection-head get/set functions.
- `training/`: centralized, local-only, and local client training/evaluation
  loops.
- `fl/`: manual FedAvg simulation helpers plus Flower `ClientApp`/`ServerApp`
  deployment entry points.
- `evaluation/`: metrics, per-client evaluation helpers, update size, and
  communication cost utilities.
- `utils/`: config, environment loading, resource tracking, and JSON output
  helpers.

## Important Entry Points

- `src/fl/detection_clientapp.py`: Flower client app for real deployment.
- `src/fl/detection_serverapp.py`: Flower server app and FedAvg strategy setup.
- `src/fl/detection_federated.py`: local/manual detection FedAvg simulation.
- `src/training/detection_baselines.py`: centralized and local-only detection
  baselines.
- `src/training/detection_trainer.py`: Faster R-CNN head training and mAP
  evaluation.
- `src/data/detection_manifest.py`: seeded client split and non-IID manifest
  generation.
- `src/data/detection_data.py`: per-client dataset bundle loading.

Expected flow:

```text
manifest -> dataset bundle -> model/head params -> train/evaluate
         -> centralized/local-only/manual FedAvg or Flower deployment
         -> per-client metrics + weighted global metrics + artifacts
```
