# Source Layout

Source package for the FL camera research code.

Default training assumption: use a pretrained backbone, freeze it by default, and train only lightweight components such as an embedding layer, task head, adapter, or LoRA module. Full-model training from scratch is out of scope for the first implementation stage.

Current modules:

- `data/`: synthetic dataset generation and IID/non-IID client partitioning.
- `models/`: frozen backbone plus trainable classifier head.
- `training/`: centralized and local-only head-only training loops.
- `fl/`: Flower ClientApp, ServerApp, and FedAvg simulation path.
- `evaluation/`: metrics, per-client evaluation, update size, communication cost.
- `utils/`: config and JSON output helpers.

This v0 uses synthetic data only. Face/PPE datasets and adapters/LoRA are later stages.
