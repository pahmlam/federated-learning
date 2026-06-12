# Source Layout

Future source package for the FL camera research code.

Default training assumption: use a pretrained backbone, freeze it by default, and train only lightweight components such as an embedding layer, task head, adapter, or LoRA module. Full-model training from scratch is out of scope for the first implementation stage.

Expected future modules:

- `data/`: dataset loading, preprocessing, and client partitioning.
- `models/`: pretrained backbone wrappers plus trainable head, embedding, adapter, and LoRA definitions.
- `training/`: centralized and local-only loops for the selected lightweight trainable parts.
- `fl/`: Flower ClientApp, ServerApp, strategy, aggregation, and client selection for lightweight updates.
- `evaluation/`: metrics, per-client evaluation, and report aggregation.
- `utils/`: shared utilities for config, logging, seeding, and paths.

No Python implementation is added in this skeleton step.
