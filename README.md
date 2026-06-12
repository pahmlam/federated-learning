# FL Camera Research

Repository skeleton for Federated Learning research and demos on camera/vision tasks, focused first on **face** and **clothing/PPE**.

The initial goal is to build a clean experimental foundation before adding new methods. Flower is used as the baseline FL framework, while the existing `demo/` directory remains a quickstart reference.

The default research assumption is **not** to train a full vision model from scratch. Experiments should start from a pretrained backbone, keep the backbone frozen or nearly frozen, and train only lightweight parts such as the embedding layer, task head, adapter, or LoRA module.

## Main Training Modes

- `centralized`: train the same lightweight trainable part on pooled data when a centralized baseline is possible.
- `local-only`: train/evaluate the lightweight trainable part for each client or site independently.
- `federated`: train the lightweight trainable part with Flower across multiple simulated or real clients.

## Research Direction

The repo is intended to grow toward:

- non-IID data partitioning by site, camera, identity, clothing/PPE type, or lighting condition;
- pretrained backbone with frozen or near-frozen weights;
- trainable head, embedding, adapter, or LoRA modules instead of full-model training;
- aggregation of only the selected trainable/update parts when possible;
- edge-aware FL experiments for compute, memory, communication cost, and client availability;
- personalized FL and communication-efficient methods for face and clothing/PPE tasks.

## Current Status

This is only a skeleton. No training code, model code, dependency file, or dataset is added in this step.
