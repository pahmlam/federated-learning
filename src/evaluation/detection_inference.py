"""Operational PPE detection inference from a saved federated final head.

Post-FL handoff, unlabeled variant: take ``outputs/<exp-id>/final_head.npz``
(the aggregated detection head produced by a Flower deployment) and run the
detector over a site's local *unlabeled* images to produce detection JSON. This
is the operational counterpart of ``final_head_eval`` (which needs labels for
mAP); it runs on the site that owns the images, since the server holds no data.

The pieces are split so the heavy parts (model build, forward pass) are isolated
from the pure logic (preprocessing math, detection extraction, report assembly),
which keeps the latter unit-testable without a GPU or a real dataset. The head is
loaded through the same name-ordered path as evaluation
(``load_head_arrays_for_model``) so both handoff scripts share one loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.transforms.functional import to_tensor

from src.evaluation.final_head_eval import load_head_arrays_for_model
from src.models.detection_model import (
    build_detection_model,
    detection_trainable_parameter_names,
    set_detection_head_parameters,
)
from src.utils.detection_config import DetectionConfig, ppe_index_to_label

# Image suffixes we treat as inputs when scanning a directory.
IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)


def collect_image_paths(
    *, image: str | Path | None = None, input_dir: str | Path | None = None
) -> list[Path]:
    """Resolve the CLI inputs to a sorted list of image paths.

    Exactly one of ``image`` / ``input_dir`` must be given. A directory is
    scanned (non-recursively) for known image suffixes; a missing file or a
    directory with no images is an explicit error rather than a silent empty run.
    """

    if bool(image) == bool(input_dir):
        raise ValueError("pass exactly one of image / input_dir")

    if image is not None:
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"image not found: {path}")
        return [path]

    directory = Path(input_dir)  # type: ignore[arg-type]
    if not directory.is_dir():
        raise ValueError(f"input_dir not found: {directory}")
    paths = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no images ({sorted(IMAGE_SUFFIXES)}) found in {directory}")
    return paths


def preprocess_image(image: Image.Image, image_size: int) -> tuple[torch.Tensor, float]:
    """Resize like ``PPEDetectionDataset`` and return (tensor, scale).

    ``scale`` (<= 1.0) is the factor applied to the image; predicted boxes are
    later divided by it to map back to the original image coordinates.
    """

    width, height = image.size
    scale = min(1.0, image_size / max(width, height))
    if scale < 1.0:
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size)
    return to_tensor(image), scale


def extract_detections(
    output: dict[str, torch.Tensor],
    *,
    score_threshold: float,
    scale: float,
) -> list[dict[str, Any]]:
    """Convert one raw model output to JSON-ready detections in original coords.

    Filters by ``score_threshold`` and rescales boxes by ``1 / scale`` so they
    refer to the original (pre-resize) image. Labels are mapped to canonical PPE
    class names; an unexpected id (e.g. background) is reported as ``unknown``.
    """

    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    labels = output["labels"].detach().cpu()
    index_to_label = ppe_index_to_label()
    inv_scale = 1.0 / scale

    detections: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        confidence = float(score)
        if confidence < score_threshold:
            continue
        class_id = int(label)
        x1, y1, x2, y2 = (float(value) * inv_scale for value in box.tolist())
        detections.append(
            {
                "label": index_to_label.get(class_id, "unknown"),
                "class_id": class_id,
                "score": confidence,
                "box": [x1, y1, x2, y2],
            }
        )
    return detections


def build_inference_report(
    *,
    config: DetectionConfig,
    image_path: str | Path,
    width: int,
    height: int,
    detections: list[dict[str, Any]],
    head_path: str | Path,
) -> dict[str, Any]:
    """Assemble the JSON-ready per-image inference report (pure, no I/O)."""

    return {
        "mode": "post-fl-final-head-inference",
        "image_path": str(image_path),
        "image_width": int(width),
        "image_height": int(height),
        "num_detections": len(detections),
        "detections": detections,
        "head_path": str(head_path),
        "config": {
            "image_size": config.image_size,
            "device": config.device,
            "score_threshold": config.score_threshold,
            "pretrained": config.pretrained,
            "seed": config.seed,
        },
    }


def annotate_image(
    image: Image.Image, detections: list[dict[str, Any]]
) -> Image.Image:
    """Return a new copy of ``image`` with detection boxes + labels drawn on it."""

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        caption = f"{detection['label']} {detection['score']:.2f}"
        draw.text((x1, max(0.0, y1 - 10)), caption, fill=(255, 0, 0))
    return annotated


def load_inference_model(
    config: DetectionConfig, head_path: str | Path
) -> nn.Module:
    """Build a fresh detector and load the saved final head into it (eval mode)."""

    model = build_detection_model(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        seed=config.seed,
    )
    param_names = detection_trainable_parameter_names(model)
    arrays = load_head_arrays_for_model(head_path, param_names)
    set_detection_head_parameters(model, arrays)
    model.eval()
    return model


@torch.no_grad()
def run_inference_on_image(
    model: nn.Module,
    image_path: str | Path,
    *,
    config: DetectionConfig,
    head_path: str | Path,
    device: str,
) -> tuple[dict[str, Any], Image.Image]:
    """Run the detector on one image; return (report, original RGB image).

    The image is returned so the caller can optionally annotate it without a
    second decode. Boxes in the report are already in original-image coordinates.
    """

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    tensor, scale = preprocess_image(image, config.image_size)
    model.to(device)
    outputs = model([tensor.to(device)])
    detections = extract_detections(
        outputs[0], score_threshold=config.score_threshold, scale=scale
    )
    report = build_inference_report(
        config=config,
        image_path=image_path,
        width=width,
        height=height,
        detections=detections,
        head_path=head_path,
    )
    return report, image
