"""Tests for post-FL operational detection inference (unlabeled images).

Cover the pure pieces (preprocessing math, detection extraction/scaling, report
assembly, input collection) directly, plus one real end-to-end smoke: build a
random-init detector, save its head to ``.npz``, reload it, and infer over a tiny
generated image, asserting the detections are JSON-serializable. The CLI script
is exercised with the heavy model calls faked, mirroring the eval-script test.
No network, GPU, or real PPE dataset is required (``pretrained=False``).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import src.evaluation.detection_inference as di
from src.evaluation.detection_inference import (
    build_inference_report,
    collect_image_paths,
    extract_detections,
    load_inference_model,
    preprocess_image,
    run_inference_on_image,
)
from src.models.detection_model import (
    build_detection_model,
    detection_trainable_parameter_names,
    get_detection_head_parameters,
)
from src.utils.detection_config import DetectionConfig

ROOT = Path(__file__).resolve().parents[1]


# --- input collection ------------------------------------------------------


def test_collect_image_paths_single_image(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    assert collect_image_paths(image=image) == [image]


def test_collect_image_paths_directory_sorted_and_filtered(tmp_path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")  # non-image ignored
    paths = collect_image_paths(input_dir=tmp_path)
    assert [p.name for p in paths] == ["a.jpg", "b.png"]


def test_collect_image_paths_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        collect_image_paths()
    with pytest.raises(ValueError, match="exactly one"):
        collect_image_paths(image=tmp_path / "a.jpg", input_dir=tmp_path)


def test_collect_image_paths_missing_image_raises(tmp_path):
    with pytest.raises(ValueError, match="image not found"):
        collect_image_paths(image=tmp_path / "missing.jpg")


def test_collect_image_paths_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        collect_image_paths(input_dir=tmp_path)


# --- preprocessing ---------------------------------------------------------


def test_preprocess_image_downscales_large_image():
    image = Image.new("RGB", (800, 400))
    tensor, scale = preprocess_image(image, image_size=200)
    assert scale == pytest.approx(0.25)
    # C, H, W after resize (longer side capped at image_size)
    assert tensor.shape == (3, 100, 200)


def test_preprocess_image_keeps_small_image():
    image = Image.new("RGB", (64, 48))
    tensor, scale = preprocess_image(image, image_size=512)
    assert scale == 1.0
    assert tensor.shape == (3, 48, 64)


# --- detection extraction --------------------------------------------------


def test_extract_detections_filters_and_rescales():
    output = {
        "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [0.0, 0.0, 5.0, 5.0]]),
        "scores": torch.tensor([0.9, 0.1]),
        "labels": torch.tensor([1, 2]),
    }
    detections = extract_detections(output, score_threshold=0.5, scale=0.5)

    assert len(detections) == 1  # low-score box dropped
    det = detections[0]
    assert det["class_id"] == 1
    assert det["score"] == pytest.approx(0.9)
    # scale=0.5 -> boxes multiplied by 1/0.5 = 2 back to original coords
    assert det["box"] == pytest.approx([20.0, 20.0, 40.0, 40.0])
    assert isinstance(det["label"], str) and det["label"] != "unknown"


def test_extract_detections_unknown_class_id_labeled_unknown():
    output = {
        "boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
        "scores": torch.tensor([0.8]),
        "labels": torch.tensor([999]),
    }
    detections = extract_detections(output, score_threshold=0.0, scale=1.0)
    assert detections[0]["label"] == "unknown"


# --- report assembly -------------------------------------------------------


def test_build_inference_report_keys_and_serializable():
    config = DetectionConfig(pretrained=False)
    report = build_inference_report(
        config=config,
        image_path="site_b/img.jpg",
        width=640,
        height=480,
        detections=[{"label": "helmet", "class_id": 1, "score": 0.7, "box": [1, 2, 3, 4]}],
        head_path="outputs/EXP-012-rerun/final_head.npz",
    )

    assert report["mode"] == "post-fl-final-head-inference"
    assert report["image_width"] == 640
    assert report["image_height"] == 480
    assert report["num_detections"] == 1
    assert report["head_path"].endswith("final_head.npz")
    for key in ("image_size", "device", "score_threshold", "pretrained", "seed"):
        assert key in report["config"]
    # entire report must be JSON-serializable
    json.dumps(report)


# --- end-to-end smoke: npz head round-trip + inference ---------------------


def _save_final_head(model, path):
    """Serialize a model's head to an npz keyed by parameter name (server layout)."""

    names = detection_trainable_parameter_names(model)
    arrays = get_detection_head_parameters(model)
    np.savez(path, **dict(zip(names, arrays, strict=True)))


def test_final_head_npz_loads_and_infers_json_serializable(tmp_path):
    config = DetectionConfig(pretrained=False, image_size=64, score_threshold=0.0, seed=7)

    # Produce a final_head.npz the way the ServerApp would (name-keyed).
    source_model = build_detection_model(
        num_classes=config.num_classes, pretrained=False, seed=config.seed
    )
    head_path = tmp_path / "final_head.npz"
    _save_final_head(source_model, head_path)

    model = load_inference_model(config, head_path)

    image_path = tmp_path / "tiny.jpg"
    Image.new("RGB", (48, 32), color=(120, 120, 120)).save(image_path)

    report, image = run_inference_on_image(
        model, image_path, config=config, head_path=head_path, device="cpu"
    )

    assert report["image_width"] == 48
    assert report["image_height"] == 32
    assert isinstance(report["detections"], list)
    assert image.size == (48, 32)
    # Detections (and the whole report) must be JSON-serializable.
    json.dumps(report)
    for det in report["detections"]:
        assert set(det) == {"label", "class_id", "score", "box"}
        assert len(det["box"]) == 4


def test_run_inference_moves_model_to_requested_device(tmp_path):
    class FakeModel:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def __call__(self, inputs):
            return [
                {
                    "boxes": torch.empty((0, 4)),
                    "scores": torch.empty((0,)),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }
            ]

    image_path = tmp_path / "tiny.jpg"
    Image.new("RGB", (48, 32), color=(120, 120, 120)).save(image_path)
    model = FakeModel()

    run_inference_on_image(
        model,
        image_path,
        config=DetectionConfig(pretrained=False, image_size=64),
        head_path=tmp_path / "final_head.npz",
        device="cpu",
    )

    assert model.device == "cpu"


# --- CLI script (heavy model calls faked) ----------------------------------


def _load_script_module():
    path = ROOT / "scripts" / "run_detection_inference.py"
    spec = importlib.util.spec_from_file_location("run_detection_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_json_and_annotated_images(tmp_path, monkeypatch):
    module = _load_script_module()

    input_dir = tmp_path / "images"
    input_dir.mkdir()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(input_dir / "img1.jpg")
    Image.new("RGB", (32, 32), color=(40, 50, 60)).save(input_dir / "img2.jpg")

    output_dir = tmp_path / "out"

    def _fake_load_model(config, head_path):
        return "MODEL"

    def _fake_infer(model, image_path, *, config, head_path, device):
        report = {
            "mode": "post-fl-final-head-inference",
            "image_path": str(image_path),
            "image_width": 32,
            "image_height": 32,
            "num_detections": 1,
            "detections": [
                {"label": "helmet", "class_id": 1, "score": 0.9, "box": [1.0, 1.0, 10.0, 10.0]}
            ],
            "head_path": str(head_path),
            "config": {},
        }
        return report, Image.open(image_path).convert("RGB")

    monkeypatch.setattr(module, "load_inference_model", _fake_load_model)
    monkeypatch.setattr(module, "run_inference_on_image", _fake_infer)

    module.main(
        [
            "--head-path",
            str(tmp_path / "final_head.npz"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--no-pretrained",
            "--score-threshold",
            "0.5",
            "--save-images",
        ]
    )

    for stem in ("img1", "img2"):
        json_path = output_dir / f"{stem}.json"
        assert json_path.is_file()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["num_detections"] == 1
        assert data["detections"][0]["label"] == "helmet"
        assert (output_dir / f"{stem}_annotated.jpg").is_file()


def test_script_requires_a_source(tmp_path, monkeypatch):
    module = _load_script_module()
    with pytest.raises(SystemExit):
        module.parse_args(
            ["--head-path", str(tmp_path / "h.npz"), "--output-dir", str(tmp_path)]
        )
