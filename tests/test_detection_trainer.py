import torch
from PIL import Image

from src.data.detection_dataset import DetectionRecord, PPEDetectionDataset
from src.models.detection_model import build_detection_model
from src.training.detection_trainer import evaluate_detection, train_detection_head


def _tiny_dataset(tmp_path, n=2):
    records = []
    for i in range(n):
        Image.new("RGB", (96, 96), color=(i * 20, 40, 80)).save(tmp_path / f"img{i}.png")
        (tmp_path / f"img{i}.xml").write_text(
            "<annotation><size><width>96</width><height>96</height><depth>3</depth></size>"
            "<object><name>helmet</name><bndbox>"
            "<xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>"
            "<object><name>gloves</name><bndbox>"
            "<xmin>55</xmin><ymin>55</ymin><xmax>85</xmax><ymax>85</ymax></bndbox></object>"
            "</annotation>",
            encoding="utf-8",
        )
        records.append(
            DetectionRecord(
                image_path=tmp_path / f"img{i}.png",
                voc_path=tmp_path / f"img{i}.xml",
                client_id="site-a",
                split="train",
            )
        )
    return PPEDetectionDataset(records, image_size=128)


def test_train_one_step_and_evaluate_map_runs(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    model = build_detection_model(num_classes=9, pretrained=False, seed=2026)

    result = train_detection_head(
        model,
        dataset,
        epochs=1,
        batch_size=1,
        lr=0.005,
        momentum=0.9,
        weight_decay=5e-4,
        device="cpu",
        seed=2026,
    )
    # Loss must be a finite number -> backward through the frozen-backbone head works.
    assert torch.isfinite(torch.tensor(result["train_loss"]))

    metrics = evaluate_detection(model, dataset, batch_size=1, device="cpu")
    # torchmetrics mAP keys are present and JSON-serializable (floats / lists).
    assert "map" in metrics and "map_50" in metrics
    assert isinstance(metrics["map"], float)
