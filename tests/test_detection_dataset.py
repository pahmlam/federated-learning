import torch
from PIL import Image

from src.data.detection_dataset import (
    DetectionRecord,
    PPEDetectionDataset,
    detection_collate_fn,
    read_voc_objects,
    voc_to_target,
)
from src.utils.detection_config import ppe_label_to_index

LABEL_MAP = ppe_label_to_index()


def _write_voc(path, objects, size=(100, 50)):
    """objects: list of (name, xmin, ymin, xmax, ymax)."""
    obj_xml = "".join(
        f"<object><name>{name}</name><bndbox>"
        f"<xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>"
        f"</bndbox></object>"
        for name, x0, y0, x1, y1 in objects
    )
    path.write_text(
        f"<annotation><size><width>{size[0]}</width><height>{size[1]}</height>"
        f"<depth>3</depth></size>{obj_xml}</annotation>",
        encoding="utf-8",
    )


def test_read_voc_objects_parses_names_and_boxes(tmp_path):
    xml = tmp_path / "a.xml"
    _write_voc(xml, [("helmet", 1, 2, 3, 4), ("person", 5, 6, 7, 8)])
    objects = read_voc_objects(xml)
    assert [o.name for o in objects] == ["helmet", "person"]
    assert objects[0].box == (1.0, 2.0, 3.0, 4.0)


def test_voc_to_target_filters_non_ppe_and_remaps_labels(tmp_path):
    xml = tmp_path / "a.xml"
    _write_voc(xml, [("helmet", 0, 0, 10, 10), ("person", 0, 0, 5, 5), ("gloves", 1, 1, 4, 4)])
    target = voc_to_target(read_voc_objects(xml), LABEL_MAP)
    # person is dropped; helmet->1, gloves->5
    assert target["labels"].tolist() == [LABEL_MAP["helmet"], LABEL_MAP["gloves"]]
    assert target["boxes"].shape == (2, 4)


def test_voc_to_target_scales_boxes_and_drops_degenerate(tmp_path):
    xml = tmp_path / "a.xml"
    _write_voc(xml, [("helmet", 0, 0, 10, 10), ("gloves", 5, 5, 5, 9)])  # gloves: xmax==xmin
    target = voc_to_target(read_voc_objects(xml), LABEL_MAP, scale=0.5)
    assert target["labels"].tolist() == [LABEL_MAP["helmet"]]
    assert target["boxes"][0].tolist() == [0.0, 0.0, 5.0, 5.0]


def test_voc_to_target_empty_when_no_ppe():
    target = voc_to_target([], LABEL_MAP)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_dataset_downscales_large_image_and_scales_boxes(tmp_path):
    Image.new("RGB", (100, 50)).save(tmp_path / "img.png")
    _write_voc(tmp_path / "img.xml", [("helmet", 0, 0, 50, 25)])
    record = DetectionRecord(
        image_path=tmp_path / "img.png",
        voc_path=tmp_path / "img.xml",
        client_id="site-a",
        split="train",
    )
    dataset = PPEDetectionDataset([record], image_size=20)
    image, target = dataset[0]
    # longer side 100 -> 20 => scale 0.2; image is (C, H, W) = (3, 10, 20)
    assert image.shape == (3, 10, 20)
    assert target["boxes"][0].tolist() == [0.0, 0.0, 10.0, 5.0]
    assert target["labels"].tolist() == [LABEL_MAP["helmet"]]


def test_collate_fn_returns_parallel_tuples():
    batch = [(torch.zeros(3, 4, 4), {"a": 1}), (torch.zeros(3, 4, 4), {"a": 2})]
    images, targets = detection_collate_fn(batch)
    assert len(images) == 2 and len(targets) == 2
    assert targets[0]["a"] == 1
