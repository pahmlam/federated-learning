from PIL import Image

from src.data.detection_data import load_detection_bundle
from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    write_detection_manifest,
)
from src.training.detection_baselines import (
    run_detection_centralized,
    run_detection_local_only,
)
from src.utils.detection_config import DetectionConfig


def _tiny_bundle(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "voc_labels").mkdir()
    spec = {f"h{i}": ["helmet"] for i in range(4)}
    spec.update({f"v{i}": ["safety-vest"] for i in range(4)})
    for stem, classes in spec.items():
        Image.new("RGB", (48, 48), color=(30, 60, 90)).save(tmp_path / "images" / f"{stem}.png")
        objects = "".join(
            f"<object><name>{name}</name><bndbox>"
            f"<xmin>2</xmin><ymin>2</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>"
            for name in classes
        )
        (tmp_path / "voc_labels" / f"{stem}.xml").write_text(
            f"<annotation>{objects}</annotation>", encoding="utf-8"
        )
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")
    rows = generate_detection_manifest_rows(
        samples, sites=["site-a", "site-b"], per_site=4, val_fraction=0.5, seed=2026
    )
    manifest = tmp_path / "m.csv"
    write_detection_manifest(rows, manifest)
    return load_detection_bundle(manifest, tmp_path, image_size=64)


def _smoke_config():
    return DetectionConfig(
        num_clients=2,
        image_size=64,
        batch_size=1,
        local_epochs=1,
        centralized_epochs=1,
        device="cpu",
        pretrained=False,
    )


def test_centralized_baseline_reports_global_and_per_client(tmp_path):
    bundle = _tiny_bundle(tmp_path)
    result = run_detection_centralized(_smoke_config(), bundle)
    assert result["mode"] == "centralized"
    assert "map_50" in result["global"]
    assert len(result["per_client"]) == 2
    assert result["update_size_bytes"] > 0
    assert result["training_time_sec"] >= 0


def test_local_only_baseline_reports_per_client_map(tmp_path):
    bundle = _tiny_bundle(tmp_path)
    result = run_detection_local_only(_smoke_config(), bundle)
    assert result["mode"] == "local-only"
    assert len(result["per_client"]) == 2
    for record in result["per_client"]:
        assert "map_50" in record and "num_examples" in record
