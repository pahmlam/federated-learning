import numpy as np
from PIL import Image

from src.data.detection_data import load_detection_bundle
from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    write_detection_manifest,
)
from src.fl.detection_federated import federated_average, run_detection_federated
from src.utils.detection_config import DetectionConfig


def test_federated_average_is_weighted_mean():
    client_a = [np.array([0.0, 0.0]), np.array([[1.0]])]
    client_b = [np.array([2.0, 4.0]), np.array([[5.0]])]
    averaged = federated_average([client_a, client_b], [1.0, 3.0])
    assert np.allclose(averaged[0], [1.5, 3.0])  # (1*0+3*2)/4, (1*0+3*4)/4
    assert np.allclose(averaged[1], [[4.0]])  # (1*1+3*5)/4


def _tiny_bundle(tmp_path, sites=("site-a", "site-b")):
    (tmp_path / "images").mkdir()
    (tmp_path / "voc_labels").mkdir()
    sample_count = len(sites) * 4
    spec = {
        f"sample{i}": ["helmet"] if i % 2 == 0 else ["safety-vest"]
        for i in range(sample_count)
    }
    for stem, classes in spec.items():
        Image.new("RGB", (48, 48), color=(20, 50, 70)).save(tmp_path / "images" / f"{stem}.png")
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
        samples, sites=list(sites), per_site=4, val_fraction=0.5, seed=2026
    )
    manifest = tmp_path / "m.csv"
    write_detection_manifest(rows, manifest)
    return load_detection_bundle(manifest, tmp_path, image_size=64)


def test_run_detection_federated_smoke(tmp_path):
    bundle = _tiny_bundle(tmp_path)
    config = DetectionConfig(
        num_clients=2,
        image_size=64,
        batch_size=1,
        local_epochs=1,
        num_rounds=1,
        device="cpu",
        pretrained=False,
    )
    result = run_detection_federated(config, bundle)
    assert result["mode"] == "federated"
    assert result["rounds"] == 1
    assert len(result["per_client"]) == 2
    assert "map_50" in result["global"]
    assert len(result["history"]) == 1
    # comm cost = update_size * clients * rounds * 2 (send + receive)
    assert result["communication_cost_bytes"] == result["update_size_bytes"] * 2 * 1 * 2


def test_run_detection_federated_with_edge_profiles_skips_failed_client(tmp_path):
    bundle = _tiny_bundle(tmp_path, sites=("site-a", "site-b", "site-c"))
    config = DetectionConfig(
        num_clients=3,
        image_size=64,
        batch_size=1,
        local_epochs=1,
        num_rounds=1,
        device="cpu",
        pretrained=False,
        edge_profiles=(
            '{"site-a":"fast",'
            '"site-b":{"tier":"slow-edge","batch_size":1,"max_train_samples":1},'
            '"site-c":{"tier":"unreliable","availability_prob":0.0}}'
        ),
    )

    result = run_detection_federated(config, bundle)

    assert result["mode"] == "federated"
    assert len(result["per_client"]) == 3
    assert len(result["history"]) == 1
    history = result["history"][0]
    assert len(history["train_clients"]) == 2
    assert len(history["train_failures"]) == 1
    assert history["train_failures"][0]["client_label"] == "site-c"
    assert "edge_profile_enabled" in history["train_clients"][0]
