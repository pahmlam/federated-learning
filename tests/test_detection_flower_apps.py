from flwr.app import Context, RecordDict
from PIL import Image

from src.data.detection_data import load_detection_bundle
from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    write_detection_manifest,
)
from src.fl.detection_clientapp import (
    detection_config_from_context,
    select_detection_client,
)
from src.fl.detection_serverapp import _round_config
from src.utils.detection_config import DetectionConfig


def _context(run_config=None, node_config=None, node_id=0):
    return Context(
        run_id=1,
        node_id=node_id,
        node_config=node_config or {},
        state=RecordDict(),
        run_config=run_config or {},
    )


def _make_bundle(tmp_path, sites=("site-a", "site-b")):
    (tmp_path / "images").mkdir()
    (tmp_path / "voc_labels").mkdir()
    spec = {f"h{i}": ["helmet"] for i in range(4)}
    spec.update({f"v{i}": ["safety-vest"] for i in range(4)})
    for stem, classes in spec.items():
        Image.new("RGB", (48, 48), color=(20, 50, 70)).save(
            tmp_path / "images" / f"{stem}.png"
        )
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
        samples,
        sites=list(sites),
        per_site=4,
        val_fraction=0.5,
        seed=2026,
    )
    manifest = tmp_path / "manifest.csv"
    write_detection_manifest(rows, manifest)
    return load_detection_bundle(manifest, tmp_path, image_size=64), manifest


def test_client_selection_uses_single_shard_without_partition(tmp_path):
    bundle, _ = _make_bundle(tmp_path, sites=("site-a",))
    client = select_detection_client(_context(node_id=99), bundle)
    assert client.client_label == "site-a"


def test_client_selection_supports_explicit_client_id(tmp_path):
    bundle, _ = _make_bundle(tmp_path)
    client = select_detection_client(_context(node_config={"client-id": "site-b"}), bundle)
    assert client.client_label == "site-b"


def test_client_selection_falls_back_to_partition_id(tmp_path):
    bundle, _ = _make_bundle(tmp_path)
    client = select_detection_client(_context(node_config={"partition-id": 3}), bundle)
    assert client.client_label == "site-b"


def test_node_config_overrides_data_location(tmp_path):
    _, manifest = _make_bundle(tmp_path)
    context = _context(
        run_config={"manifest-path": "wrong.csv", "root-dir": "wrong"},
        node_config={"manifest-path": str(manifest), "data-root": str(tmp_path)},
    )
    config = detection_config_from_context(context)
    assert config.manifest_path == str(manifest)
    assert config.root_dir == str(tmp_path)


def test_server_round_config_contains_detection_hyperparameters():
    config = DetectionConfig(batch_size=1, lr=0.123, pretrained=False)
    record = _round_config(config)
    assert record["batch_size"] == 1
    assert record["lr"] == 0.123
    assert record["score_threshold"] == config.score_threshold
