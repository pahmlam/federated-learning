from PIL import Image

from src.data.detection_data import load_detection_bundle
from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    write_detection_manifest,
)


def _make_root(root, spec):
    (root / "images").mkdir()
    (root / "voc_labels").mkdir()
    for stem, classes in spec.items():
        Image.new("RGB", (40, 40)).save(root / "images" / f"{stem}.png")
        objects = "".join(
            f"<object><name>{name}</name><bndbox>"
            f"<xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>"
            for name in classes
        )
        (root / "voc_labels" / f"{stem}.xml").write_text(
            f"<annotation>{objects}</annotation>", encoding="utf-8"
        )


def _build_manifest(tmp_path):
    spec = {f"h{i}": ["helmet"] for i in range(4)}
    spec.update({f"v{i}": ["safety-vest"] for i in range(4)})
    _make_root(tmp_path, spec)
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")
    rows = generate_detection_manifest_rows(
        samples,
        sites=["site-a", "site-b"],
        per_site=4,
        val_fraction=0.5,
        seed=2026,
    )
    manifest = tmp_path / "manifest.csv"
    write_detection_manifest(rows, manifest)
    return manifest


def test_bundle_has_clients_pooled_and_correct_class_count(tmp_path):
    manifest = _build_manifest(tmp_path)
    bundle = load_detection_bundle(manifest, tmp_path, image_size=64)

    assert [c.client_label for c in bundle.clients] == ["site-a", "site-b"]
    assert bundle.num_classes == 9
    # pooled = sum of per-client splits
    assert len(bundle.pooled_train) == sum(len(c.train) for c in bundle.clients)
    assert len(bundle.pooled_val) == sum(len(c.val) for c in bundle.clients)


def test_bundle_histogram_reflects_site_skew(tmp_path):
    manifest = _build_manifest(tmp_path)
    bundle = load_detection_bundle(manifest, tmp_path, image_size=64)
    site_a = next(c for c in bundle.clients if c.client_label == "site-a")
    # site-a is helmet-focused -> helmet presence > safety-vest presence.
    assert site_a.label_histogram["helmet"] >= site_a.label_histogram["safety-vest"]


def test_bundle_dataset_items_are_loadable(tmp_path):
    manifest = _build_manifest(tmp_path)
    bundle = load_detection_bundle(manifest, tmp_path, image_size=64)
    image, target = bundle.clients[0].train[0]
    assert image.shape[0] == 3
    assert "boxes" in target and "labels" in target
