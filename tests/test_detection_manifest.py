import pytest
from PIL import Image

from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    summarize_detection_rows,
    write_detection_manifest,
)


def _make_dataset(root, spec):
    """spec: dict stem -> list of class names. Creates images/ + voc_labels/."""
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


def test_collect_keeps_only_ppe_positive_images(tmp_path):
    _make_dataset(
        tmp_path,
        {
            "a": ["helmet", "person"],
            "b": ["safety-vest"],
            "c": ["person", "tools"],  # no PPE -> excluded
        },
    )
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")
    stems = {s.stem for s in samples}
    assert stems == {"a", "b"}
    helmet_sample = next(s for s in samples if s.stem == "a")
    assert "helmet" in helmet_sample.ppe_classes
    assert "person" not in helmet_sample.ppe_classes  # non-PPE filtered out


def test_generate_rows_are_leakage_free_and_skewed(tmp_path):
    spec = {f"h{i}": ["helmet"] for i in range(3)}
    spec.update({f"v{i}": ["safety-vest"] for i in range(2)})
    spec.update({f"g{i}": ["gloves"] for i in range(2)})
    _make_dataset(tmp_path, spec)
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")

    rows = generate_detection_manifest_rows(
        samples,
        sites=["site-a", "site-b", "site-c"],
        per_site=2,
        val_fraction=0.5,
        seed=2026,
    )

    # No image stem appears under more than one site (leakage-free).
    by_site: dict[str, set[str]] = {}
    for row in rows:
        stem = row.sample_id.rsplit("_", 1)[-1]
        by_site.setdefault(row.client_id, set()).add(stem)
    all_stems = [stem for stems in by_site.values() for stem in stems]
    assert len(all_stems) == len(set(all_stems))

    # site-a is helmet-focused -> its images are the helmet ones.
    assert all(stem.startswith("h") for stem in by_site["site-a"])

    # Each site has a train and a val split.
    summary = summarize_detection_rows(rows)["per_site"]
    for site in ("site-a", "site-b", "site-c"):
        assert summary[site]["train"] >= 1 and summary[site]["val"] >= 1


def test_generate_rows_raise_when_insufficient_samples(tmp_path):
    _make_dataset(tmp_path, {"h0": ["helmet"]})
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")
    with pytest.raises(ValueError, match="not enough samples"):
        generate_detection_manifest_rows(
            samples, sites=["site-a"], per_site=2, val_fraction=0.5, seed=1
        )


def test_write_manifest_has_expected_columns(tmp_path):
    _make_dataset(tmp_path, {f"h{i}": ["helmet"] for i in range(4)})
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")
    rows = generate_detection_manifest_rows(
        samples, sites=["site-a"], per_site=4, val_fraction=0.25, seed=1
    )
    out = tmp_path / "m.csv"
    write_detection_manifest(rows, out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == "sample_id,image_path,voc_path,client_id,split"
